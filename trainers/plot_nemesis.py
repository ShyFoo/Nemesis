import os.path as osp
import pdb
import random
import math

import torch
import torch.nn as nn
from torch.nn import functional as F
from torch.cuda.amp import GradScaler, autocast

from dassl.engine import TRAINER_REGISTRY, TrainerX
from dassl.metrics import compute_accuracy
from dassl.utils import load_pretrained_weights, load_checkpoint
from dassl.optim import build_optimizer, build_lr_scheduler

from tqdm import tqdm
from clip_plot import clip
from clip_plot.simple_tokenizer import SimpleTokenizer as _Tokenizer

_tokenizer = _Tokenizer()


def load_clip_to_cpu(cfg):
    backbone_name = cfg.MODEL.BACKBONE.NAME
    url = clip._MODELS[backbone_name]
    model_path = clip._download(url)

    try:
        # loading JIT archive
        model = torch.jit.load(model_path, map_location="cpu").eval()
        state_dict = None

    except RuntimeError:
        state_dict = torch.load(model_path, map_location="cpu")

    model = clip.build_model(state_dict or model.state_dict())

    return model


class TextEncoder(nn.Module):
    def __init__(self, clip_model):
        super().__init__()
        self.transformer = clip_model.transformer
        self.positional_embedding = clip_model.positional_embedding
        self.ln_final = clip_model.ln_final
        self.text_projection = clip_model.text_projection
        self.dtype = clip_model.dtype

    def forward(self, prompts, tokenized_prompts):
        x = prompts + self.positional_embedding.type(self.dtype)

        x = x.permute(1, 0, 2)  # NLD -> LND
        x = self.transformer(x)
        x = x.permute(1, 0, 2)  # LND -> NLD
        x = self.ln_final(x).type(self.dtype)

        x = x[torch.arange(x.shape[0]), tokenized_prompts.argmax(dim=-1)] @ self.text_projection

        return x


class PromptLearner(nn.Module):
    def __init__(self, cfg, classnames, clip_model):
        super().__init__()
        n_cls = len(classnames)
        n_ctx = cfg.TRAINER.PLOTNemesis.N_CTX
        ctx_init = cfg.TRAINER.PLOTNemesis.CTX_INIT
        dtype = clip_model.dtype
        ctx_dim = clip_model.ln_final.weight.shape[0]
        clip_imsize = clip_model.visual.input_resolution
        cfg_imsize = cfg.INPUT.SIZE[0]
        self.N = cfg.TRAINER.PLOTNemesis.N
        assert cfg_imsize == clip_imsize, f"cfg_imsize ({cfg_imsize}) must equal to clip_imsize ({clip_imsize})"

        if ctx_init:
            # use given words to initialize context vectors
            ctx_init = ctx_init.replace("_", " ")
            n_ctx = len(ctx_init.split(" "))
            prompt = clip.tokenize(ctx_init)
            with torch.no_grad():
                embedding = clip_model.token_embedding(prompt).type(dtype)
            ctx_vectors = embedding[0, 1: 1 + n_ctx, :]
            prompt_prefix = ctx_init

        else:
            # random initialization
            if cfg.TRAINER.PLOTNemesis.CSC:
                print("Initializing class-specific contexts")
                ctx_vectors = torch.empty(n_cls, self.N, n_ctx, ctx_dim, dtype=dtype)
            else:
                print("Initializing a generic context")
                ctx_vectors = torch.empty(self.N, n_ctx, ctx_dim, dtype=dtype)

            nn.init.normal_(ctx_vectors, std=0.02)  # define the prompt to be trained
            prompt_prefix = " ".join(["X"] * n_ctx)

        print(f'Initial context: "{prompt_prefix}"')
        print(f"Number of context words (tokens): {n_ctx}")

        self.ctx = nn.Parameter(ctx_vectors)  # to be optimized

        classnames = [name.replace("_", " ") for name in classnames]
        name_lens = [len(_tokenizer.encode(name)) for name in classnames]
        prompts = [prompt_prefix + " " + name + "." for name in classnames]

        tokenized_prompts = torch.cat([clip.tokenize(p) for p in prompts])
        tokenized_prompts = tokenized_prompts.repeat(self.N, 1)

        with torch.no_grad():
            embedding = clip_model.token_embedding(tokenized_prompts).type(dtype)

        # These token vectors will be saved when in save_model(),
        # but they should be ignored in load_model() as we want to use
        # those computed using the current class names
        self.register_buffer("token_prefix", embedding[:, :1, :])  # SOS
        self.register_buffer("token_suffix", embedding[:, 1 + n_ctx:, :])  # CLS, EOS

        self.n_cls = n_cls
        self.n_ctx = n_ctx
        self.tokenized_prompts = tokenized_prompts  # torch.Tensor
        self.name_lens = name_lens
        self.class_token_position = cfg.TRAINER.PLOTNemesis.CLASS_TOKEN_POSITION

    def forward(self):

        ctx = self.ctx
        if ctx.dim() == 3:
            ctx = ctx.unsqueeze(0).expand(self.n_cls, -1, -1, -1)

        ctx = ctx.permute(1, 0, 2, 3)
        ctx = ctx.contiguous().view(self.N * self.n_cls, self.n_ctx, ctx.shape[3])

        prefix = self.token_prefix
        suffix = self.token_suffix

        if self.class_token_position == "end":
            prompts = torch.cat(
                [
                    prefix,  # (n_cls, 1, dim)
                    ctx,  # (n_cls, n_ctx, dim)
                    suffix,  # (n_cls, *, dim)
                ],
                dim=1,
            )

        elif self.class_token_position == "middle":
            half_n_ctx = self.n_ctx // 2
            prompts = []
            for i in range(self.n_cls):
                name_len = self.name_lens[i]
                prefix_i = prefix[i: i + 1, :, :]
                class_i = suffix[i: i + 1, :name_len, :]
                suffix_i = suffix[i: i + 1, name_len:, :]
                ctx_i_half1 = ctx[i: i + 1, :half_n_ctx, :]
                ctx_i_half2 = ctx[i: i + 1, half_n_ctx:, :]
                prompt = torch.cat(
                    [
                        prefix_i,  # (1, 1, dim)
                        ctx_i_half1,  # (1, n_ctx//2, dim)
                        class_i,  # (1, name_len, dim)
                        ctx_i_half2,  # (1, n_ctx//2, dim)
                        suffix_i,  # (1, *, dim)
                    ],
                    dim=1,
                )
                prompts.append(prompt)
            prompts = torch.cat(prompts, dim=0)

        elif self.class_token_position == "front":
            prompts = []
            for i in range(self.n_cls):
                name_len = self.name_lens[i]
                prefix_i = prefix[i: i + 1, :, :]
                class_i = suffix[i: i + 1, :name_len, :]
                suffix_i = suffix[i: i + 1, name_len:, :]
                ctx_i = ctx[i: i + 1, :, :]
                prompt = torch.cat(
                    [
                        prefix_i,  # (1, 1, dim)
                        class_i,  # (1, name_len, dim)
                        ctx_i,  # (1, n_ctx, dim)
                        suffix_i,  # (1, *, dim)
                    ],
                    dim=1,
                )
                prompts.append(prompt)
            prompts = torch.cat(prompts, dim=0)

        else:
            raise ValueError

        return prompts

    @torch.no_grad()
    def inference_batch(self, weight, pos_list, num_proxy):
        ctx = self.ctx
        expand_len = len(pos_list) + 1

        if ctx.dim() == 3:
            ctx = ctx.unsqueeze(0).expand(self.n_cls, -1, -1, -1)

        ctx = ctx.permute(1, 0, 2, 3)
        ctx = ctx.contiguous().view(num_proxy * self.n_cls, self.n_ctx, ctx.shape[-1])
        ctx = ctx.repeat(expand_len, 1, 1)

        for i, j in enumerate(pos_list):
            ctx[(i + 1) * num_proxy * self.n_cls: (i + 2) * num_proxy * self.n_cls, j, :] *= weight

        prefix = self.token_prefix.repeat(expand_len, 1, 1)
        suffix = self.token_suffix.repeat(expand_len, 1, 1)

        if self.class_token_position == "end":
            prompts = torch.cat(
                [
                    prefix,  # (n_cls, 1, dim)
                    ctx,  # (n_cls, n_ctx, dim)
                    suffix,  # (n_cls, *, dim)
                ],
                dim=1,
            )

        else:
            raise ValueError

        return prompts


class CustomCLIP(nn.Module):
    def __init__(self, cfg, classnames, clip_model):
        super().__init__()
        self.n_cls = len(classnames)
        self.prompt_learner = PromptLearner(cfg, classnames, clip_model)
        self.tokenized_prompts = self.prompt_learner.tokenized_prompts
        self.image_encoder = clip_model.visual
        self.text_encoder = TextEncoder(clip_model)
        self.logit_scale = clip_model.logit_scale
        self.dtype = clip_model.dtype
        self.device = torch.device("cuda")
        self.N = cfg.TRAINER.PLOTNemesis.N
        self.dataset = cfg.DATASET.NAME
        self.use_uniform = True
        self.eps = 0.1
        self.max_iter = 100

    def Sinkhorn(self, K, u, v):
        r = torch.ones_like(u)
        c = torch.ones_like(v)
        thresh = 1e-2
        for i in range(self.max_iter):
            r0 = r
            r = u / torch.matmul(K, c.unsqueeze(-1)).squeeze(-1)
            c = v / torch.matmul(K.permute(0, 2, 1).contiguous(), r.unsqueeze(-1)).squeeze(-1)
            err = (r - r0).abs().mean()
            if err.item() < thresh:
                break

        T = torch.matmul(r.unsqueeze(-1), c.unsqueeze(-2)) * K
        return T

    def forward(self, image):
        b = image.shape[0]
        image_features = self.image_encoder(image.type(self.dtype))
        M = image_features.shape[0]
        self.d = image_features.shape[-1]

        prompts = self.prompt_learner()
        tokenized_prompts = self.tokenized_prompts
        text_features = self.text_encoder(prompts, tokenized_prompts)
        text_features = text_features.contiguous().view(self.N, self.n_cls, self.d)

        image_features = F.normalize(image_features, dim=2)
        text_features = F.normalize(text_features, dim=2)

        sim = torch.einsum('mbd, ncd -> mnbc', image_features, text_features).contiguous()
        sim = sim.view(M, self.N, b * self.n_cls)
        sim = sim.permute(2, 0, 1)
        wdist = 1.0 - sim
        xx = torch.zeros(b * self.n_cls, M, dtype=sim.dtype, device=sim.device).fill_(1. / M)
        yy = torch.zeros(b * self.n_cls, self.N, dtype=sim.dtype, device=sim.device).fill_(1. / self.N)

        with torch.no_grad():
            KK = torch.exp(-wdist / self.eps)
            T = self.Sinkhorn(KK, xx, yy)
        if torch.isnan(T).any():
            return None

        sim_op = torch.sum(T * sim, dim=(1, 2))
        sim_op = sim_op.contiguous().view(b, self.n_cls)

        logit_scale = self.logit_scale.exp()
        logits = logit_scale * sim_op
        return logits

    @torch.no_grad()
    def inference_batch(self, image, weight, pos_list):
        b = image.shape[0]
        expand_len = len(pos_list) + 1

        image_features = self.image_encoder(image.type(self.dtype))
        M = image_features.shape[0]
        self.d = image_features.shape[-1]

        prompts = self.prompt_learner.inference_batch(weight, pos_list, self.N)
        tokenized_prompts = self.tokenized_prompts.repeat(expand_len, 1)
        text_features = self.text_encoder(prompts, tokenized_prompts)

        text_features = text_features.contiguous().view(expand_len * self.N, self.n_cls, self.d)

        image_features = F.normalize(image_features, dim=2)
        text_features = F.normalize(text_features, dim=2)

        sim = torch.einsum('mbd, ncd -> mnbc', image_features, text_features).contiguous()
        sim = sim.view(M, expand_len, self.N, b * self.n_cls)

        sim = sim.permute(1, 3, 0, 2)
        wdist = 1.0 - sim
        xx = torch.zeros(b * self.n_cls, M, dtype=sim.dtype, device=sim.device).fill_(1. / M)
        yy = torch.zeros(b * self.n_cls, self.N, dtype=sim.dtype, device=sim.device).fill_(1. / self.N)

        logits_list = torch.tensor([], dtype=self.dtype, device=self.device)
        for i in range(expand_len):
            with torch.no_grad():
                KK = torch.exp(-wdist[i] / self.eps)
                T = self.Sinkhorn(KK, xx, yy)
            if torch.isnan(T).any():
                return None

            sim_op = torch.sum(T * sim[i], dim=(1, 2))
            sim_op = sim_op.contiguous().view(b, self.n_cls)

            logit_scale = self.logit_scale.exp()
            logits = logit_scale * sim_op
            logits_list = torch.cat((logits_list, logits.unsqueeze(0)), dim=0)
        return logits_list


@TRAINER_REGISTRY.register()
class PLOTNemesis(TrainerX):
    """
    It is based on CoOp.
    """
    def check_cfg(self, cfg):
        assert cfg.TRAINER.PLOTNemesis.PREC in ["fp16", "fp32", "amp"]

    def build_model(self):
        cfg = self.cfg
        classnames = self.dm.dataset.classnames

        print(f"Loading CLIP (backbone: {cfg.MODEL.BACKBONE.NAME})")
        clip_model = load_clip_to_cpu(cfg)

        if cfg.TRAINER.PLOTNemesis.PREC == "fp32" or cfg.TRAINER.PLOTNemesis.PREC == "amp":
            # CLIP's default precision is fp16
            clip_model.float()

        print("Building custom CLIP")
        self.model = CustomCLIP(cfg, classnames, clip_model)

        print("Turning off gradients in both the image and the text encoder")
        for name, param in self.model.named_parameters():
            if "prompt_learner" not in name:
                param.requires_grad_(False)
        # Double check
        enabled = set()
        for name, param in self.model.named_parameters():
            if param.requires_grad:
                enabled.add(name)
        print(f"Parameters to be updated: {enabled}")

        if cfg.MODEL.INIT_WEIGHTS:
            load_pretrained_weights(self.model.prompt_learner, cfg.MODEL.INIT_WEIGHTS)

        self.model.to(self.device)

        # NOTE: only give prompt_learner to the optimizer
        self.optim = build_optimizer(self.model.prompt_learner, cfg.OPTIM)
        self.sched = build_lr_scheduler(self.optim, cfg.OPTIM)
        self.register_model("prompt_learner", self.model.prompt_learner, self.optim, self.sched)

        self.scaler = GradScaler() if cfg.TRAINER.PLOTNemesis.PREC == "amp" else None

        # Note that multi-gpu training could be slow because CLIP's size is
        # big, which slows down the copy operation in DataParallel
        # device_count = torch.cuda.device_count()
        # if device_count > 1:
        #     print(f"Multiple GPUs detected (n_gpus={device_count}), use all of them!")
        #     self.model = nn.DataParallel(self.model)

    @torch.no_grad()
    def update_alpha(self, image, label, crt_weight):
        # torch.cuda.empty_cache()
        if isinstance(self.cfg.TRAINER.PLOTNemesis.NP, str):
            if "random" in self.cfg.TRAINER.PLOTNemesis.NP:
                pos_num = int(self.cfg.TRAINER.PLOTNemesis.NP.split("random")[-1])
            else:
                raise ValueError
        else:
            pos_num = self.cfg.TRAINER.PLOTNemesis.NP

        if self.cfg.DATASET.NAME == "Food101" or self.cfg.DATASET.NAME == "OxfordPets":
            scale = 50.
        else:
            scale = self.cfg.TRAINER.PLOTNemesis.A

        # initialize alpha to zero
        self.pos_alpha = torch.zeros((self.model.prompt_learner.ctx.size(-2)), device=self.device)

        pos_list = random.sample(range(self.model.prompt_learner.ctx.size(-2)), pos_num)
        output = self.model.inference_batch(image, crt_weight, pos_list)
        _, pred = output.topk(1)
        pred = pred.squeeze()
        ori_accu = torch.sum(pred.eq(label.expand_as(pred)), dim=-1)[0]
        crt_accu = torch.sum(pred.eq(label.expand_as(pred)), dim=-1)[1:].view(len(pos_list), -1)
        # update alpha at a position level
        if isinstance(self.cfg.TRAINER.PLOTNemesis.NP, str):
            if "random" in self.cfg.TRAINER.PLOTNemesis.NP:
                self.pos_alpha[pos_list] = scale
            else:
                raise ValueError
        else:
            self.pos_alpha[pos_list] = (crt_accu > ori_accu).float().squeeze() * scale

    def normalize(self, pos_alpha):
        if torch.count_nonzero(pos_alpha) != 0:
            count = torch.count_nonzero(pos_alpha)
        else:
            count = 1.
        if self.cfg.TRAINER.PLOTNemesis.NT == "two":
            n_loss = (pos_alpha * torch.norm(self.model.prompt_learner.ctx, p=2, dim=-1)).sum() / count

        elif self.cfg.TRAINER.PLOTNemesis.NT == "one":
            n_loss = (pos_alpha * torch.norm(self.model.prompt_learner.ctx, p=1, dim=-1)).sum() / count

        elif self.cfg.TRAINER.PLOTNemesis.NT == "inf":
            n_loss = (pos_alpha * torch.norm(self.model.prompt_learner.ctx, p=torch.inf, dim=-1)).sum() / count

        else:
            n_loss = None
            NotImplementedError("Do not implement this type of normalization")
        return n_loss

    def calculate_norm(self):
        with torch.no_grad():
            if self.cfg.TRAINER.PLOTNemesis.NT == "two":
                matrix_norm = torch.linalg.matrix_norm(self.model.prompt_learner.ctx, ord="fro").mean().item()
                if torch.norm(self.model.prompt_learner.ctx, p=2, dim=-1).dim() > 1:
                    # class-specific or ensemble method
                    vector_norm = torch.norm(self.model.prompt_learner.ctx, p=2, dim=-1).mean(dim=0)
                else:
                    vector_norm = torch.norm(self.model.prompt_learner.ctx, p=2, dim=-1)
            elif self.cfg.TRAINER.PLOTNemesis.NT == "one":
                matrix_norm = torch.linalg.matrix_norm(self.model.prompt_learner.ctx, ord=1).mean().item()
                if torch.norm(self.model.prompt_learner.ctx, p=1, dim=-1).dim() > 1:
                    # class-specific or ensemble method
                    vector_norm = torch.norm(self.model.prompt_learner.ctx, p=1, dim=-1).mean(dim=0)
                else:
                    vector_norm = torch.norm(self.model.prompt_learner.ctx, p=1, dim=-1)
            elif self.cfg.TRAINER.PLOTNemesis.NT == "inf":
                matrix_norm = torch.linalg.matrix_norm(self.model.prompt_learner.ctx, ord=torch.inf).mean().item()
                if torch.norm(self.model.prompt_learner.ctx, p=torch.inf, dim=-1).dim() > 1:
                    # class-specific or ensemble method
                    vector_norm = torch.norm(self.model.prompt_learner.ctx, p=torch.inf, dim=-1).mean(dim=0)
                else:
                    vector_norm = torch.norm(self.model.prompt_learner.ctx, p=torch.inf, dim=-1)
            else:
                matrix_norm, vector_norm = None, None
                NotImplementedError("Do not implement this type of normalization")
        return matrix_norm, vector_norm

    # logistic function
    def omega_schedule(self):
        # growth rate
        k = 0.2
        value = 1 / (1 + math.exp(-k * (self.epoch - 1 / 2 * self.max_epoch)))
        value = 1 - value
        return value

    def forward_backward(self, batch):
        image, label = self.parse_batch_train(batch)
        prec = self.cfg.TRAINER.PLOTNemesis.PREC

        # omega schedule, sometimes achieving better results
        if self.cfg.DATASET.NAME == "Food101" or self.cfg.DATASET.NAME == "OxfordPets":
            omega = 1.
        else:
            omega = self.omega_schedule()

        beta = self.cfg.TRAINER.PLOTNemesis.BETA
        crt_weight = self.cfg.TRAINER.PLOTNemesis.CW

        # only the pun loss
        if beta == 1.:
            self.pos_alpha = self.cfg.TRAINER.PLOTNemesis.A * \
                             torch.ones((self.model.prompt_learner.ctx.size(-2)), device=self.device)
            n_loss = self.normalize(self.pos_alpha)

            if prec == "amp":
                with autocast():
                    output = self.model(image)
                    ce_loss = F.cross_entropy(output, label)
                    total_loss = ce_loss + omega * n_loss
                self.optim.zero_grad()
                self.scaler.scale(total_loss).backward()
                self.scaler.step(self.optim)
                self.scaler.update()
                train_accu = compute_accuracy(output, label)[0].item()
            else:
                output = self.model(image)
                ce_loss = F.cross_entropy(output, label)
                total_loss = ce_loss + omega * n_loss
                self.model_backward_and_update(total_loss)
                train_accu = compute_accuracy(output, label)[0].item()

        # pun + pan
        elif 0. < beta < 1.:
            # pun loss
            self.pos_alpha = self.cfg.TRAINER.PLOTNemesis.A * \
                             torch.ones((self.model.prompt_learner.ctx.size(-2)), device=self.device)
            pun_loss = self.normalize(self.pos_alpha)
            # pan loss
            self.update_alpha(image, label, crt_weight)
            pan_loss = self.normalize(self.pos_alpha)
            n_loss = pun_loss + pan_loss

            if prec == "amp":
                with autocast():
                    output = self.model(image)
                    ce_loss = F.cross_entropy(output, label)
                    total_loss = ce_loss + omega * (beta * pun_loss + (1 - beta) * pan_loss)
                self.optim.zero_grad()
                self.scaler.scale(total_loss).backward()
                self.scaler.step(self.optim)
                self.scaler.update()
                train_accu = compute_accuracy(output, label)[0].item()
            else:
                output = self.model(image)
                ce_loss = F.cross_entropy(output, label)
                total_loss = ce_loss + omega * (beta * pun_loss + (1 - beta) * pan_loss)
                self.model_backward_and_update(total_loss)
                train_accu = compute_accuracy(output, label)[0].item()

        # only the pan loss
        elif beta == 0.:
            self.update_alpha(image, label, crt_weight)
            n_loss = self.normalize(self.pos_alpha)

            if prec == "amp":
                with autocast():
                    output = self.model(image)
                    ce_loss = F.cross_entropy(output, label)
                    total_loss = ce_loss + omega * n_loss
                self.optim.zero_grad()
                self.scaler.scale(total_loss).backward()
                self.scaler.step(self.optim)
                self.scaler.update()
                train_accu = compute_accuracy(output, label)[0].item()
            else:
                output = self.model(image)
                ce_loss = F.cross_entropy(output, label)
                total_loss = ce_loss + omega * n_loss
                self.model_backward_and_update(total_loss)
                train_accu = compute_accuracy(output, label)[0].item()

        else:
            raise ValueError

        matrix_norm, vector_norm = self.calculate_norm()

        loss_summary = {
            "ce_loss": ce_loss.item(),
            "n_loss": n_loss.item(),
            "train_acc": train_accu,
            "matrix_norm": matrix_norm
        }

        v_norm_dict = {}
        for i in range(len(vector_norm)):
            key = f"v{i}_norm"
            value = vector_norm[i].item()
            v_norm_dict[key] = value
        loss_summary.update(v_norm_dict)

        alpha_dict = {}
        for i in range(len(self.pos_alpha)):
            key = f"alpha_{i}"
            value = self.pos_alpha[i].item()
            alpha_dict[key] = value
        loss_summary.update(alpha_dict)

        if (self.batch_idx + 1) == self.num_batches:
            self.update_lr()

        return loss_summary

    def after_epoch(self):
        last_epoch = (self.epoch + 1) == self.max_epoch
        meet_checkpoint_freq = (
            (self.epoch + 1) % self.cfg.TRAIN.CHECKPOINT_FREQ == 0
            if self.cfg.TRAIN.CHECKPOINT_FREQ > 0 else False
        )

        correct = 0
        total = 0
        if self.cfg.TEST.DO_TEST:
            with torch.no_grad():
                for batch in tqdm(self.test_loader):
                    input, label = self.parse_batch_test(batch)
                    output = self.model(input)
                    _, pred = output.topk(1)
                    pred = pred.squeeze()
                    correct += torch.sum(pred.eq(label.expand_as(pred))).item()
                    total += len(label)
                test_accu = correct / total * 100
            info = []
            info += [f"epoch [{self.epoch + 1}/{self.max_epoch}]"]
            info += [f"test accu ({test_accu:.4f})"]
            print(" ".join(info))
            self.write_scalar("train/test_accu", test_accu, self.epoch + 1)

        if meet_checkpoint_freq or last_epoch:
            self.save_model(self.epoch, self.output_dir)

    def parse_batch_train(self, batch):
        input = batch["img"]
        label = batch["label"]
        input = input.to(self.device)
        label = label.to(self.device)
        return input, label

    def load_model(self, directory, epoch=None):
        if not directory:
            print("Note that load_model() is skipped as no pretrained model is given")
            return

        names = self.get_model_names()

        # By default, the best model is loaded
        model_file = "model-best.pth.tar"

        if epoch is not None:
            model_file = "model.pth.tar-" + str(epoch)

        for name in names:
            model_path = osp.join(directory, name, model_file)

            if not osp.exists(model_path):
                raise FileNotFoundError('Model not found at "{}"'.format(model_path))

            checkpoint = load_checkpoint(model_path)
            state_dict = checkpoint["state_dict"]
            epoch = checkpoint["epoch"]

            # Ignore fixed token vectors
            if "token_prefix" in state_dict:
                del state_dict["token_prefix"]

            if "token_suffix" in state_dict:
                del state_dict["token_suffix"]

            print("Loading weights to {} " 'from "{}" (epoch = {})'.format(name, model_path, epoch))
            self._models[name].load_state_dict(state_dict, strict=False)
