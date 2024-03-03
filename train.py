import argparse
import pdb

import torch

from dassl.utils import setup_logger, set_random_seed, collect_env_info
from dassl.config import get_cfg_default
from dassl.engine import build_trainer

# custom functions
import datasets.oxford_pets
import datasets.oxford_flowers
import datasets.fgvc_aircraft
import datasets.dtd
import datasets.eurosat
import datasets.stanford_cars
import datasets.food101
import datasets.sun397
import datasets.caltech101
import datasets.ucf101
import datasets.imagenet

import datasets.imagenet_sketch
import datasets.imagenetv2
import datasets.imagenet_a
import datasets.imagenet_r

import trainers.coop
import trainers.coop_crt
import trainers.coop_nemesis
import trainers.plot
import trainers.plot_nemesis


def print_args(args, cfg):
    print("***************")
    print("** Arguments **")
    print("***************")
    optkeys = list(args.__dict__.keys())
    optkeys.sort()
    for key in optkeys:
        print("{}: {}".format(key, args.__dict__[key]))
    print("************")
    print("** Config **")
    print("************")
    print(cfg)


def reset_cfg(cfg, args):
    if args.root:
        cfg.DATASET.ROOT = args.root

    if args.output_dir:
        cfg.OUTPUT_DIR = args.output_dir

    if args.resume:
        cfg.RESUME = args.resume

    if args.seed:
        cfg.SEED = args.seed

    if args.source_domains:
        cfg.DATASET.SOURCE_DOMAINS = args.source_domains

    if args.target_domains:
        cfg.DATASET.TARGET_DOMAINS = args.target_domains

    if args.transforms:
        cfg.INPUT.TRANSFORMS = args.transforms

    if args.trainer:
        cfg.TRAINER.NAME = args.trainer

    if args.backbone:
        cfg.MODEL.BACKBONE.NAME = args.backbone

    if args.head:
        cfg.MODEL.HEAD.NAME = args.head

    if args.corrupt_type:
        cfg.POS = args.corrupt_position
        cfg.WEIGHT = args.crt_factor

    cfg.TEST.DO_TEST = args.do_test


def extend_cfg(cfg):
    """
    Add new config variables.

    E.g.
        from yacs.config import CfgNode as CN
        cfg.TRAINER.MY_MODEL = CN()
        cfg.TRAINER.MY_MODEL.PARAM_A = 1.
        cfg.TRAINER.MY_MODEL.PARAM_B = 0.5
        cfg.TRAINER.MY_MODEL.PARAM_C = False
    """
    from yacs.config import CfgNode as CN

    cfg.TRAINER.CoOp = CN()
    cfg.TRAINER.CoOp.N_CTX = 16  # number of context vectors
    cfg.TRAINER.CoOp.CSC = False  # class-specific context
    cfg.TRAINER.CoOp.CTX_INIT = ""  # initialization words
    cfg.TRAINER.CoOp.PREC = "fp16"  # fp16, fp32, amp
    cfg.TRAINER.CoOp.CLASS_TOKEN_POSITION = "end"  # 'middle' or 'end' or 'front'

    cfg.TRAINER.CoOpCRT = CN()
    cfg.TRAINER.CoOpCRT.N_CTX = 16  # number of context vectors
    cfg.TRAINER.CoOpCRT.CSC = False  # class-specific context
    cfg.TRAINER.CoOpCRT.CTX_INIT = ""  # initialization words
    cfg.TRAINER.CoOpCRT.PREC = "fp16"  # fp16, fp32, amp
    cfg.TRAINER.CoOpCRT.CLASS_TOKEN_POSITION = "end"  # 'middle' or 'end' or 'front'

    cfg.TRAINER.CoOpNemesis = CN()
    cfg.TRAINER.CoOpNemesis.N_CTX = 16  # number of context vectors
    cfg.TRAINER.CoOpNemesis.CSC = False  # class-specific context
    cfg.TRAINER.CoOpNemesis.CTX_INIT = ""  # initialization words
    cfg.TRAINER.CoOpNemesis.PREC = "fp16"  # fp16, fp32, amp
    cfg.TRAINER.CoOpNemesis.CLASS_TOKEN_POSITION = "end"  # 'middle' or 'end' or 'front'
    cfg.TRAINER.CoOpNemesis.A = None  # the weight of normalization loss
    cfg.TRAINER.CoOpNemesis.NT = "two"  # norm type, "one", "two", "inf"
    cfg.TRAINER.CoOpNemesis.NP = None  # the number of position for currupting during training
    cfg.TRAINER.CoOpNemesis.CW = None  # corrupted weight during training
    cfg.TRAINER.CoOpNemesis.BETA = None  # beta coefficient for combining two normalization losses

    cfg.TRAINER.PLOT = CN()
    cfg.TRAINER.PLOT.N_CTX = 16  # number of context vectors
    cfg.TRAINER.PLOT.CSC = False  # class-specific context
    cfg.TRAINER.PLOT.CTX_INIT = ""  # initialization words
    cfg.TRAINER.PLOT.PREC = "fp16"  # fp16, fp32, amp
    cfg.TRAINER.PLOT.CLASS_TOKEN_POSITION = "end"  # 'middle' or 'end' or 'front'
    cfg.TRAINER.PLOT.N = 4

    cfg.TRAINER.PLOTNemesis = CN()
    cfg.TRAINER.PLOTNemesis.N_CTX = 16  # number of context vectors
    cfg.TRAINER.PLOTNemesis.CSC = False  # class-specific context
    cfg.TRAINER.PLOTNemesis.CTX_INIT = ""  # initialization words
    cfg.TRAINER.PLOTNemesis.PREC = "fp16"  # fp16, fp32, amp
    cfg.TRAINER.PLOTNemesis.CLASS_TOKEN_POSITION = "end"  # 'middle' or 'end' or 'front'
    cfg.TRAINER.PLOTNemesis.N = 4
    cfg.TRAINER.PLOTNemesis.A = None  # the weight of normalization loss
    cfg.TRAINER.PLOTNemesis.NT = "two"  # norm type, "one", "two", "inf"
    cfg.TRAINER.PLOTNemesis.NP = None  # the number of position for currupting during training
    cfg.TRAINER.PLOTNemesis.CW = None  # corrupted weight during training
    cfg.TRAINER.PLOTNemesis.BETA = None  # beta coefficient for combining two normalization losses

    cfg.DATASET.SUBSAMPLE_CLASSES = "all"  # all, base or new


def setup_cfg(args):
    cfg = get_cfg_default()
    extend_cfg(cfg)

    # 1. From the dataset config file
    if args.dataset_config_file:
        cfg.merge_from_file(args.dataset_config_file)

    # 2. From the method config file
    if args.config_file:
        cfg.merge_from_file(args.config_file)

    # 3. From input arguments
    reset_cfg(cfg, args)

    # 4. From optional input arguments
    cfg.merge_from_list(args.opts)

    if cfg.DATASET.NAME == "ImageNet":
        cfg.OPTIM.MAX_EPOCH = 50

    cfg.freeze()

    return cfg


def main(args):
    cfg = setup_cfg(args)
    if cfg.SEED >= 0:
        print("Setting fixed seed: {}".format(cfg.SEED))
        set_random_seed(cfg.SEED)
    setup_logger(cfg.OUTPUT_DIR)

    if torch.cuda.is_available() and cfg.USE_CUDA:
        torch.backends.cudnn.benchmark = True

    print_args(args, cfg)
    print("Collecting env info ...")
    print("** System info **\n{}\n".format(collect_env_info()))

    trainer = build_trainer(cfg) 

    if args.eval_only and args.corrupt_type:
        if cfg.DATASET.NAME == "ImageNet":
            args.load_epoch = 50
            trainer.load_model(args.model_dir, epoch=args.load_epoch, corrupt_type=args.corrupt_type)
        else:
            trainer.load_model(args.model_dir, epoch=args.load_epoch, corrupt_type=args.corrupt_type)
        trainer.test()
        trainer.customized_test()
    elif args.eval_only and not args.corrupt_type:
        trainer.load_model(args.model_dir, epoch=args.load_epoch)
        trainer.test()
    else:
        trainer.train()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=str, default="", help="path to dataset")
    parser.add_argument("--output-dir", type=str, default="", help="output directory")
    parser.add_argument(
        "--resume",
        type=str,
        default="",
        help="checkpoint directory (from which the training resumes)",
    )
    parser.add_argument(
        "--seed", type=int, default=-1, help="only positive value enables a fixed seed"
    )
    parser.add_argument(
        "--source-domains", type=str, nargs="+", help="source domains for DA/DG"
    )
    parser.add_argument(
        "--target-domains", type=str, nargs="+", help="target domains for DA/DG"
    )
    parser.add_argument(
        "--transforms", type=str, nargs="+", help="data augmentation methods"
    )
    parser.add_argument(
        "--config-file", type=str, default="", help="path to config file"
    )
    parser.add_argument(
        "--dataset-config-file",
        type=str,
        default="",
        help="path to config file for dataset setup",
    )
    parser.add_argument("--trainer", type=str, default="", help="name of trainer")
    parser.add_argument("--backbone", type=str, default="", help="name of CNN backbone")
    parser.add_argument("--head", type=str, default="", help="name of head")
    parser.add_argument("--eval-only", action="store_true", help="evaluation only")
    parser.add_argument(
        "--model-dir",
        type=str,
        default="",
        help="load model from this directory for eval-only mode",
    )
    parser.add_argument(
        "--load-epoch", type=int, help="load model weights at this epoch for evaluation"
    )
    parser.add_argument(
        "opts",
        default=None,
        nargs=argparse.REMAINDER,
        help="modify config options using the command-line",
    )
    parser.add_argument(
        "--corrupt-type", type=str, default=None, choices=["replace", "scale", "original"],
        help="the type of corrupted prompt"
    )
    parser.add_argument(
        "--crt-factor", type=float, default=None,
        help="the corruption weight in corruption experiments (std in replacing operations or scaling factor in rescaling operations)"
    )
    parser.add_argument(
        "--corrupt-position", type=int, default=None,
        help="the corrupt position of prompt in the corrupting experiments"
    )
    parser.add_argument("--do-test", action="store_true", help="do testing during training")
    args = parser.parse_args()
    main(args)
