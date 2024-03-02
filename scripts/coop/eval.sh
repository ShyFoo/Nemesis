#!/bin/bash
# This bash file is used for the evaluation of domain generalization tasks
# Please ensure that you have the checkpoints of the ImageNet dataset

# custom configs
DATA=/path/to/dataset

SHOTS=16
NCTX=16
CSC=False
CTP=end
CFG="rn50"

TRAINER=$1  # CoOp, CoOpNemesis, etc.
DATASET=$2  # imagenet_a, imagenet_r, imagenet_sketch, imagenetv2

for SEED in 1 2 3
do
    DIR=output/imagenet/${TRAINER}/${CFG}_${SHOTS}shots/nctx${NCTX}_csc${CSC}_ctp${CTP}/seed${SEED}
    OUTDIR=output_dg/evaluation/${DATASET}/${TRAINER}/seed${SEED}
    LOG_FILE="${OUTDIR}/log.txt"
    if [ -d "$OUTDIR" ] && [ -f "${LOG_FILE}" ] && tail -n 2 "${LOG_FILE}" | grep -q "macro_f1"; then
            echo "Oops! The results exist at ${OUTDIR} (so skip this job)"
    else
        # if the directory exists but the task was interrupted, remove all files and start over
        if [ -d "$OUTDIR" ]; then
            rm -r ${OUTDIR}/*
        fi
        python train.py \
        --root ${DATA} \
        --seed ${SEED} \
        --trainer ${TRAINER} \
        --dataset-config-file configs/datasets/${DATASET}.yaml \
        --config-file configs/trainers/${TRAINER}/${CFG}.yaml \
        --output-dir ${OUTDIR} \
        --model-dir ${DIR} \
        --load-epoch 50 \
        --eval-only \
        TRAINER.CoOp.N_CTX ${NCTX} \
        TRAINER.CoOp.CSC ${CSC} \
        TRAINER.CoOp.CLASS_TOKEN_POSITION ${CTP} \
        DATASET.NUM_SHOTS ${SHOTS}
    fi
done
