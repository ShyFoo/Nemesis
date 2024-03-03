#!/bin/bash

# custom config
DATA=/path/to/dataset

SHOTS=16
NCTX=16
CSC=False
CTP=end
CFG="rn50"

TRAINER=$1  # CoOp, CoOpNemesis, etc.
DATASET=$2  # imagenet_a, imagenet_r, imagenet_sketch, imagenetv2
ALPHA=$3  # the scaling weight of the normalization loss
NT=$4  # normalization types ("one", "two", "inf")
NP=$5  # the number of positions for currupting during training: [1, N_CTX], only works when alpha = adaptive
CW=$6  # the threshold of corruption weight, only used when beta != 1.0
BETA=$7  # the beta coefficient for balancing the pan loss and the pun loss (decimals like 0.1, 0.2, ..., 0: only pan loss, 1: only the pun loss)


for SEED in 1 2 3
do
    if [ ${BETA} != 1.0 ]; then
        DIR=output/${DATASET}/${TRAINER}/Beta-${BETA}/${CFG}_${SHOTS}shots/alpha${ALPHA}_cw${CW}_pn${NP}_nctx${NCTX}_csc${CSC}_ctp${CTP}_nt${NT}/seed${SEED}
        OUTDIR=output_dg/${DATASET}/${TRAINER}/Beta-${BETA}/${CFG}_${SHOTS}shots/alpha${ALPHA}_cw${CW}_pn${NP}_nctx${NCTX}_csc${CSC}_ctp${CTP}_nt${NT}/seed${SEED}
    else
        DIR=output/${DATASET}/${TRAINER}/Beta-${BETA}/${CFG}_${SHOTS}shots/alpha${ALPHA}_nctx${NCTX}_csc${CSC}_ctp${CTP}_nt${NT}/seed${SEED}
        OUTDIR=output_dg/${DATASET}/${TRAINER}/Beta-${BETA}/${CFG}_${SHOTS}shots/alpha${ALPHA}_nctx${NCTX}_csc${CSC}_ctp${CTP}_nt${NT}/seed${SEED}
    fi

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
        TRAINER.CoOpNemesis.N_CTX ${NCTX} \
        TRAINER.CoOpNemesis.CSC ${CSC} \
        TRAINER.CoOpNemesis.CLASS_TOKEN_POSITION ${CTP} \
        TRAINER.CoOpNemesis.A ${ALPHA} \
        TRAINER.CoOpNemesis.NT ${NT} \
        TRAINER.CoOpNemesis.NP ${NP} \
        TRAINER.CoOpNemesis.CW ${CW} \
        DATASET.NUM_SHOTS ${SHOTS}
    fi
done
