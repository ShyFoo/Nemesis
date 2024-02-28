#!/bin/bash

# custom configs
DATA=/path/to/dataset
TRAINER=CoOp

# input configs
DATASET=$1  # imagenet, sun397, caltech101, eurosat, food101, oxford_flowers, stanford_cars, ucf101, dtd, fgvc_aircraft, oxford_pets
CFG=$2  # config file
CTP=$3  # class token position (end or middle)
NCTX=$4  # number of context tokens
SHOTS=$5  # number of shots (1, 2, 4, 8, 16)
CSC=$6  # class-specific context (False or True)

for SEED in 1 2 3
do
    LOGDIR=output/${DATASET}/${TRAINER}/${CFG}_${SHOTS}shots/nctx${NCTX}_csc${CSC}_ctp${CTP}/seed${SEED}
    LOG_FILE="${LOGDIR}/log.txt"
    if [ -d "$DIR" ] && [ -f "${LOG_FILE}" ] && tail -n 2 "${LOG_FILE}" | grep -q "Elapsed"; then
        echo "Oops! The results exist at ${LOGDIR} (so skip this job)"
    else
        # if the directory exists but the task was interrupted, remove all files and start over
        if [ -d "$DIR" ]; then
            rm -r ${LOGDIR}/*
        fi
        python train.py \
        --root ${DATA} \
        --seed ${SEED} \
        --trainer ${TRAINER} \
        --dataset-config-file configs/datasets/${DATASET}.yaml \
        --config-file configs/trainers/${TRAINER}/${CFG}.yaml \
        --output-dir ${LOGDIR} \
        TRAINER.CoOp.N_CTX ${NCTX} \
        TRAINER.CoOp.CSC ${CSC} \
        TRAINER.CoOp.CLASS_TOKEN_POSITION ${CTP} \
        DATASET.NUM_SHOTS ${SHOTS}
    fi
done