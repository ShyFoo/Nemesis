#!/bin/bash

# custom configs
DATA=/path/to/dataset
TRAINER=PLOT
N=4  # number of proxy

DATASET=$1  # imagenet, sun397, caltech101, eurosat, food101, oxford_flowers, stanford_cars, ucf101, dtd, fgvc_aircraft, oxford_pets
CFG=$2  # config file (rn50_ep50, rn50_ep100, rn50, etc.)
CTP=$3  # class token position (end or middle)
NCTX=$4  # number of context tokens (4, 16, etc.)
SHOTS=$5  # number of shots (1, 2, 4, 8, 16)
CSC=$6  # class-specific context (False or True)


for SEED in 1 2 3
do
    DIR=output/${DATASET}/${TRAINER}/${CFG}_${SHOTS}shots/N${N}_nctx${NCTX}_csc${CSC}_ctp${CTP}/seed${SEED}
    LOG_FILE="${DIR}/log.txt"

    if [ -d "$DIR" ] && [ -f "${LOG_FILE}" ] && tail -n 2 "${LOG_FILE}" | grep -q "Elapsed"; then
        echo "Oops! The results exist at ${DIR} (so skip this job)"
    else
        # if the directory exists but the task was interrupted, remove all files and start over
        if [ -d "$DIR" ]; then
            rm -r ${DIR}/*
        fi

        python train.py \
        --root ${DATA} \
        --seed ${SEED} \
        --trainer ${TRAINER} \
        --dataset-config-file configs/datasets/${DATASET}.yaml \
        --config-file configs/trainers/${TRAINER}/${CFG}.yaml \
        --output-dir ${DIR} \
        TRAINER.PLOT.N_CTX ${NCTX} \
        TRAINER.PLOT.CSC ${CSC} \
        TRAINER.PLOT.CLASS_TOKEN_POSITION ${CTP} \
        DATASET.NUM_SHOTS ${SHOTS} \
        TRAINER.PLOT.N ${N}
    fi
done