#!/bin/bash

# custom config
DATA=/path/to/dataset
TRAINER=CoOpNemesis

DATASET=$1
CFG=$2  # config file
CTP=$3  # class token position (end or middle)
NCTX=$4  # number of context tokens
SHOTS=$5  # number of shots (1, 2, 4, 8, 16)
CSC=$6  # class-specific context (False or True)
ALPHA=$7  # the scaling weight of the normalization loss
NT=$8  # normalization types ("one", "two", "inf")
NP=$9  # the number of positions for currupting during training: [1, N_CTX], only works when alpha = adaptive
CW=${10}  # the threshold of corruption weight, only used when beta != 1.0
BETA=${11}  # the beta coefficient for balancing the pan loss and the pun loss (decimals like 0.1, 0.2, ..., 0: only pan loss, 1: only the pun loss)

for SEED in 1 2 3
do
    if [ ${BETA} != 1.0 ]; then
        DIR=output/${DATASET}/${TRAINER}/Beta-${BETA}/${CFG}_${SHOTS}shots/alpha${ALPHA}_cw${CW}_pn${NP}_nctx${NCTX}_csc${CSC}_ctp${CTP}_nt${NT}/seed${SEED}
    else
        DIR=output/${DATASET}/${TRAINER}/Beta-${BETA}/${CFG}_${SHOTS}shots/alpha${ALPHA}_nctx${NCTX}_csc${CSC}_ctp${CTP}_nt${NT}/seed${SEED}
    fi

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
        TRAINER.CoOpNemesis.N_CTX ${NCTX} \
        TRAINER.CoOpNemesis.CSC ${CSC} \
        TRAINER.CoOpNemesis.CLASS_TOKEN_POSITION ${CTP} \
        DATASET.NUM_SHOTS ${SHOTS} \
        TRAINER.CoOpNemesis.A ${ALPHA} \
        TRAINER.CoOpNemesis.NT ${NT} \
        TRAINER.CoOpNemesis.NP ${NP} \
        TRAINER.CoOpNemesis.CW ${CW} \
        TRAINER.CoOpNemesis.BETA ${BETA}
    fi
done