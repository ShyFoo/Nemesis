#!/bin/bash

# custom configs
DATA=/path/to/data
TRAINER=CoOp
CRT_TRAINER=CoOpCRT

DATASET=(
    "caltech101"
    "eurosat"
    "food101"
    "oxford_flowers"
    "stanford_cars"
    "ucf101"
    "dtd"
    "fgvc_aircraft"
    "oxford_pets"
    "sun397"
    "imagenet"
)

CFG=$1  # # config file (rn50_ep50.yaml, rn50_ep100.yaml, rn50.yaml, etc.)
CTP=$2  # class token position (end or middle)
NCTX=$3  # number of context tokens (4, 16, etc.)
SHOTS=$4  # number of shots (1, 2, 4, 8, 16)
CSC=$5  # class-specific context (False or True)
EPOCH=$6  # number of epochs
CORRUPT=$7  # corrupt type (original, replace, rescale). p.s. when corrupt type equals original, both B and C still need to take arbitrary values like 666, to ensure running
POS=$8  # corrupt position, range: [0, NCTX-1]
CRT_FACTOR=$9  # the corruption weight in corruption experiments (std in replacing operations: 0, 0.0001, ... or scaling factor in rescaling operations: 0.5, 0.1, ...)

for dataset in "${DATASET[@]}"
do
    for SEED in 1 2 3
    do
        DIR=output/${DATASET}/${TRAINER}/${CFG}_${SHOTS}shots/nctx${NCTX}_csc${CSC}_ctp${CTP}/seed${SEED}
        if [ ${CORRUPT} = "original" ]; then
            OUTDIR=output_crt/${CRT_TRAINER}/${dataset}/${TRAINER}/${CFG}_${SHOTS}shots/${CORRUPT}/seed${SEED}
        elif [ ${CORRUPT} = "replace" ]; then
            OUTDIR=output_crt/${CRT_TRAINER}/${dataset}/${TRAINER}/${CFG}_${SHOTS}shots/${CORRUPT}/pos-${POS}_std-${CRT_FACTOR}/seed${SEED}
        elif [ ${CORRUPT} = "rescale" ]; then
            OUTDIR=output_crt/${CRT_TRAINER}/${dataset}/${TRAINER}/${CFG}_${SHOTS}shots/${CORRUPT}/pos-${POS}_scale-${CRT_FACTOR}/seed${SEED}
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
            --dataset-config-file configs/datasets/${dataset}.yaml \
            --config-file configs/trainers/${TRAINER}/${CFG}.yaml \
            --corrupt-type ${CORRUPT} \
            --corrupt-position ${POS} \
            --crt-factor ${CRT_FACTOR} \
            --output-dir ${OUTDIR} \
            --model-dir ${DIR} \
            --load-epoch ${EPOCH} \
            --eval-only \
            TRAINER.CoOpCRT.N_CTX ${NCTX} \
            TRAINER.CoOpCRT.CSC ${CSC} \
            TRAINER.CoOpCRT.CLASS_TOKEN_POSITION ${CTP}
        fi
    done
done