#!/usr/bin/env bash
DTU_TESTING="/home/kong/Workspace/dtu_test/"
CKPT_FILE="./checkpoints/d192/model_000015.ckpt"
python eval.py --dataset=dtu_yao_eval --batch_size=1 --testpath=$DTU_TESTING --testlist lists/dtu/test.txt --loadckpt $CKPT_FILE $@
