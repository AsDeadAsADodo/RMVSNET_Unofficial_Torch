#!/usr/bin/env bash
MVS_TRAINING="/home/kong/Workspace/dtu_training"
python train.py --dataset=dtu_yao --batch_size=4 --trainpath=$MVS_TRAINING --trainlist lists/dtu/train.txt --testlist lists/dtu/test.txt --numdepth=192 --logdir ./checkpoints/d192 $@

