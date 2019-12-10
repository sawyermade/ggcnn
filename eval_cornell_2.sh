#!/bin/bash
python3 eval_ggcnn.py \
	--network output/models/191206_1505_ggcnn2_50/epoch_44_iou_0.93 \
	--dataset cornell \
	--dataset-path ~/DATASETS/cornell_dataset/ \
	--use-depth 1 \
	--use-rgb 1 \
	--split 0.9 \
	--ds-rotate 0.0 \
	--num-workers 8 \
	--n-grasps 1 \
	--cuda $1 \
	--iou-eval #\
	# --vis #\
	# --augment #\
	# --jacquard-output