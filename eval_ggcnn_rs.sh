#!/bin/bash
python3 eval_ggcnn_rs.py \
	--network output/models/ggcnn_50_dc_default/epoch_44_iou_0.93 \
	--dataset rs \
	--dataset-path ./temp_img_dir \
	--use-depth 1 \
	--use-rgb 1 \
	--split 0.0 \
	--ds-rotate 0.0 \
	--num-workers 8 \
	--n-grasps 1 \
	--cuda $1 \
	--vis
	# --iou-eval #\
	# --vis #\
	# --augment #\
	# --jacquard-output