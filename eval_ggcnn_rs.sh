#!/bin/bash
python3 eval_ggcnn_rs.py \
	--network output/models/ggcnn2_50_dc_default/epoch_37_iou_0.95 \
	--dataset rs \
	--dataset-path ./temp_img_dir \
	--use-depth 1 \
	--use-rgb 1 \
	--split 0.0 \
	--ds-rotate 0.0 \
	--num-workers 8 \
	--n-grasps 3 \
	--cuda $1 \
	--vis
	# --iou-eval #\
	# --vis #\
	# --augment #\
	# --jacquard-output