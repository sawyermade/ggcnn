#!/bin/bash
python3 eval_ggcnn_rs_dir.py \
	--network output/models/191206_1505_ggcnn2_50/epoch_44_iou_0.93 \
	--dataset rs_dir \
	--dataset-path ./realsense/remote_frames_png/ \
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