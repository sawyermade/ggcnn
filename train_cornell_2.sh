#!/bin/bash
python3 train_ggcnn.py \
	--description ggcnn2_50 \
	--network ggcnn2 \
	--dataset cornell \
	--dataset-path ~/DATASETS/cornell_dataset/ \
	--epochs 50 \
	--use-depth 1 \
	--use-rgb 1 \
	--cuda $1
#	--vis
