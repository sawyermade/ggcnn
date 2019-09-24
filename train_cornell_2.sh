#!/bin/bash
python3 train_ggcnn.py \
	--description ggcnn2_500 \
	--network ggcnn2 \
	--dataset cornell \
	--dataset-path ~/DATASETS/cornell_dataset/ \
	--epochs 500 \
	--cuda $1