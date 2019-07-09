#!/bin/bash
python3 train_ggcnn.py \
	--description ggcnn1_50 \
	--network ggcnn \
	--dataset cornell \
	--dataset-path cornell_dataset/ \
	--epochs 50 \
	--cuda $1