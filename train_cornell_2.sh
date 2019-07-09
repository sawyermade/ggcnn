#!/bin/bash
python3 train_ggcnn.py \
	--description ggcnn2_50 \
	--network ggcnn2 \
	--dataset cornell \
	--dataset-path cornell_dataset/ \
	--epochs 50 \
	--cuda $1