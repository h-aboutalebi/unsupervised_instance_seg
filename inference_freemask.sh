# Copyright (c) 2021-2022, NVIDIA Corporation & Affiliates. All rights reserved.
#
# This work is made available under the Nvidia Source Code License-NC.
# To view a copy of this license, visit
# https://github.com/NVlabs/FreeSOLO/blob/main/LICENSE

# generate free masks for train2017 images
CUDA_VISIBLE_DEVICES=0 python demo/inference_freemask.py --config-file configs/freesolo/freemask.yaml \
	--input datasets/truck/val \
	--output /home/hossein/github/FreeSOLO/datasets/truck/val/ann.json \
	--split -1 \
	--opts MODEL.WEIGHTS training_dir/pre-trained/DenseCL/densecl_r101_imagenet_200ep.pkl \

