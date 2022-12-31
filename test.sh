# Copyright (c) 2021-2022, NVIDIA Corporation & Affiliates. All rights reserved.
#
# This work is made available under the Nvidia Source Code License-NC.
# To view a copy of this license, visit
# https://github.com/NVlabs/FreeSOLO/blob/main/LICENSE

#bash test.sh /home/hossein/github/FreeSOLO/training_dir/FreeSOLO_R101_30k_pl.pth
CUDA_VISIBLE_DEVICES=0 python train_net.py \
	--dist-url tcp://127.0.0.1:$(( RANDOM % 1000 + 50000 )) \
	--eval-only \
	--num-gpus 1 \
	--config configs/freesolo/freesolo_30k.yaml \
	OUTPUT_DIR /home/hossein/data/tmp2 \
	MODEL.WEIGHTS $1

# evaluate using official coco api
# python tools/eval_cocoapi.py
