# Copyright (c) 2021-2022, NVIDIA Corporation & Affiliates. All rights reserved.
#
# This work is made available under the Nvidia Source Code License-NC.
# To view a copy of this license, visit
# https://github.com/NVlabs/FreeSOLO/blob/main/LICENSE

#export GLOO_SOCKET_IFNAME=eth0
CUDA_VISIBLE_DEVICES=0 python train_net.py \
	--dist-url tcp://127.0.0.1:$(( RANDOM % 1000 + 50000 )) \
	--eval-only \
	--num-gpus 1 \
	--config configs/freesolo/freesolo_30k.yaml \
	OUTPUT_DIR /home/hossein/data/tmp3 \
	DATASETS.TEST  '("truck_val",)' \
	MODEL.SOLOV2.UPDATE_THR 0.3 \
	MODEL.SOLOV2.MAX_PER_IMG 20 \
	MODEL.WEIGHTS /home/hossein/data/results_free_solo/model_0004999.pth

# convert to annotation format
# CUDA_VISIBLE_DEVICES=0 python tools/gen_pseudo_labels.py
