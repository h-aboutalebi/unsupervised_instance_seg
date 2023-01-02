# Copyright (c) 2021-2022, NVIDIA Corporation & Affiliates. All rights reserved.
#
# This work is made available under the Nvidia Source Code License-NC.
# To view a copy of this license, visit
# https://github.com/NVlabs/FreeSOLO/blob/main/LICENSE

import os
import argparse
import json
import glob
from tqdm import tqdm
parser = argparse.ArgumentParser(description='generate pseudo labels')
parser.add_argument('-o', '--output_path', default=os.path.expanduser('~') + '/data/tmp3',
                    help='output path for files produced by the agent')
parser.add_argument('--input_path', default=os.path.expanduser('~') + '/data/tmp3/inference/coco_instances_results.json',
                    help='output path for files produced by the agent')
args = parser.parse_args()
save_path = os.path.join(args.output_path, 'ann.json')
ann_dict = json.load(open("datasets/truck/val/ann.json"))
anns = json.load(open(args.input_path))
print('original {} images, {} objects.'.format(len(ann_dict['images']), len(ann_dict['annotations'])))

start_id = len(ann_dict['annotations'])
new_anns = []
for id, ann in enumerate(anns):
    # filter
    box = ann['bbox']
    h, w = ann['segmentation']['size']
    if (box[2] - box[0]) >= 0.95 * w:
        continue
    ann['id'] = id + start_id
    new_anns.append(ann)

ann_dict['annotations'] = new_anns
#ann_dict['annotations'].extend(new_anns)
json.dump(ann_dict, open(save_path, 'w'))
print('{} images, {} objects.'.format(len(ann_dict['images']), len(ann_dict['annotations'])))




