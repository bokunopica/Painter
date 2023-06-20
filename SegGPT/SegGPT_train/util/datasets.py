# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
# --------------------------------------------------------
# References:
# DeiT: https://github.com/facebookresearch/deit
# --------------------------------------------------------

import os
import PIL
import json
import random

from torchvision import datasets, transforms

from timm.data import create_transform
from timm.data.constants import IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD


def build_dataset(is_train, args):
    transform = build_transform(is_train, args)

    root = os.path.join(args.data_path, 'train' if is_train else 'val')
    dataset = datasets.ImageFolder(root, transform=transform)

    print(dataset)

    return dataset


def build_transform(is_train, args):
    mean = IMAGENET_DEFAULT_MEAN
    std = IMAGENET_DEFAULT_STD
    # train transform
    if is_train:
        # this should always dispatch to transforms_imagenet_train
        transform = create_transform(
            input_size=args.input_size,
            is_training=True,
            color_jitter=args.color_jitter,
            auto_augment=args.aa,
            interpolation='bicubic',
            re_prob=args.reprob,
            re_mode=args.remode,
            re_count=args.recount,
            mean=mean,
            std=std,
        )
        return transform

    # eval transform
    t = []
    if args.input_size <= 224:
        crop_pct = 224 / 256
    else:
        crop_pct = 1.0
    size = int(args.input_size / crop_pct)
    t.append(
        transforms.Resize(size, interpolation=PIL.Image.BICUBIC),  # to maintain same ratio w.r.t. 224 images
    )
    t.append(transforms.CenterCrop(args.input_size))

    t.append(transforms.ToTensor())
    t.append(transforms.Normalize(mean, std))
    return transforms.Compose(t)


def build_breast_cancer_data_json():
    source_base_dir = '/mnt/c/data/breastCancer/processed/source'
    target_base_dir = '/mnt/c/data/breastCancer/processed/target'
    json_path = '/mnt/c/data/breastCancer/processed/meta.json'
    # type_list = ['benign', 'normal', 'malignant']
    source_filename_list = os.listdir(source_base_dir)
    with open(json_path, 'w') as f:
        meta = []
        for source in source_filename_list:
            target = source.replace('.png', '_mask.png')
            single_meta = {
                "image_path": f"{source_base_dir}/{source}",
                "target_path": f"{target_base_dir}/{target}",
                "type": "instance"
            }
            meta.append(single_meta)
        f.write(json.dumps(meta))





def split_breast_cancer_data_json(percentage=0.8):
    json_path = '/run/media/breastCancer/processed/meta.json'
    train_json_path = '/run/media/breastCancer/processed/meta_train.json'
    test_json_path = '/run/media/breastCancer/processed/meta_test.json'
    with open(json_path, 'r') as f:
        data_list = json.loads(f.read())

    
    def get_class(image_path):
        _class_list = ['benign', 'malignant', 'normal']
        for _class in _class_list:
            if _class in image_path:
                return _class
        return None
    
    def split_single_train_test(single_data_source_list, percentage):
        total_len = len(single_data_source_list)
        cut_len = int(percentage*total_len)
        return single_data_source_list[:cut_len], single_data_source_list[cut_len:]


    def split_train_test(source_list, percentage):
        train_list = []
        test_list = []
        for single_data_source_list in source_list:
            _train_list, _test_list = split_single_train_test(single_data_source_list, percentage)
            train_list += _train_list
            test_list += _test_list
        return train_list, test_list


    benign_list = []
    malignant_list = []
    normal_list = []

    
    for i in range(len(data_list)):
        data = data_list[i]
        _class = get_class(data['image_path'])
        data['_class'] = _class
        if 'benign' == _class:
            benign_list.append(data)
        elif 'malignant' == _class:
            malignant_list.append(data)
        else:
            normal_list.append(data)

    random.seed(111)
    random.shuffle(benign_list)
    random.shuffle(malignant_list)
    random.shuffle(normal_list)

    train_list, test_list = split_train_test([benign_list, malignant_list, normal_list], percentage)
    with open(train_json_path, 'w') as f:
        f.write(json.dumps([data for data in train_list]))
    with open(test_json_path, 'w') as f:
        f.write(json.dumps([data for data in test_list]))


if __name__ == "__main__":
    split_breast_cancer_data_json()