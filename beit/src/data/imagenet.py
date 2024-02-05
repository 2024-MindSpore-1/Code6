# Copyright 2023 Huawei Technologies Co., Ltd
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ============================================================================
"""
Data operations, will be used in train.py and eval.py
"""
import math
import os
from typing import Optional, Tuple

import mindspore.common.dtype as mstype
import mindspore.dataset as ds
import mindspore.dataset.transforms as transforms
import mindspore.dataset.vision as vision
from mindspore.dataset.vision.utils import Inter

from src.data.augment.auto_augment import pil_interp, rand_augment_transform
from src.data.augment.mixup import Mixup
from src.data.augment.random_erasing import RandomErasing

IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)
IMAGENET_INCEPTION_MEAN = (0.5, 0.5, 0.5)
IMAGENET_INCEPTION_STD = (0.5, 0.5, 0.5)
DEFAULT_CROP_PCT = 0.875


class ImageNet:
    """ImageNet Define"""

    def __init__(self, args, training=True):
        train_dir = os.path.join(args.data_url, args.train_dir)
        val_dir = os.path.join(args.data_url, args.val_dir)
        if training:
            self.train_dataset = create_dataset_imagenet(
                train_dir, training=True, args=args
            )
        self.val_dataset = create_dataset_imagenet(
            val_dir, training=False, args=args
        )


def get_validation_transforms(
        image_size: int,
        mean: Tuple[float, float, float],
        std: Tuple[float, float, float],
        crop_pct: Optional[float] = None,
):
    """Get only validation transforms."""
    crop_pct = crop_pct or DEFAULT_CROP_PCT
    scale_size = int(math.floor(image_size / crop_pct))

    transform_img = [
        vision.Decode(),
        vision.Resize(scale_size, interpolation=Inter.PILCUBIC),
        vision.CenterCrop(image_size),
        vision.ToTensor(),
        vision.Normalize(mean=mean, std=std, is_hwc=False),
    ]
    return transform_img


def get_main_transforms(
        image_size,
        interpolation,
        auto_augment,
        min_crop,
        re_prob,
        re_mode,
        re_count,
        crop_pct: Optional[float] = None,
        training: bool = False,
        imagenet_default_mean_and_std: bool = True,
):
    """Get main transforms."""
    # define map operations
    # BICUBIC: 3

    mean = (IMAGENET_MEAN if imagenet_default_mean_and_std
            else IMAGENET_INCEPTION_MEAN)
    std = (IMAGENET_STD if imagenet_default_mean_and_std
           else IMAGENET_INCEPTION_STD)
    if training:
        aa_params = dict(
            translate_const=int(image_size * 0.45),
            img_mean=tuple([min(255, round(255 * x)) for x in mean]),
        )
        interpolation = interpolation
        auto_augment = auto_augment
        aa_params['interpolation'] = pil_interp(interpolation)

        assert 0 <= min_crop < 1
        transform_img = [
            vision.RandomCropDecodeResize(
                image_size,
                scale=(min_crop, 1.0),
                ratio=(3 / 4, 4 / 3),
                interpolation=Inter.PILCUBIC
            ),
            vision.RandomHorizontalFlip(prob=0.5),
            vision.ToPIL()
        ]
        if auto_augment != "None":
            transform_img += [rand_augment_transform(auto_augment, aa_params)]
        transform_img += [
            vision.ToTensor(),
            vision.Normalize(mean=mean, std=std, is_hwc=False),
            RandomErasing(re_prob, mode=re_mode, max_count=re_count)
        ]
    else:
        transform_img = get_validation_transforms(
            image_size, mean, std, crop_pct
        )
    return transform_img


def create_dataset_imagenet(
        dataset_dir,
        args,
        repeat_num=1,
        training=True,
):
    """Create a train or eval ImageNet2012 dataset for ConvMixer.

    Parameters
    ----------
    dataset_dir: str
        The path of dataset.
    args: argparse.Namespace
        Parsed configuration and command line arguments.
    repeat_num: int
        The repeat times of dataset. Default: 1
    training: bool
        Whether dataset is used for train or eval.

    Returns
    -------
        Dataset
    """

    device_num, rank_id = _get_rank_info()
    shuffle = training
    drop_remainder = training

    if device_num == 1 or not training:
        val_num_samples = None if args.val_num_samples == -1 \
            else args.val_num_samples
        data_set = ds.ImageFolderDataset(
            dataset_dir,
            num_parallel_workers=args.num_parallel_workers,
            shuffle=shuffle,
            num_samples=val_num_samples,
        )
    else:
        train_num_samples = None if args.train_num_samples == -1 \
            else args.train_num_samples
        data_set = ds.ImageFolderDataset(
            dataset_dir,
            num_parallel_workers=args.num_parallel_workers,
            shuffle=shuffle,
            num_shards=device_num,
            shard_id=rank_id,
            num_samples=train_num_samples)

    image_size = args.image_size

    transform_img = get_main_transforms(
        image_size, args.aa_interpolation, args.auto_augment, args.min_crop,
        args.re_prob, args.re_mode, args.re_count, args.crop_pct,
        training=training,
        imagenet_default_mean_and_std=args.imagenet_default_mean_and_std,
    )

    transform_label = transforms.TypeCast(mstype.int32)

    data_set = data_set.map(input_columns="image",
                            num_parallel_workers=args.num_parallel_workers,
                            operations=transform_img)
    data_set = data_set.map(input_columns="label",
                            num_parallel_workers=args.num_parallel_workers,
                            operations=transform_label)
    if (args.mixup > 0.0 or args.cutmix > 0.0) and not training:
        # if use mixup and not training(False), one hot val data label
        one_hot = transforms.OneHot(num_classes=args.num_classes)
        cast_type = transforms.TypeCast(mstype.float32)
        data_set = data_set.map(input_columns=["label"],
                                num_parallel_workers=args.num_parallel_workers,
                                operations=[one_hot, cast_type])
    # apply batch operations
    data_set = data_set.batch(args.batch_size, drop_remainder=drop_remainder,
                              num_parallel_workers=args.num_parallel_workers)

    if (args.mixup > 0.0 or args.cutmix > 0.0) and training:
        mixup_fn = Mixup(mixup_alpha=args.mixup, cutmix_alpha=args.cutmix,
                         cutmix_minmax=None, prob=args.mixup_prob,
                         switch_prob=args.switch_prob, mode=args.mixup_mode,
                         label_smoothing=args.label_smoothing,
                         num_classes=args.num_classes)
        data_set = data_set.map(operations=mixup_fn,
                                input_columns=["image", "label"],
                                num_parallel_workers=args.num_parallel_workers)

    # apply dataset repeat operation
    data_set = data_set.repeat(repeat_num)

    return data_set


def _get_rank_info():
    """
    get rank size and rank id
    """
    rank_size = int(os.environ.get("RANK_SIZE", 1))

    if rank_size > 1:
        from mindspore.communication.management import get_rank, get_group_size
        rank_size = get_group_size()
        rank_id = get_rank()
    else:
        rank_size = rank_id = None

    return rank_size, rank_id
