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
Run prediction on folder or single image, output results and save them to
JSON file.
"""
import argparse
import json
import os
import logging
from functools import reduce
from pathlib import Path

import mindspore as ms
import mindspore.nn as nn
from mindspore import context
from src.data.imagenet import (
    get_validation_transforms, IMAGENET_INCEPTION_MEAN, IMAGENET_INCEPTION_STD
)
from src.tools.cell import cast_amp
from src.tools.get_misc import get_model, config_logging


class NetWithSoftmax(nn.Cell):
    """
    Network with softmax at the end.
    """

    def __init__(self, net):
        super().__init__()
        self.net = net
        self.softmax = nn.Softmax()

    def construct(self, *inputs, **kwargs):
        x = inputs[0]
        results = self.net(x)
        res = self.softmax(results)
        return res


def parse_args():
    """
    Create and parse command-line arguments.

    Returns
    -------
    argparse.Namespace
        Parsed command-line arguments.

    """
    parser = argparse.ArgumentParser(
        description=__doc__, add_help=False,
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )

    parser.add_argument(
        '-h', '--help',
        action='help',
        default=argparse.SUPPRESS,
        help='Show this help message and exit.'
    )
    parser.add_argument(
        'data', type=Path,
        help='Path to dataset for prediction.'
    )
    parser.add_argument(
        '-c', '--checkpoint', type=Path, required=True,
        help='Path to checkpoint to load.'
    )
    parser.add_argument(
        '-a', '--arch',
        help='Model architecture name.'
    )
    parser.add_argument(
        '-o', '--output', type=Path, default=Path('predictions.json'),
        help='Path to output JSON file.'
    )
    parser.add_argument(
        '--image_size', type=int, default=224, help='Image size.'
    )
    parser.add_argument(
        '--amp_level', default='O0', choices=['O0', 'O1', 'O2', 'O3'],
        help='AMP optimization level.'
    )
    parser.add_argument(
        '--device_target', default='GPU', choices=['GPU'],
        help='Target computation platform.'
    )
    parser.add_argument(
        '--num_classes', type=int, default=1000,
        help='Number of dataset classes the model was trained on.'
    )

    return parser.parse_args()


def data_loader(path: Path, image_size: int):
    """Load image or images from folder in generator."""
    preprocess = get_validation_transforms(
        image_size=image_size, crop_pct=0.9,
        mean=IMAGENET_INCEPTION_MEAN,
        std=IMAGENET_INCEPTION_STD
    )

    def apply(img):
        for p in preprocess:
            img = p(img)
        return img

    extensions = ('.png', '.jpg', '.jpeg')
    if path.is_dir():
        logging.info('Load directory.')
        for item in path.iterdir():
            if item.is_dir():
                continue
            if item.suffix.lower() not in extensions:
                continue
            with open(item, 'rb') as f:
                image_data = f.read()
            image = apply(image_data)
            yield str(item), ms.Tensor(image[None])
    else:
        logging.info('Load single image.')
        assert path.suffix.lower() in extensions
        with open(path, 'rb') as f:
            image_data = f.read()
        image = apply(image_data)
        yield str(path), ms.Tensor(image[None])


def main():
    """Entry point."""
    args = parse_args()
    os.environ['DEVICE_TARGET'] = args.device_target
    loader = data_loader(args.data, args.image_size)
    d = {}

    if args.checkpoint.suffix == '.ckpt':
        logging.info('Use pretrained checkpoint.')
        net = get_model(
            args.image_size,
            args.arch,
            args.num_classes,
            False,
        )
        cast_amp(net, args)
        if args.checkpoint:
            ms.load_checkpoint(str(args.checkpoint.absolute()), net=net)
        logging.info(
            'Number of parameters: %d',
            sum(
                reduce(lambda x, y: x * y, params.shape)
                for params in net.trainable_params()
            )
        )
        cast_amp(net, args)
        net = NetWithSoftmax(net)
    elif args.checkpoint.suffix == '.mindir':
        logging.info('Use MINDIR model.')
        graph = ms.load(str(args.checkpoint))
        net = nn.GraphCell(graph)
    else:
        raise ValueError(
            f'Unsupported checkpoint file format for "{args.checkpoint}".'
        )

    context.set_context(mode=context.GRAPH_MODE, device_target='GPU')

    argmax = ms.ops.Argmax(output_type=ms.int32)
    for (name, img) in loader:
        res = argmax(net(img))
        logging.info('%s (class: %d)', name, res)
        d[name] = int(res)

    with args.output.open(mode='w') as f:
        json.dump(d, f, indent=1)


if __name__ == '__main__':
    config_logging()
    main()
