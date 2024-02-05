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
"""Functions of cells"""
import logging

import mindspore.nn as nn
from mindspore import dtype as mstype
from mindspore.ops import functional as F


class OutputTo16(nn.Cell):
    """Wrap cell for amp. Cast network output back to float16."""

    def __init__(self, op):
        super(OutputTo16, self).__init__(auto_prefix=False)
        self._op = op

    def construct(self, *inputs, **kwargs):
        x = inputs[0]
        return F.cast(self._op(x), mstype.float16)


def do_keep_fp16(network, cell_types):
    """Cast cell to fp32 if cell in cell_types."""
    for _, cell in network.cells_and_names():
        if isinstance(cell, cell_types):
            cell.to_float(mstype.float16)


def do_keep_fp32(network, cell_types):
    """Cast cell to fp32 if cell in cell_types."""
    for _, cell in network.cells_and_names():
        if isinstance(cell, cell_types):
            cell.to_float(mstype.float32)


def cast_amp(net, args):
    """Cast network amp_level."""
    logging.info('=> using amp_level %s', args.amp_level)
    if args.amp_level == 'O2':
        cell_types = (nn.LayerNorm, nn.BatchNorm2d)
        net.to_float(mstype.float16)
        do_keep_fp32(net, cell_types)
    elif args.amp_level == 'O1':
        cell_types = (nn.LayerNorm, nn.Softmax, nn.BatchNorm2d)
        net.to_float(mstype.float16)
        do_keep_fp32(net, cell_types)
    elif args.amp_level == 'O3':
        net.to_float(mstype.float16)
    else:
        args.loss_scale = 1.
        args.is_dynamic_loss_scale = 0
        logging.info('=> When amp_level is O0, using fixed loss_scale with %s',
                     args.loss_scale)
