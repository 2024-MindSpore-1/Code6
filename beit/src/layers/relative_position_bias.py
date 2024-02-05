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
#
# This file has been derived from the
# https://github.com/microsoft/unilm/tree/master/beit
# repository and modified.
# ============================================================================

import mindspore as ms
import mindspore.nn as nn
from mindspore.common import initializer as init


class RelativePositionBias(nn.Cell):
    def __init__(self, window_size, num_heads):
        super().__init__()
        self.window_size = window_size
        self.num_relative_distance = ((2 * window_size[0] - 1)
                                      * (2 * window_size[1] - 1)
                                      + 3)
        # 2*Wh-1 * 2*Ww-1, nH
        self.relative_position_bias_table = ms.Parameter(
            init.initializer(init.Constant(0),
                             (self.num_relative_distance, num_heads))
        )
        # cls to token & token 2 cls & cls to cls

        # get pair-wise relative position index for each token inside
        # the window
        coords_h = ms.numpy.arange(window_size[0])
        coords_w = ms.numpy.arange(window_size[1])
        c = ms.ops.meshgrid((coords_h, coords_w))
        # 2, Wh, Ww
        coords = ms.ops.stack(c)
        coords = ms.ops.Cast()(coords, ms.int64)
        # 2, Wh*Ww
        coords_flatten = ms.ops.flatten(coords)
        # 2, Wh * Ww, Wh * Ww
        coords_0 = coords_flatten[:, :, None]
        coords_1 = coords_flatten[:, None, :]
        relative_coords = coords_0 - coords_1
        # Wh*Ww, Wh*Ww, 2
        relative_coords = ms.ops.transpose(relative_coords, (1, 2, 0))
        # shift to start from 0
        relative_coords[:, :, 0] += window_size[0] - 1
        relative_coords[:, :, 1] += window_size[1] - 1
        relative_coords[:, :, 0] *= 2 * window_size[1] - 1
        relative_position_index = ms.numpy.zeros(
            (window_size[0] * window_size[1] + 1,) * 2,
            dtype=relative_coords.dtype
        )
        # Wh*Ww, Wh*Ww
        relative_position_index[1:, 1:] = relative_coords.sum(axis=-1)
        relative_position_index[0, 0:] = self.num_relative_distance - 3
        relative_position_index[0:, 0] = self.num_relative_distance - 2
        relative_position_index[0, 0] = self.num_relative_distance - 1
        self.relative_position_index = ms.Parameter(
            relative_position_index, requires_grad=False)

    def construct(self):
        relative_position_bias = self.relative_position_bias_table[
            self.relative_position_index.view(-1)
        ].view(
            self.window_size[0] * self.window_size[1] + 1,
            self.window_size[0] * self.window_size[1] + 1,
            -1
        )  # Wh*Ww,Wh*Ww,nH
        relative_position_bias = ms.ops.transpose(relative_position_bias,
                                                  (2, 0, 1))
        return relative_position_bias
