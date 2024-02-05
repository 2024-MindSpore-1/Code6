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
"""Attention layer of TransformerBlock."""

import mindspore as ms
import mindspore.nn as nn
import mindspore.common.initializer as init
from mindspore import ops


class Attention(nn.Cell):
    def __init__(
            self,
            dim,
            num_heads=8,
            qkv_bias=False,
            qk_scale=None,
            attn_drop=0.0,
            proj_drop=0.0,
            window_size=None,
            attn_head_dim=None,
    ):
        super().__init__()
        self.num_heads = num_heads
        head_dim = dim // num_heads
        if attn_head_dim is not None:
            head_dim = attn_head_dim
        all_head_dim = head_dim * self.num_heads
        self.scale = qk_scale or head_dim ** -0.5

        # out_dim = in_dim or dim
        self.qkv = nn.Dense(dim, all_head_dim * 3, has_bias=False)
        if qkv_bias:
            self.q_bias = ms.Parameter(init.initializer(init.Constant(0),
                                                        (all_head_dim,)))
            self.v_bias = ms.Parameter(init.initializer(init.Constant(0),
                                                        (all_head_dim,)))
        else:
            self.q_bias = None
            self.v_bias = None
        if window_size:
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
            c = ops.meshgrid((coords_h, coords_w))
            # 2, Wh, Ww
            coords = ops.stack(c)
            coords = ops.Cast()(coords, ms.int64)
            # 2, Wh*Ww
            coords_flatten = ops.flatten(coords)
            # 2, Wh * Ww, Wh * Ww
            coords_0 = coords_flatten[:, :, None]
            coords_1 = coords_flatten[:, None, :]
            relative_coords = coords_0 - coords_1
            # Wh*Ww, Wh*Ww, 2
            relative_coords = ops.transpose(relative_coords, (1, 2, 0))
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
        else:
            self.window_size = None
            self.relative_position_bias_table = None
            self.relative_position_index = None

        self.attn_drop = nn.Dropout(keep_prob=1 - attn_drop)
        self.proj = nn.Dense(all_head_dim, dim)
        self.proj_drop = nn.Dropout(keep_prob=1 - proj_drop)

        self.use_qkv_bias = False
        if self.q_bias is not None:
            self.k_bias = ms.numpy.zeros_like(self.v_bias)
            self.use_qkv_bias = True

    def construct(self, x, rel_pos_bias=None):
        B, N, C = x.shape
        qkv = self.qkv(x)
        if self.use_qkv_bias:
            qkv_bias = ops.concat((self.q_bias, self.k_bias, self.v_bias))
            qkv = qkv + ops.expand_dims(ops.expand_dims(qkv_bias, 0), 0)
        qkv = qkv.reshape(B, N, 3, self.num_heads, C // self.num_heads)
        qkv = ops.transpose(qkv, (2, 0, 3, 1, 4))
        q, k, v = qkv[0], qkv[1], qkv[2]

        q = q * self.scale
        attn = ops.matmul(q, k.transpose(0, 1, 3, 2))

        if self.relative_position_bias_table is not None:
            relative_position_bias = self.relative_position_bias_table[
                self.relative_position_index.view(-1)
            ].view(
                self.window_size[0] * self.window_size[1] + 1,
                self.window_size[0] * self.window_size[1] + 1,
                -1
            )  # Wh*Ww,Wh*Ww,nH
            relative_position_bias = ops.transpose(relative_position_bias,
                                                   (2, 0, 1))
            attn = attn + ops.expand_dims(relative_position_bias, 0)

        if rel_pos_bias is not None:
            attn = attn + rel_pos_bias

        attn = ops.softmax(attn, axis=-1)
        attn = self.attn_drop(attn)

        x = ops.matmul(attn, v).transpose(0, 2, 1, 3).reshape(B, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)

        return x
