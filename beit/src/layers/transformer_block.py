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

from functools import partial

import mindspore as ms
import mindspore.nn as nn

from .attention import Attention
from .drop_path_timm import DropPath
from .mlp import Mlp
from .custom_identity import CustomIdentity


class Block(nn.Cell):
    def __init__(
            self,
            dim,
            num_heads,
            mlp_ratio=4.0,
            qkv_bias=False,
            qk_scale=None,
            drop=0.,
            attn_drop=0.,
            drop_path=0.,
            init_values=None,
            act_layer=nn.GELU(),
            norm_layer=partial(nn.LayerNorm, epsilon=1e-6),
            window_size=None,
            attn_head_dim=None
    ):
        super().__init__()
        self.norm1 = norm_layer((dim,))
        self.attn = Attention(
            dim,
            num_heads=num_heads,
            qkv_bias=qkv_bias,
            qk_scale=qk_scale,
            attn_drop=attn_drop,
            proj_drop=drop,
            window_size=window_size,
            attn_head_dim=attn_head_dim
        )
        self.drop_path = DropPath(keep_prob=1 - drop_path) \
            if drop_path > 0. \
            else CustomIdentity()
        self.norm2 = norm_layer((dim,))
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = Mlp(
            in_features=dim,
            hidden_features=mlp_hidden_dim,
            act_layer=act_layer,
            drop=drop
        )
        if init_values is not None and init_values > 0:
            self.gamma_1 = ms.Parameter(init_values * ms.numpy.ones((dim,)),
                                        requires_grad=True)
            self.gamma_2 = ms.Parameter(init_values * ms.numpy.ones((dim,)),
                                        requires_grad=True)
        else:
            self.gamma_1, self.gamma_2 = None, None

    def construct(self, x, rel_pos_bias=None):
        if self.gamma_1 is None:
            x = x + self.drop_path(
                self.attn(self.norm1(x), rel_pos_bias=rel_pos_bias)
            )
            x = x + self.drop_path(self.mlp(self.norm2(x)))
        else:
            x = x + self.drop_path(
                self.gamma_1
                * self.attn(self.norm1(x), rel_pos_bias=rel_pos_bias)
            )
            x = x + self.drop_path(self.gamma_2 * self.mlp(self.norm2(x)))
        return x
