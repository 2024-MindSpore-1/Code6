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
import math

import mindspore as ms
import mindspore.nn as nn
from mindspore.common import initializer as init
from mindspore import ops

from .layers import Block, PatchEmbed, RelativePositionBias, CustomIdentity


class VisionTransformer(nn.Cell):
    """
    Vision Transformer with support for patch or hybrid CNN input stage.
    """
    def __init__(self, img_size=224, patch_size=16, in_chans=3,
                 num_classes=1000, embed_dim=768, depth=12, num_heads=12,
                 mlp_ratio=4., qkv_bias=False, qk_scale=None, drop_rate=0.,
                 attn_drop_rate=0., drop_path_rate=0., norm_layer=nn.LayerNorm,
                 init_values=None, use_abs_pos_emb=True,
                 use_rel_pos_bias=False, use_shared_rel_pos_bias=False,
                 use_mean_pooling=True, init_scale=0.001,
                 approximate_gelu: bool = True):
        super().__init__()
        self.num_classes = num_classes
        # num_features for consistency with other models
        self.num_features = self.embed_dim = embed_dim

        self.patch_embed = PatchEmbed(
            img_size=img_size, patch_size=patch_size, in_chans=in_chans,
            embed_dim=embed_dim)
        num_patches = self.patch_embed.num_patches

        self.cls_token = ms.Parameter(init.initializer(
            init.TruncatedNormal(0.02), (1, 1, embed_dim), ms.float32
        ))
        if use_abs_pos_emb:
            self.pos_embed = ms.Parameter(init.initializer(
                init.TruncatedNormal(0.02),
                (1, num_patches + 1, embed_dim),
                ms.float32
            ))
        else:
            self.pos_embed = None
        self.pos_drop = nn.Dropout(keep_prob=1 - drop_rate)

        if use_shared_rel_pos_bias:
            self.rel_pos_bias = RelativePositionBias(
                window_size=self.patch_embed.patch_shape, num_heads=num_heads)
        else:
            self.rel_pos_bias = None

        # stochastic depth decay rule
        dpr = list(ms.numpy.linspace(
            ms.Tensor(0, ms.float32),
            ms.Tensor(drop_path_rate, ms.float32),
            depth
        ))
        self.use_rel_pos_bias = use_rel_pos_bias
        window_size = (self.patch_embed.patch_shape
                       if use_rel_pos_bias else None)
        self.blocks = nn.CellList([
            Block(dim=embed_dim, num_heads=num_heads, mlp_ratio=mlp_ratio,
                  qkv_bias=qkv_bias, qk_scale=qk_scale, drop=drop_rate,
                  attn_drop=attn_drop_rate, drop_path=dpr[i],
                  norm_layer=norm_layer, init_values=init_values,
                  window_size=window_size,
                  act_layer=nn.GELU(approximate=approximate_gelu))
            for i in range(depth)
        ])
        self.norm = (CustomIdentity()
                     if use_mean_pooling else norm_layer((embed_dim,)))
        self.fc_norm = norm_layer((embed_dim,)) if use_mean_pooling else None
        self.head = (nn.Dense(embed_dim, num_classes)
                     if num_classes > 0 else CustomIdentity())
        self._init_weights()
        self.fix_init_weight()

        if isinstance(self.head, nn.Dense):
            self.head.weight.set_data(self.head.weight * init_scale)
            self.head.bias.set_data(self.head.bias * init_scale)

    def _init_weights(self):
        for _, m in self.parameters_and_names():
            if isinstance(m, nn.Dense):
                m.weight.set_data(
                    init.initializer(
                        init.TruncatedNormal(sigma=0.02),
                        m.weight.shape,
                        m.weight.dtype
                    )
                )
                if isinstance(m, nn.Dense) and m.bias is not None:
                    m.bias.set_data(
                        init.initializer(
                            init.Constant(0),
                            m.bias.shape,
                            m.bias.dtype
                        )
                    )
            elif isinstance(m, nn.LayerNorm):
                m.gamma.set_data(
                    init.initializer(
                        init.Constant(1.0),
                        m.gamma.shape,
                        m.gamma.dtype
                    )
                )
                m.beta.set_data(
                    init.initializer(
                        init.Constant(0),
                        m.beta.shape,
                        m.beta.dtype
                    )
                )

    def fix_init_weight(self):
        def rescale(param: ms.Parameter, layer_id):
            assert layer_id > 0
            param.set_data(param / math.sqrt(2.0 * layer_id))

        for layer_id, layer in enumerate(self.blocks):
            assert isinstance(layer, Block)
            rescale(layer.attn.proj.weight, layer_id + 1)
            rescale(layer.mlp.fc2.weight, layer_id + 1)

    def get_num_layers(self):
        return len(self.blocks)

    def get_classifier(self):
        return self.head

    def reset_classifier(self, num_classes):
        self.num_classes = num_classes
        self.head = (nn.Dense(self.embed_dim, num_classes)
                     if num_classes > 0 else CustomIdentity())

    def forward_features(self, x):
        x = self.patch_embed(x)
        batch_size, _, _ = x.shape

        cls_token = ms.ops.cast(self.cls_token, x.dtype)
        cls_token = ops.tile(cls_token, (batch_size, 1, 1))

        x = ops.concat([cls_token, x], axis=1)
        if self.pos_embed is not None:
            x = x + ms.ops.cast(self.pos_embed, x.dtype)
        x = self.pos_drop(x)

        rel_pos_bias = (self.rel_pos_bias()
                        if self.rel_pos_bias is not None else None)

        for blk in self.blocks:
            x = blk(x, rel_pos_bias=rel_pos_bias)

        x = self.norm(x)
        if self.fc_norm is not None:
            t = x[:, 1:, :]
            return self.fc_norm(ops.mean(t, 1))
        return x[:, 0]

    def construct(self, x):
        x = self.forward_features(x)
        x = self.head(x)
        return x

    def no_weight_decay(self):
        return {'pos_embed', 'cls_token'}

    def get_intermediate_layers(self, x):
        x = self.patch_embed(x)
        batch_size, _, _ = x.shape

        cls_token = ops.tile(self.cls_token, (batch_size, 1, 1))

        x = ops.concat([cls_token, x], axis=1)
        if self.pos_embed is not None:
            x = x + ms.ops.cast(self.pos_embed, x.dtype)
        x = self.pos_drop(x)

        rel_pos_bias = (self.rel_pos_bias()
                        if self.rel_pos_bias is not None else None)

        features = []
        for blk in self.blocks:
            x = blk(x, rel_pos_bias=rel_pos_bias)
            features.append(x)

        return features


def beit_base_patch16_224(**kwargs):
    model = VisionTransformer(
        patch_size=16, embed_dim=768, depth=12, num_heads=12, mlp_ratio=4,
        qkv_bias=True, norm_layer=partial(nn.LayerNorm, epsilon=1e-6), **kwargs)
    return model


def beit_base_patch16_384(**kwargs):
    model = VisionTransformer(
        img_size=384, patch_size=16, embed_dim=768, depth=12, num_heads=12,
        mlp_ratio=4, qkv_bias=True,
        norm_layer=partial(nn.LayerNorm, epsilon=1e-6), **kwargs)
    return model


def beit_large_patch16_224(**kwargs):
    model = VisionTransformer(
        patch_size=16, embed_dim=1024, depth=24, num_heads=16, mlp_ratio=4,
        qkv_bias=True, norm_layer=partial(nn.LayerNorm, epsilon=1e-6), **kwargs)
    return model


def beit_large_patch16_384(**kwargs):
    model = VisionTransformer(
        img_size=384, patch_size=16, embed_dim=1024, depth=24, num_heads=16,
        mlp_ratio=4, qkv_bias=True,
        norm_layer=partial(nn.LayerNorm, epsilon=1e-6), **kwargs)
    return model


def beit_large_patch16_512(**kwargs):
    model = VisionTransformer(
        img_size=512, patch_size=16, embed_dim=1024, depth=24, num_heads=16,
        mlp_ratio=4, qkv_bias=True,
        norm_layer=partial(nn.LayerNorm, epsilon=1e-6), **kwargs)
    return model
