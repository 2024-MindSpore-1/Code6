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

import mindspore.nn as nn
from mindspore import ops


class PatchEmbed(nn.Cell):
    """Image to Patch Embedding."""
    def __init__(self, img_size=224, patch_size=16, in_chans=3, embed_dim=768):
        super().__init__()
        if isinstance(img_size, int):
            img_size = (img_size, img_size)
        if isinstance(patch_size, int):
            patch_size = (patch_size, patch_size)
        num_patches = ((img_size[1] // patch_size[1])
                       * (img_size[0] // patch_size[0]))
        self.patch_shape = ((img_size[0] // patch_size[0], img_size[1]
                             // patch_size[1]))
        self.img_size = img_size
        self.patch_size = patch_size
        self.num_patches = num_patches
        self.embed_dim = embed_dim

        self.proj = nn.Conv2d(in_chans, embed_dim, kernel_size=patch_size,
                              stride=patch_size, has_bias=True)

    def construct(self, x):
        B, _, H, W = x.shape
        # FIXME look at relaxing size constraints
        assert H == self.img_size[0] and W == self.img_size[1], \
            f'Input image size ({H}*{W}) doesn\'t match model' \
            f' ({self.img_size[0]}*{self.img_size[1]}).'
        x = self.proj(x)
        x = ops.reshape(x, (B, self.embed_dim, self.num_patches))
        x = ops.transpose(x, (0, 2, 1))
        return x
