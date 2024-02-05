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
# https://github.com/huggingface/pytorch-image-models
# repository and modified.
# ============================================================================
"""Pytorch Timm implementation based DropOut layer."""

import mindspore.nn as nn
import mindspore.ops as ops
import mindspore.numpy as np
from mindspore import dtype as mstype


class DropPath(nn.Cell):
    """Pytorch Timm implementation based DropOut layer."""

    def __init__(self, keep_prob):
        """DropOut layer initializer."""
        super(DropPath, self).__init__()
        self.keep_prob = keep_prob
        self.div = ops.Div()

    def construct(self, x):
        if not self.training:
            return x

        if self.keep_prob == 1:
            return x

        shape = (x.shape[0],) + (1,) * (x.ndim - 1)
        random_tensor = self.keep_prob + np.rand(shape, dtype=mstype.float32)
        random_tensor = ops.Floor()(random_tensor)  # binarize
        output = self.div(x, self.keep_prob) * random_tensor
        return output
