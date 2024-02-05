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
"""Optimized criterion functionality."""
import logging

import mindspore.nn as nn
from mindspore import Tensor
from mindspore import ops
from mindspore.common import dtype as mstype
from mindspore.nn.loss.loss import LossBase
from mindspore.ops import functional as F
from mindspore.ops import operations as P


class SoftTargetCrossEntropy(LossBase):
    """SoftTargetCrossEntropy for MixUp Augment"""

    def __init__(self):
        super(SoftTargetCrossEntropy, self).__init__()
        self.mean_ops = P.ReduceMean(keep_dims=False)
        self.sum_ops = P.ReduceSum(keep_dims=False)
        self.log_softmax = P.LogSoftmax()

    def construct(self, logits, labels):
        logits = P.Cast()(logits, mstype.float32)
        labels = P.Cast()(labels, mstype.float32)
        loss = self.sum_ops((-1 * labels) * self.log_softmax(logits), -1)
        return self.mean_ops(loss)


class CrossEntropySmooth(LossBase):
    """CrossEntropy"""

    def __init__(
            self, sparse=True, reduction='mean', smooth_factor=0.,
            num_classes=1000
    ):
        super(CrossEntropySmooth, self).__init__()
        self.onehot = P.OneHot()
        self.sparse = sparse
        self.on_value = Tensor(1.0 - smooth_factor, mstype.float32)
        self.off_value = Tensor(
            1.0 * smooth_factor / (num_classes - 1), mstype.float32
        )
        self.ce = nn.SoftmaxCrossEntropyWithLogits(reduction=reduction)
        self.cast = ops.Cast()

    def construct(self, logits, labels):
        if self.sparse:
            labels = self.onehot(
                labels, F.shape(logits)[1], self.on_value, self.off_value
            )
        labels = P.Cast()(labels, mstype.float32)
        logits = P.Cast()(logits, mstype.float32)
        loss2 = self.ce(logits, labels)
        return loss2


def get_criterion(args):
    """Get loss function from args.label_smooth and args.mixup"""
    assert 0. <= args.label_smoothing <= 1.

    if args.mixup > 0. or args.cutmix > 0.:
        logging.info('Using MixBatch')
        # smoothing is handled with mixup label transform
        criterion = SoftTargetCrossEntropy()
    elif args.label_smoothing > 0.:
        logging.info('Using label smoothing')
        criterion = CrossEntropySmooth(sparse=True, reduction="mean",
                                       smooth_factor=args.label_smoothing,
                                       num_classes=args.num_classes)
    else:
        logging.info('Using Simple CE')
        criterion = CrossEntropySmooth(
            sparse=True, reduction="mean", num_classes=args.num_classes
        )

    return criterion


class NetWithLoss(nn.Cell):
    """
    NetWithLoss: Only support Network with Classification.
    """

    def __init__(self, model, criterion):
        super(NetWithLoss, self).__init__()
        self.model = model
        self.criterion = criterion

    def construct(self, *inputs, **kwargs):
        data = inputs[0]
        label = inputs[1]
        predict = self.model(data)
        loss = self.criterion(predict, label)
        return loss
