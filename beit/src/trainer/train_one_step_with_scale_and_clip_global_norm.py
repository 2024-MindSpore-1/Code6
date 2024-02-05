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
"""TrainOneStepWithLossScaleCellGlobalNormClip"""
import mindspore.nn as nn
from mindspore.common import RowTensor
from mindspore.ops import composite, functional, operations

_grad_scale = composite.MultitypeFuncGraph('grad_scale')
reciprocal = operations.Reciprocal()


@_grad_scale.register('Tensor', 'Tensor')
def tensor_grad_scale(scale, grad):
    return grad * functional.cast(reciprocal(scale), functional.dtype(grad))


@_grad_scale.register('Tensor', 'RowTensor')
def tensor_grad_scale_row_tensor(scale, grad):
    return RowTensor(
        grad.indices,
        grad.values * functional.cast(
            reciprocal(scale), functional.dtype(grad.values)
        ),
        grad.dense_shape
    )


_grad_overflow = composite.MultitypeFuncGraph('_grad_overflow')
grad_overflow = operations.FloatStatus()


class TrainOneStepWithLossScaleCellGlobalNormClip(
        nn.TrainOneStepWithLossScaleCell
):
    """
    Encapsulation class of ConvMixer network training.

    Append an optimizer to the training network after that the construct
    function can be called to create the backward graph.

    Parameters
    ----------
    network: Cell
        The training network. Note that loss function should have been added.
    optimizer: Optimizer
        Optimizer for updating the weights.
    scale_sense: Cell
        The adjust parameter. Default: 1.0.
    use_global_norm: bool
        Whether apply global norm before optimizer. Default: True.
    clip_global_norm_value: bool
        The norm coefficient to scale gradients. Default: 1.0.
    """

    def __init__(self, network, optimizer,
                 scale_sense, use_global_norm=True,
                 clip_global_norm_value=1.0):
        super(TrainOneStepWithLossScaleCellGlobalNormClip, self).__init__(
            network, optimizer, scale_sense
        )
        self.use_global_norm = use_global_norm
        self.clip_global_norm_value = clip_global_norm_value
        self.print = operations.Print()

    def construct(self, *inputs):
        """Construct."""
        weights = self.weights
        loss = self.network(*inputs)
        scaling_sens = self.scale_sense

        status, scaling_sens = self.start_overflow_check(loss, scaling_sens)

        scaling_sens_filled = composite.ones_like(loss) * functional.cast(
            scaling_sens, functional.dtype(loss)
        )
        grads = self.grad(self.network, weights)(*inputs, scaling_sens_filled)
        if self.use_global_norm:
            grads = composite.clip_by_global_norm(
                grads,
                clip_norm=self.clip_global_norm_value
            )
        grads = self.hyper_map(
            functional.partial(_grad_scale, scaling_sens),
            grads
        )
        # apply grad reducer on grads
        grads = self.grad_reducer(grads)
        # get the overflow buffer
        cond = self.get_overflow_status(status, grads)
        overflow = self.process_loss_scale(cond)
        # if there is no overflow, do optimize
        if not overflow:
            self.optimizer(grads)
        else:
            self.print('=============Over Flow, skipping=============')
        return loss
