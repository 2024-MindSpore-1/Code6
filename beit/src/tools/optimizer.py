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
"""Optimizer creation."""

import logging
import json
from typing import List, Dict

import numpy as np
import mindspore as ms
import mindspore.nn as nn
from mindspore.nn.optim import AdamWeightDecay, Adam, SGD
from mindspore.nn.optim.momentum import Momentum

from src.tools.schedulers import get_policy


def get_learning_rate(args, batch_num):
    """Get learning rate"""
    return get_policy(args.lr_scheduler)(args, batch_num)


def update_learning_rate_for_groups(
        param_groups: List[Dict], learning_rate: np.ndarray):
    logging.debug('Learning rate len: %s', len(learning_rate))
    for i, param_group in enumerate(param_groups):
        lr_scale = param_group.pop('lr_scale')
        param_group['lr'] = ms.Tensor(
            (learning_rate.copy() * lr_scale).astype(np.float32))
    for i, param_group in enumerate(param_groups):
        logging.debug('Mean lr for group #%s: %s (%s)',
                      i, param_group["lr"], param_group["lr"])


def get_optimizer_beit(args, model, batch_num, get_num_layer=None,
                       get_layer_scale=None, filter_bias_and_bn=True,
                       skip_list=None):
    """Get optimizer for training"""
    logging.info('When using train_wrapper, using optimizer %s',
                 args.optimizer)
    optim_type = args.optimizer.lower()
    weight_decay = args.weight_decay
    use_beit_groups = False
    if weight_decay and filter_bias_and_bn:
        skip = {}
        if skip_list is not None:
            skip = skip_list
        elif hasattr(model, 'np_weight_decay'):
            skip = model.no_weight_decay()
        logging.info('Skip list: %s', skip)

        params = get_param_groups_beit(model, weight_decay, skip,
                                       get_num_layer, get_layer_scale)
        use_beit_groups = True
    else:
        params = model.trainable_params()
    learning_rate = get_learning_rate(args, batch_num)
    step = int(args.start_epoch * batch_num)
    train_step = len(learning_rate)
    learning_rate = learning_rate[step:]
    logging.info('Get LR from epoch: %d', args.start_epoch)
    logging.info('Start step: %d', step)
    logging.info('Total step: %d', train_step)

    if use_beit_groups:
        update_learning_rate_for_groups(params, learning_rate)

    logging.info('learning_rate %f', np.max(learning_rate))
    if optim_type == 'momentum':
        optim = Momentum(
            params=params,
            learning_rate=learning_rate,
            momentum=args.momentum,
            weight_decay=args.weight_decay
        )
    elif optim_type == 'adamw':
        optim = AdamWeightDecay(
            params=params,
            learning_rate=learning_rate,
            beta1=args.beta[0],
            beta2=args.beta[1],
            eps=args.eps,
            weight_decay=args.weight_decay
        )
    elif optim_type == 'adam':
        optim = Adam(
            params=params,
            learning_rate=learning_rate,
            beta1=args.beta[0],
            beta2=args.beta[1],
            eps=args.eps
        )
    elif optim_type == 'sgd':
        optim = SGD(
            params=params,
            learning_rate=learning_rate,
            momentum=args.momentum,
            weight_decay=args.weight_decay,
        )
    else:
        raise ValueError(f'optimizer {optim_type} is not supported')

    return optim


def get_param_groups_beit(network: nn.Cell, weight_decay, skip_list=(),
                          get_num_layer=None, get_layer_scale=None):
    """get param groups"""
    parameter_group_names = {}
    parameter_group_vars = {}

    # Iterate over params that requires grad
    for param in network.trainable_params():
        assert isinstance(param, ms.Parameter)
        name = param.name
        if not param.requires_grad:
            continue

        if len(param.shape) == 1 or name.endswith('.beta') or name in skip_list:
            group_name = 'no_decay'
            this_weight_decay = 0.0
        else:
            group_name = 'decay'
            this_weight_decay = weight_decay

        if get_num_layer is not None:
            layer_id = get_num_layer(name)
            group_name = f'layer_{layer_id}_{group_name}'
        else:
            layer_id = None
        logging.debug('Layer id: %s. Group name: %s',
                      layer_id, group_name)

        if group_name not in parameter_group_names:
            if get_layer_scale is not None:
                scale = get_layer_scale(layer_id)
            else:
                scale = 1.0

            parameter_group_names[group_name] = {
                'weight_decay': this_weight_decay,
                'params': [],
                'lr_scale': scale
            }
            parameter_group_vars[group_name] = {
                'weight_decay': this_weight_decay,
                'params': [],
                'lr_scale': scale
            }
        parameter_group_vars[group_name]['params'].append(param)
        parameter_group_names[group_name]['params'].append(name)

    logging.info('Param groups = %s', json.dumps(parameter_group_names,
                                                 indent=2))
    return list(parameter_group_vars.values())
