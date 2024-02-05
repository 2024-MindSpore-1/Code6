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
"""Train script."""
from functools import reduce
import sys
import logging
from pathlib import Path

from mindspore import nn
from mindspore import Model
from mindspore import context
from mindspore.common import set_seed
from mindspore.nn.metrics.accuracy import Accuracy
from mindspore.nn.metrics.loss import Loss as MetricLoss

from src.config import get_config
from src.tools.cell import cast_amp
from src.tools.criterion import get_criterion, NetWithLoss
from src.tools.get_misc import get_dataset, set_device, get_model, \
    load_pretrained, get_train_one_step, get_directories, save_config, \
    get_callbacks, config_logging, add_logging_file_handler, \
    LayerDecayValueAssigner
from src.tools.optimizer import get_optimizer_beit


def create_assigner(layer_decay, num_layers):
    if layer_decay < 1.0:
        lr_scales = list(layer_decay ** (num_layers + 1 - i)
                         for i in range(num_layers + 2))
        logging.info('LR scaler values: %s', lr_scales)
        assigner = LayerDecayValueAssigner(lr_scales)
        logging.info('Use LayerDecayValueAssigner.')
    else:
        assigner = None
        logging.info('LayerDecayValueAssigner is not used.')
    return assigner


def main():
    args = get_config()

    set_seed(args.seed)
    mode = context.PYNATIVE_MODE if args.pynative_mode else context.GRAPH_MODE
    context.set_context(mode=mode, device_target=args.device_target,
                        enable_graph_kernel=False)
    rank = set_device(args)

    summary_dir, ckpt_dir, best_ckpt_dir, logs_dir = get_directories(
        args.arch,
        args.summary_root_dir,
        args.ckpt_root_dir,
        args.best_ckpt_root_dir,
        args.logs_root_dir,
        args.model_postfix,
        rank,
    )
    add_logging_file_handler(str(Path(logs_dir) / 'train_log'))

    net = get_model(args.image_size, args.arch, args.num_classes,
                    not args.disable_approximate_gelu,
                    use_rel_pos_bias=args.rel_pos_bias,
                    use_abs_pos_emb=args.abs_pos_emb,
                    init_values=args.layer_scale_init_value)
    if args.pretrained != '':
        load_pretrained(args, net, args.exclude_epoch_state)

    logging.info('Number of parameters: %d',
                 sum(reduce(lambda x, y: x * y, params.shape)
                     for params in net.trainable_params()))

    num_layers = net.get_num_layers()
    assigner = create_assigner(args.layer_decay, num_layers)

    cast_amp(net, args)
    criterion = get_criterion(args)
    net_with_loss = NetWithLoss(net, criterion)

    data = get_dataset(args)
    batch_num = data.train_dataset.get_dataset_size()
    skip_weight_decay_list = net.no_weight_decay()
    if args.disable_weight_decay_on_rel_pos_bias:
        for i in range(num_layers):
            skip_weight_decay_list.add(
                f'blocks.{i}.attn.relative_position_bias_table')
    skip_weight_decay_list = skip_weight_decay_list.union({
        f'model.{name}' for name in skip_weight_decay_list})

    optimizer = get_optimizer_beit(
        args, net, batch_num,
        get_num_layer=assigner.get_layer_id if assigner else None,
        get_layer_scale=assigner.get_scale if assigner else None,
        skip_list=skip_weight_decay_list,
    )

    net_with_loss = get_train_one_step(args, net_with_loss, optimizer)

    cast_amp(net, args)
    eval_network = nn.WithEvalCell(
        net, criterion, args.amp_level in ['O2', 'O3', 'auto']
    )
    eval_indexes = [0, 1, 2]
    model = Model(net_with_loss,
                  metrics={'acc': Accuracy(), 'loss': MetricLoss()},
                  eval_network=eval_network, eval_indexes=eval_indexes)

    callbacks = get_callbacks(
        args.arch,
        data.train_dataset.get_dataset_size(),
        data.val_dataset.get_dataset_size(),
        summary_dir,
        logs_dir,
        ckpt_dir,
        best_ckpt_dir,
        rank,
        args.ckpt_save_every_step,
        args.ckpt_save_every_seconds,
        args.ckpt_keep_num,
        args.best_ckpt_num,
        print_loss_every=args.print_loss_every,
        collect_freq=args.summary_loss_collect_freq,
        collect_graph=args.dump_graph
    )
    save_config(args, sys.argv, best_ckpt_dir)

    logging.info(
        'Number of samples in dataset: train=%d, val=%d',
        data.train_dataset.get_dataset_size(),
        data.val_dataset.get_dataset_size()
    )

    logging.info('begin train')
    model.fit(int(args.epochs - args.start_epoch),
              data.train_dataset, data.val_dataset,
              callbacks=callbacks,
              dataset_sink_mode=args.dataset_sink_mode,
              valid_dataset_sink_mode=False)
    logging.info('train success')


if __name__ == '__main__':
    config_logging()
    main()
