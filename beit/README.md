# Contents

<!-- TOC -->

* [Contents](#contents)
* [BEiT Description](#beit-description)
* [Model pretraining process](#model-pretraining-process)
* [Dataset](#dataset)
* [Environment Requirements](#environment-requirements)
* [Quick Start](#quick-start)
    * [Prepare the model](#prepare-the-model)
    * [Run the scripts](#run-the-scripts)
* [Script Description](#script-description)
    * [Script and Sample Code](#script-and-sample-code)
        * [Directory structure](#directory-structure)
        * [Script Parameters](#script-parameters)
    * [Training Process](#training-process)
        * [Training on GPU](#training-on-gpu)
            * [Training on multiple GPUs](#training-on-multiple-gpus)
            * [Training on single GPU](#training-on-single-gpu)
            * [Arguments description](#arguments-description)
        * [Training with CPU](#training-with-cpu)
        * [Transfer training](#transfer-training)
    * [Evaluation](#evaluation)
        * [Evaluation process](#evaluation-process)
            * [Evaluation with checkpoint](#evaluation-with-checkpoint)
        * [Evaluation results](#evaluation-results)
    * [Inference](#inference)
        * [Inference with checkpoint](#inference-with-checkpoint)
        * [Inference results](#inference-results)
    * [Export](#export)
        * [Export process](#export-process)
        * [Export results](#export-results)
* [Model Description](#model-description)
    * [Performance](#performance)
        * [Training Performance](#training-performance)
* [Description of Random Situation](#description-of-random-situation)
* [ModelZoo Homepage](#modelzoo-homepage)

<!-- TOC -->

# [BEiT Description](#contents)

Authors introduced a self-supervised vision representation model BEIT,
which stands for Bidirectional Encoder representation from Image Transformers.
Following BERT developed in the natural language processing area, they propose
a masked image modeling task to pretrain vision Transformers. Specifically,
each image has two views in our pre-training, i.e., image patches (such as
16×16 pixels), and visual tokens (i.e., discrete tokens). Authors first
“tokenize” the original image into visual tokens. Then they randomly mask
some image patches and fed them into the backbone Transformer.
The pre-training objective is to recover the original visual tokens based on
the corrupted image patches. After pre-training BEIT, we directly fine-tune
the model parameters on downstream tasks by appending task layers upon the pretrained encoder.

[Paper](https://arxiv.org/pdf/2106.08254.pdf): Hangbo Bao, Li Dong, Songhao Piao,
Furu Wei. 2022.

# [Model pretraining process](#contents)

Before pre-training, authors learn an “image tokenizer” via autoencoding-style
reconstruction, where an image is tokenized into discrete visual tokens
according to the learned vocabulary. During pre-training, each image has
two views, i.e., image patches, and visual tokens. Authors randomly mask
some proportion of image patches (gray patches in the figure) and replace
them with a special mask embedding. Then the patches are fed to a backbone
vision Transformer. The pre-training task aims at predicting the visual tokens
of the original image based on the encoding vectors of the corrupted image.

Authors use the standard Transformer as the backbone network to directly
compare results with previous work in terms of the network architecture.

# [Dataset](#contents)

Note that you can run the scripts based on the dataset mentioned in original
paper or widely used in relevant domain/network architecture. In the following
sections, we will introduce how to run the scripts using the related dataset
below.

Dataset used: [ImageNet2012](http://www.image-net.org/)

* Dataset size：146.6G
    * Train：139.3G，1281167 images
    * Val：6.3G，50000 images
    * Annotations：each image is in label folder
* Data format：images sorted by label folders
    * Note：Data will be processed in imagenet.py

# [Environment Requirements](#contents)

* Install [MindSpore](https://www.mindspore.cn/install/en).
* Download the dataset ImageNet dataset.
* We use ImageNet2012 as training dataset in this example by default, and you
  can also use your own datasets.

For ImageNet-like dataset the directory structure is as follows:

```shell
 .
 └─imagenet
   ├─train
   │ ├─class1
   │ │ ├─image1.jpeg
   │ │ ├─image2.jpeg
   │ │ └─...
   │ ├─...
   │ └─class1000
   ├─val
   │ ├─class1
   │ ├─...
   │ └─class1000
   └─test
```

# [Quick Start](#contents)

## Prepare the model

1. Chose the model by changing the `arch` in `configs/beit_XXX_patch16_YYY.yaml`. `XXX` is the corresponding model architecture configuration,
   allowed options are: `base`, `large`. `YYY` is an input image size which may be `224` or `384`.
2. Change the dataset config in the corresponding config. `configs/beit_XXX_patch16_YYY.yaml`.
   Especially, set the correct path to data.
3. Change the hardware setup.
4. Change the artifacts setup to set the correct folders to save checkpoints and mindinsight logs.

Note, that you also can pass the config options as CLI arguments, and they are
preferred over config in YAML.
Also, all possible options must be defined in `yaml` config file.

## Run the scripts

After installing MindSpore via the official website,
you can start training and evaluation as follows.

```shell
# distributed training on GPU
bash run_distribute_train_gpu.sh CONFIG [--num_devices NUM_DEVICES] [--device_ids DEVICE_IDS (e.g. '0,1,2,3')] [--checkpoint CHECKPOINT] [--extra *EXTRA_ARGS]

# standalone training on GPU
bash run_standalone_train_gpu.sh CONFIG [--device DEVICE_ID (default: 0)] [--checkpoint CHECKPOINT] [--extra *EXTRA_ARGS]

# run eval on GPU
bash run_eval_gpu.sh CONFIG [--device DEVICE_ID (default: 0)] [--checkpoint CHECKPOINT] [--extra *EXTRA_ARGS]
```

# [Script Description](#contents)

## [Script and Sample Code](#contents)

### Directory structure

```shell
beit
├── scripts
│   ├── run_distribute_train_gpu.sh                          # shell script for distributed training on GPU
│   ├── run_eval_gpu.sh                                      # shell script for evaluation on GPU
│   ├── run_infer_gpu.sh                                     # shell script for inference on GPU
│   └── run_standalone_train_gpu.sh                          # shell script for training on GPU
├── src
│  ├── configs
│  │  ├── beit_base_patch16_224_ft1k.yaml                    # example of configuration for BEiT-base
│  │  ├── beit_base_patch16_384_pt22k_ft1k.yaml              # example of configuration for BEiT-base
│  │  └── beit_large_patch16_224_ft1k.yaml                   # example of configuration for BEiT-large
│  ├── data
│  │  ├── augment
│  │  │  ├── __init__.py
│  │  │  ├── auto_augment.py                                 # augmentation set builder
│  │  │  ├── mixup.py                                        # MixUp augmentation
│  │  │  └── random_erasing.py                               # Random Erasing augmentation
│  │  ├── __init__.py
│  │  └── imagenet.py                                        # wrapper for reading ImageNet dataset
│  ├── layers                                                # layers used in BEiT implementation
│  │  ├── __init__.py
│  │  ├── attention.py                                       # Attention layer
│  │  ├── custom_identity.py                                 # Identity layer
│  │  ├── drop_path_timm.py                                  # Implementation of drop path the same way as in TIMM
│  │  ├── mlp.py                                             # MLP block implementation
│  │  ├── patch_embed.py                                     # Layer to get patch embeddings from image
│  │  ├── relative_position_bias.py                          # Relative position bias layer
│  │  └── transformer_block.py                               # Main block of transformer architecture
│  ├── tools
│  │  ├── __init__.py
│  │  ├── callback.py                                       # callback functions (implementation)
│  │  ├── cell.py                                            # tune model layers/parameters
│  │  ├── criterion.py                                       # model training objective function (implementation)
│  │  ├── get_misc.py                                        # initialize optimizers, callbacks and other arguments for training process
│  │  ├── optimizer.py                                       # model optimizer function (implementation)
│  │  └── schedulers.py                                      # training (LR) scheduling function (implementation)
│  ├── trainer
│  │  ├── __init__.py
│  │  ├── ema.py                                             # EMA implementation
│  │  ├── train_one_step_with_ema.py                         # utils for training with EMA
│  │  └── train_one_step_with_scale_and_clip_global_norm.py  # utils for training with gradient clipping
│  ├── __init__.py
│  ├── config.py                                             # YAML and CLI configuration parser
│  └── vit.py                                                # BEiT architecture
├── eval.py                                                  # evaluation script
├── export.py                                                # export checkpoint files into ONNX, MINDIR and AIR formats
├── infer.py                                                 # inference script
├── README.md                                                # BEiT descriptions
├── requirements.txt                                         # python requirements
└── train.py                                                 # training script
```

### [Script Parameters](#contents)

```yaml
# ===== Dataset ===== #
dataset: ImageNet
data_url: /data/imagenet/ILSVRC/Data/CLS-LOC/
train_dir: train
val_dir: validation_preprocess
train_num_samples: -1
val_num_samples: -1
imagenet_default_mean_and_std: False

# ===== Augmentations ==== #
auto_augment: rand-m9-mstd0.5-inc1
aa_interpolation: bicubic
re_mode: pixel
re_prob: 0.25
re_count: 1
cutmix: 1.0
mixup: 0.8
mixup_prob: 1.0
mixup_mode: batch
mixup_off_epoch: 0.0
switch_prob: 0.5
label_smoothing: 0.1
min_crop: 0.08
crop_pct: 0.9

# ===== Optimizer ======== #
optimizer: adamw
beta: [ 0.9, 0.999 ]
eps: 1.0e-8
base_lr: 2.0e-3  # 2e-3, 3e-3, 4e-3, 5e-3  # Increased if layer decay is used
min_lr: 1.0e-6
lr_scheduler: cosine_lr
lr_adjust: 30
lr_gamma: 0.97
momentum: 0.9
weight_decay: 0.05
layer_decay: 0.9


# ===== Network training config ===== #
epochs: 100
batch_size: 64
is_dynamic_loss_scale: True
loss_scale: 1024
num_parallel_workers: 8
start_epoch: 0
warmup_length: 20
warmup_lr: 0.000007
# Gradient clipping
use_clip_grad_norm: False
clip_grad_norm: 1.0
# Load pretrained setup
exclude_epoch_state: True
seed: 0
# EMA
with_ema: False
ema_decay: 0.9999

pynative_mode: False
dataset_sink_mode: True

# ==== Model arguments ==== #
arch: beit_base_patch16_224
amp_level: O0
file_format: ONNX
pretrained: ''
image_size: 224
num_classes: 1000
drop: 0.0
drop_block: 0.0
drop_path: 0.1  # Stochasitc depth
disable_approximate_gelu: False
use_pytorch_maxpool: False
layer_scale_init_value: 0.1  # 0.1 for base, 1e-5 for large. set 0 to disable layer scale
rel_pos_bias: True
abs_pos_emb: False


# ===== Hardware setup ===== #
device_id: 0
device_num: 1
device_target: GPU

# ===== Callbacks setup ===== #
summary_root_dir: /experiments/summary_dir/
ckpt_root_dir: /experiments/checkpoints/
best_ckpt_root_dir: /experiments/best_checkpoints/
logs_root_dir: /experiments/logs/
ckpt_keep_num: 10
best_ckpt_num: 5
ckpt_save_every_step: 0
ckpt_save_every_seconds: 1800
print_loss_every: 100
summary_loss_collect_freq: 20
model_postfix: 0
collect_input_data: False
dump_graph: False
```

## [Training Process](#contents)

In the examples below the only required argument is YAML config file.

### Training on GPU

#### Training on multiple GPUs

Usage

```shell
# distributed training on GPU
run_distribute_train_gpu.sh CONFIG [--num_devices NUM_DEVICES] [--device_ids DEVICE_IDS (e.g. '0,1,2,3')] [--checkpoint CHECKPOINT] [--extra *EXTRA_ARGS]
```

Example

```bash
# Without extra arguments
bash run_distribute_train.sh ../src/configs/beit_base_patch16_224_ft1k.yaml --num_devices 4 --device_ids 0,1,2,3 --checkpoint /experiments/beit_base_patch16_224_pt22k.ckpt

# With extra arguments
bash run_distribute_train.sh ../src/configs/beit_base_patch16_224_ft1k.yaml --num_devices 4 --device_ids 0,1,2,3 --checkpoint /experiments/beit_base_patch16_224_pt22k.ckpt --extra --amp_level O0 --batch_size 100 --start_epoch 0 --num_parallel_workers 8
```

#### Training on single GPU

Usage

```shell
# standalone training on GPU
run_standalone_train_gpu.sh CONFIG [--device DEVICE_ID (default: 0)] [--checkpoint CHECKPOINT] [--extra *EXTRA_ARGS]
```

Example

```bash
# Without extra arguments:
bash run_standalone_train.sh ../src/configs/beit_base_patch16_224_ft1k.yaml --device 0 --checkpoint /experiments/beit_base_patch16_224_pt22k.ckpt
# With extra arguments:
bash run_standalone_train.sh ../src/configs/beit_base_patch16_224_ft1k.yaml --device 0 --checkpoint /experiments/beit_base_patch16_224_pt22k.ckpt --extra --amp_level O0 --batch_size 48 --start_epoch 0 --num_parallel_workers 8
```

Running the Python scripts directly is also allowed.

```shell
# show help with description of options
python train.py --help

# standalone training on GPU
python train.py --config_path path/to/config.yaml [OTHER OPTIONS]
```

#### Arguments description

`bash` scripts have the following arguments

* `CONFIG`: path to YAML file with configuration.
* `--num_devices`: the device number for distributed train.
* `--device_ids`: ids of devices to train.
* `--checkpoint`: path to checkpoint to continue training from.
* `--extra`: any other arguments of `train.py`.

By default, training process produces four folders (configured):

* Best checkpoints
* Current checkpoints
* Mindinsight logs
* Terminal logs

### Training with CPU

**It is recommended to run models on GPU.**

### Transfer training

You can train your own model based on pretrained classification
model. You can perform transfer training by following steps.

1. Convert your own dataset to ImageFolderDataset style. Otherwise, you have to add your own data preprocess code.
2. Change `beit_XXX_patch16_YYY.yaml` according to your own dataset, especially the `num_classes`.
3. Prepare a pretrained checkpoint. You can load the pretrained checkpoint by `--pretrained` argument.
4. Build your own bash scripts using new config and arguments for further convenient.

## [Evaluation](#contents)

### Evaluation process

#### Evaluation with checkpoint

Usage

```shell
run_eval_gpu.sh CONFIG [--device DEVICE_ID (default: 0)] [--checkpoint CHECKPOINT] [--extra *EXTRA_ARGS]
```

Examples

```shell

# Without extra args
bash run_eval_gpu.sh  ../src/configs/beit_base_patch16_224_ft1k.yaml --checkpoint /data/models/beit_base.ckpt

# With extra args
bash run_eval_gpu.sh  ../src/configs/beit_base_patch16_224_ft1k.yaml --checkpoint /data/models/beit_base.ckpt --extra --data_url /data/imagenet/ --val_dir validation_preprocess
```

Running the Python script directly is also allowed.

```shell
# run eval on GPU
python eval.py --config_path path/to/config.yaml [OTHER OPTIONS]
```

The Python script has the same arguments as the training script (`train.py`),
but it uses only validation subset of dataset to evaluate.
Also, `--pretrained` is expected.

### Evaluation results

Results will be printed to console.

```shell
# checkpoint evaluation result
eval results: {'Loss': TODO, 'Top1-Acc': TODO, 'Top5-Acc': TODO}
```

## [Inference](#contents)

Inference may be performed with checkpoint or ONNX model.

### Inference with checkpoint

Usage

```shell
run_infer_gpu.sh DATA [--checkpoint CHECKPOINT] [--arch ARCHITECTURE] [--output OUTPUT_JSON_FILE (default: predictions.json)]
```

Example for folder

```shell
bash run_infer_gpu.sh /data/images/cheetah/ --checkpoint /data/models/beit_base.ckpt --arch beit_base_patch16_224
```

Example for single image

```shell
bash run_infer_gpu.sh /data/images/American\ black\ bear/ILSVRC2012_validation_preprocess_00011726.JPEG --checkpoint /data/models/beit_base.ckpt --arch beit_base_patch16_224
```

### Inference results

Predictions will be output in logs and saved in JSON file. File content is
dictionary where key is image path and value is class number. It's supported
predictions for folder of images (png, jpeg file in folder root) and single image.

Results for single image in console

```shell
/data/images/cheetah/ILSVRC2012_validation_preprocess_00011726.JPEG (class: 295)
```

Results for single image in JSON file

```json
{
 "/data/images/American black bear/ILSVRC2012_validation_preprocess_00011726.JPEG": 295
}
```

Results for directory in console

```shell
/data/images/American black bear/ILSVRC2012_validation_preprocess_00011726.JPEG (class: 295)
/data/images/American black bear/ILSVRC2012_validation_preprocess_00014576.JPEG (class: 294)
/data/images/American black bear/ILSVRC2012_validation_preprocess_00000865.JPEG (class: 295)
/data/images/American black bear/ILSVRC2012_validation_preprocess_00007093.JPEG (class: 295)
/data/images/American black bear/ILSVRC2012_validation_preprocess_00014029.JPEG (class: 295)

```

Results for directory in JSON file

```json
{
 "/data/images/American black bear/ILSVRC2012_validation_preprocess_00011726.JPEG": 295,
 "/data/images/American black bear/ILSVRC2012_validation_preprocess_00014576.JPEG": 294,
 "/data/images/American black bear/ILSVRC2012_validation_preprocess_00000865.JPEG": 295,
 "/data/images/American black bear/ILSVRC2012_validation_preprocess_00007093.JPEG": 295,
 "/data/images/American black bear/ILSVRC2012_validation_preprocess_00014029.JPEG": 295
}
```

## [Export](#contents)

### Export process

Trained checkpoints may be exported to `ONNX`, `MINDIR` and `AIR` (currently not checked).

Usage

```shell
python export.py --config path/to/config.yaml --file_format FILE_FORMAT --pretrained path/to/checkpoint.ckpt --arch ARCHITECTURE_NAME
```

Example

```shell

# Export to MINDIR
python export.py --config src/configs/beit_base_patch16_224_ft1k.yaml --file_format MINDIR --pretrained /data/models/beit_base.ckpt --arch beit_base_patch16_224
```

### Export results

Exported models saved in the current directory with name the same as architecture.

# [Model Description](#contents)

## Performance

### Training Performance

| Parameters                 | GPU                                         |
|----------------------------|---------------------------------------------|
| Model Version              | BEiT_base_patch16_224                       |
| Resource                   | 4xGPU (NVIDIA GeForce RTX 3090)             |
| Uploaded Date              | 08/23/2023 (month/day/year)                 |
| MindSpore Version          | 1.9.0                                       |
| Dataset                    | ImageNet                                    |
| Training Parameters        | src/configs/beit_base_patch16_224_ft1k.yaml |
| Optimizer                  | AdamW                                       |
| Loss Function              | SoftmaxCrossEntropy                         |
| Outputs                    | logits                                      |
| Accuracy                   | ACC1 [81.9]                                 |
| Total time                 | 329 h                                       |
| Params                     | 86530984                                    |
| Checkpoint for Fine tuning | 996 M                                       |
| Scripts                    |                                             |

# [Description of Random Situation](#contents)

We use fixed seed in train.py.

# [ModelZoo Homepage](#contents)

Please check the official [homepage](https://gitee.com/mindspore/models).
