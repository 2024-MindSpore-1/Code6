#!/bin/bash
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
SCRIPT_DIR=$(cd "$(dirname ${BASH_SOURCE[0]})" && pwd)
echo "SCRIPT_DIR=$SCRIPT_DIR"


display_usage() {
    echo -e "Usage: $0 CONFIG [--num_devices NUM_DEVICES] [--device_ids DEVICE_IDS (e.g. '0,1,2,3')] [--checkpoint CHECKPOINT] [--extra *EXTRA_ARGS]"
    echo "Important! Extra args must be the last argument."
}

# Check if help in CLI arguments
for arg in "$@"
do
  if [[ "$arg" == "--help" || "$arg" == "-h" ]]; then
    display_usage
    exit 0
  fi
done

# Check if there are enough arguments
# If yes, parse the first three
if [[ $# -lt 2 ]]; then
    echo "Not enough arguments"
    exit 1
else
    CONFIG_PATH="$1"
    shift 1
fi

DEVICE_IDS="0"
NUM_DEVICES=1
EXTRA=""

# Parse remain arguments
while [[ "$#" -gt 0 ]]; do
    case $1 in
        --checkpoint)
            if [[ "$#" -lt 2 ]]; then
                echo "No CHECKPOINT option"
                exit 1
            else
                CHECKPOINT="$2";
                shift
            fi ;;
        --num_devices)
            if [[ "$#" -lt 2 ]]; then
                echo "No NUM_DEVICES option"
                exit 1
            else
                NUM_DEVICES="$2";
                shift
            fi ;;
        --device_ids)
            if [[ "$#" -lt 2 ]]; then
                echo "No DEVICE_IDS option"
                exit 1
            else
                DEVICE_IDS="$2";
                shift
            fi ;;
        --extra)
            if [[ "$#" -lt 2 ]]; then
                echo "No EXTRA option"
                exit 1
            else
                shift;
                EXTRA="$*";
                break
            fi ;;
        *) echo "Unknown option: '$1'"; exit 1 ;;
    esac
    shift
done

TRAIN_SCRIPT_DIR="$SCRIPT_DIR/.."
export CUDA_VISIBLE_DEVICES=$DEVICE_IDS

# If variable CHECKPOINT is empty then run training from scratch.
# Otherwise, continue training from the checkpoint.
echo "Start training on devices $DEVICE_IDS"
if [ -z "$CHECKPOINT" ]; then
    mpirun -n $NUM_DEVICES --allow-run-as-root \
      python "$TRAIN_SCRIPT_DIR/train.py" --config "$CONFIG_PATH" \
      --device_num $NUM_DEVICES --device_target GPU \
      $EXTRA 2>&1 | tee distributed_training.log
else
    mpirun -n $NUM_DEVICES --allow-run-as-root \
      python "$TRAIN_SCRIPT_DIR/train.py" --config "$CONFIG_PATH" \
      --device_num $NUM_DEVICES --device_target GPU --pretrained "$CHECKPOINT" \
      $EXTRA  2>&1 | tee distributed_training.log
fi
