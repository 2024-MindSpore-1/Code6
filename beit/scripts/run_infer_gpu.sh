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
OUTPUT_JSON_FILE="predictions.json"

display_usage() {
    echo -e "Usage: $0 DATA [--checkpoint CHECKPOINT] [--arch ARCHITECTURE] [--output OUTPUT_JSON_FILE (default: $OUTPUT_JSON_FILE)]"
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
# If yes, parse them
if [[ $# -lt 1 ]]; then
    echo "Not enough arguments"
    exit 1
else
    DATA="$1"
    shift 1
fi



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
        --arch)
            if [[ "$#" -lt 2 ]]; then
                echo "No ARCHITECTURE option"
                exit 1
            else
                ARCHITECTURE="$2";
                shift
            fi ;;
        --output)
            if [[ "$#" -lt 2 ]]; then
                echo "No OUTPUT_JSON_FILE option"
                exit 1
            else
                OUTPUT_JSON_FILE="$2";
                shift
            fi ;;
        *) echo "Unknown option: '$1'"; exit 1 ;;
    esac
    shift
done

INFER_SCRIPT_DIR="$SCRIPT_DIR/.."

echo "Start inference for device $DEVICE_ID"
if [ -z "$CHECKPOINT" ]; then
    echo "Error! Expected --checkpoint option is not empty. "
    exit 1
fi

python3 "$INFER_SCRIPT_DIR/infer.py" "$DATA" --checkpoint "$CHECKPOINT" \
  --arch "$ARCHITECTURE" --output "$OUTPUT_JSON_FILE"
