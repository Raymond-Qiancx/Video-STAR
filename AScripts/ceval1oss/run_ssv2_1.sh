#!/bin/bash

# 检查参数数量
# if [ "$#" -lt 2 ]; then
#     echo "用法: $0 <python脚本路径> <模型名称>"
#     exit 1
# fi

# 获取传入的参数
PYTHON_SCRIPT=$1
MODEL_NAME=$2
# PYTHON_SCRIPT="4subtol.py"
# # MODEL_NAME="grpo_3B_ssv2"
# MODEL_NAME="Qwen2.5-VL-3B-Instruct"

# 创建输出目录
OUTPUT_DIR="/mnt/xmap_nas_alg/yzl/Recognition/AScripts/ceval1oss/output_tab1/${MODEL_NAME}"
mkdir -p "${OUTPUT_DIR}"
echo "输出目录已创建: ${OUTPUT_DIR}"

# SSv2数据集
echo "开始处理 SSv2 数据集..."

CUDA_VISIBLE_DEVICES=4 python "${PYTHON_SCRIPT}" \
  --prompt-path /mnt/xmap_nas_alg/yzl/Recognition/AFile/json/SSv2/test.json \
  --model-path /mnt/xmap_nas_alg/yzl/Recognition/AFile/models/${MODEL_NAME} \
  --json-file-path /mnt/xmap_nas_alg/yzl/Recognition/AFile/json/SSv2/test_rephrased.json \
  --original-video-path /mnt/xmap_nas_alg/yzl/Recognition/mmaction2/data/SSv2/SSv2/20bn-something-something-v2/ \
  --annotated-video-path /mnt/xmap_nas_alg/yzl/Recognition/mmaction2/data/SSv2/SSv2/20bn-something-something-v2/ \
  --output-path ${OUTPUT_DIR}/SSv2_test_2.json

echo "SSv2 数据集处理完成！"