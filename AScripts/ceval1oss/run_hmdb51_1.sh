#!/bin/bash

# 检查参数数量
if [ "$#" -lt 2 ]; then
    echo "用法: $0 <python脚本路径> <模型名称>"
    exit 1
fi

# 获取传入的参数
PYTHON_SCRIPT=$1
MODEL_NAME=$2

# 创建输出目录
OUTPUT_DIR="/mnt/xmap_nas_alg/yzl/Recognition/AScripts/ceval1oss/output_tab1/${MODEL_NAME}"
mkdir -p "${OUTPUT_DIR}"
echo "输出目录已创建: ${OUTPUT_DIR}"

# hmdb51数据集处理
echo "开始处理 hmdb51 数据集..."

CUDA_VISIBLE_DEVICES=6  python "${PYTHON_SCRIPT}" \
  --prompt-path /mnt/xmap_nas_alg/yzl/Recognition/AFile/json/hmdb51/test.json \
  --model-path /mnt/xmap_nas_alg/yzl/Recognition/AFile/models/${MODEL_NAME} \
  --json-file-path /mnt/xmap_nas_alg/yzl/Recognition/AFile/json/hmdb51/test_rephrased.json \
  --original-video-path /mnt/xmap_nas_alg/yzl/Recognition/AFile/datasets/hmdb51/ \
  --annotated-video-path /mnt/xmap_nas_alg/yzl/Recognition/AFile/datasets/hmdb51_pose/ \
  --output-path ${OUTPUT_DIR}/hmdb51_test.json \

echo "hmdb51 数据集处理完成！"
