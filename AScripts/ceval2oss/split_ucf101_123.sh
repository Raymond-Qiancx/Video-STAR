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
OUTPUT_DIR="/mnt/xmap_nas_alg/yzl/Recognition/AScripts/ceval3oss/output_tab2/${MODEL_NAME}"
mkdir -p "${OUTPUT_DIR}"
echo "输出目录已创建: ${OUTPUT_DIR}"


# ucf101数据集
echo "开始处理 ucf101 数据集..."

CUDA_VISIBLE_DEVICES=4 python "${PYTHON_SCRIPT}" \
  --prompt-path /mnt/xmap_nas_alg/yzl/Recognition/AFile/json_split/ucf101/ucf_split_1.json \
  --model-path /mnt/xmap_nas_alg/yzl/Recognition/AFile/models/${MODEL_NAME} \
  --json-file-path /mnt/xmap_nas_alg/yzl/Recognition/AFile/json_split/ucf101/ucf101_rephrased_classes.json \
  --original-video-path /mnt/xmap_nas_alg/yzl/Recognition/mmaction2/data/ucf101/UCF-101/ \
  --annotated-video-path /mnt/xmap_nas_alg/yzl/Recognition/mmaction2/data/ucf101/UCF-101/ \
  --output-path ${OUTPUT_DIR}/ucf_split_1.json

CUDA_VISIBLE_DEVICES=4 python "${PYTHON_SCRIPT}" \
  --prompt-path /mnt/xmap_nas_alg/yzl/Recognition/AFile/json_split/ucf101/ucf_split_2.json \
  --model-path /mnt/xmap_nas_alg/yzl/Recognition/AFile/models/${MODEL_NAME} \
  --json-file-path /mnt/xmap_nas_alg/yzl/Recognition/AFile/json_split/ucf101/ucf101_rephrased_classes.json \
  --original-video-path /mnt/xmap_nas_alg/yzl/Recognition/mmaction2/data/ucf101/UCF-101/ \
  --annotated-video-path /mnt/xmap_nas_alg/yzl/Recognition/mmaction2/data/ucf101/UCF-101/ \
  --output-path ${OUTPUT_DIR}/ucf_split_2.json

CUDA_VISIBLE_DEVICES=4 python "${PYTHON_SCRIPT}" \
  --prompt-path /mnt/xmap_nas_alg/yzl/Recognition/AFile/json_split/ucf101/ucf_split_3.json \
  --model-path /mnt/xmap_nas_alg/yzl/Recognition/AFile/models/${MODEL_NAME} \
  --json-file-path /mnt/xmap_nas_alg/yzl/Recognition/AFile/json_split/ucf101/ucf101_rephrased_classes.json \
  --original-video-path /mnt/xmap_nas_alg/yzl/Recognition/mmaction2/data/ucf101/UCF-101/ \
  --annotated-video-path /mnt/xmap_nas_alg/yzl/Recognition/mmaction2/data/ucf101/UCF-101/ \
  --output-path ${OUTPUT_DIR}/ucf_split_3.json

echo "ucf101 数据集处理完成！"