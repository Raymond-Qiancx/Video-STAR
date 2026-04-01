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
OUTPUT_DIR="/data/oss_bucket_0/yzl/Recognition/AScripts/ceval3oss/output_tab3/${MODEL_NAME}"
mkdir -p "${OUTPUT_DIR}"
echo "输出目录已创建: ${OUTPUT_DIR}"


# kinetics400数据集
echo "开始处理 kinetics400 数据集..."

CUDA_VISIBLE_DEVICES=4 python "${PYTHON_SCRIPT}" \
  --prompt-path /data/oss_bucket_0/yzl/Recognition/AFile/json_split/Kinetics-600-val/k600_split_3.json \
  --model-path /data/oss_bucket_0/yzl/Recognition/AFile/models/${MODEL_NAME} \
  --json-file-path /data/oss_bucket_0/yzl/Recognition/AFile/json_split/Kinetics-600-val/k600_split3_rephrased_classes.json \
  --original-video-path /data/oss_bucket_0/yzl/Recognition/mmaction2/data/Kinetics-600-val/share/common/VideoDatasets/Kinetics-600/videos/val_256/ \
  --annotated-video-path /data/oss_bucket_0/yzl/Recognition/mmaction2/data/Kinetics-600-val/share/common/VideoDatasets/Kinetics-600/videos/val_256/ \
  --output-path ${OUTPUT_DIR}/k600_split_3.json

echo "kinetics400 数据集处理完成！"