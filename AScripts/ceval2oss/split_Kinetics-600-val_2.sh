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


# kinetics400数据集
echo "开始处理 kinetics400 数据集..."

CUDA_VISIBLE_DEVICES=2 python "${PYTHON_SCRIPT}" \
  --prompt-path /mnt/xmap_nas_alg/yzl/Recognition/AFile/json_split/Kinetics-600-val/k600_split_2.json \
  --model-path /mnt/xmap_nas_alg/yzl/Recognition/AFile/models/${MODEL_NAME} \
  --json-file-path /mnt/xmap_nas_alg/yzl/Recognition/AFile/json_split/Kinetics-600-val/k600_split2_rephrased_classes.json \
  --original-video-path /mnt/xmap_nas_alg/yzl/Recognition/mmaction2/data/Kinetics-600-val/share/common/VideoDatasets/Kinetics-600/videos/val_256/ \
  --annotated-video-path /mnt/xmap_nas_alg/yzl/Recognition/mmaction2/data/Kinetics-600-val/share/common/VideoDatasets/Kinetics-600/videos/val_256/ \
  --output-path ${OUTPUT_DIR}/k600_split_2.json

echo "kinetics400 数据集处理完成！"