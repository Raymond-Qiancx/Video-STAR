import os
import json
import requests
import traceback
import argparse
from tqdm import tqdm
from PIL import Image
import base64
import cv2
import numpy as np

# HMDB51/UCF101/SSv2 视频数据集示例:
# python api_image.py \
#     --input-json /mnt/xmap_nas_alg/yzl/Recognition_Github/AFile/json/hmdb51/test.json \
#     --output-json /mnt/xmap_nas_alg/yzl/Recognition_Github/AFile/json/hmdb51/test_description.json \
#     --data-root /mnt/xmap_nas_alg/yzl/Recognition/AFile/datasets/hmdb51

# python api_image.py \
#     --input-json /mnt/xmap_nas_alg/yzl/Recognition_Github/AFile/json/hmdb51/test.json \
#     --data-root /mnt/xmap_nas_alg/yzl/Recognition/AFile/datasets/hmdb51

# API配置
DASHSCOPE_API_KEY = ''

def detect_json_format(annotations):
    """
    检测JSON数据的格式，返回字段名的映射
    支持列表格式: [{"file_name": "xxx.avi"}, ...]
    支持字典格式: {"data": [...], "metainfo": ...}
    """
    # 如果是字典格式，尝试找到数据列表
    if isinstance(annotations, dict):
        # 检查常见的字典格式
        if 'data' in annotations and isinstance(annotations['data'], list):
            annotations = annotations['data']
            print(f"检测到字典格式，已提取 'data' 字段下的 {len(annotations)} 条记录")
        elif 'annotations' in annotations and isinstance(annotations['annotations'], list):
            annotations = annotations['annotations']
            print(f"检测到字典格式，已提取 'annotations' 字段下的 {len(annotations)} 条记录")
        elif 'samples' in annotations and isinstance(annotations['samples'], list):
            annotations = annotations['samples']
            print(f"检测到字典格式，已提取 'samples' 字段下的 {len(annotations)} 条记录")
        elif 'images' in annotations and isinstance(annotations['images'], list):
            annotations = annotations['images']
            print(f"检测到字典格式，已提取 'images' 字段下的 {len(annotations)} 条记录")
        else:
            # 字典但没有标准字段，可能整个字典就是一条记录
            return {'file_name': 'path', 'is_single_item': True}

    if not annotations:
        return None

    first_item = annotations[0]

    # 可能的文件名/ID字段（按优先级排序）
    file_name_candidates = ['path', 'file_name', 'video_name', 'img_id', 'image_id', 'id', 'image_name', 'video']

    field_map = {}

    # 查找 file_name 字段
    for candidate in file_name_candidates:
        if candidate in first_item:
            field_map['file_name'] = candidate
            break
    else:
        # 如果都没找到，尝试使用第一个字符串类型的值
        for key, value in first_item.items():
            if isinstance(value, str) and ('.avi' in value or '.mp4' in value or '.webm' in value or '.jpg' in value or '.png' in value):
                field_map['file_name'] = key
                break

    return field_map


def extract_keyframes_from_video(video_path, num_frames=3):
    """
    从视频中提取关键帧
    使用均匀采样策略，从视频开头、中间、结尾提取帧
    """
    try:
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            return None

        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        if total_frames < num_frames:
            # 视频太短，提取所有帧
            frame_indices = list(range(total_frames))
        else:
            # 均匀采样
            frame_indices = [int(i * total_frames / num_frames) for i in range(num_frames)]

        frames = []
        for idx in frame_indices:
            cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
            ret, frame = cap.read()
            if ret:
                # 转换为RGB格式
                frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                frames.append(frame_rgb)

        cap.release()
        return frames if frames else None

    except Exception as e:
        print(f"Error extracting frames from video {video_path}: {e}")
        return None


def encode_frame_to_base64(frame):
    """将numpy数组帧编码为base64"""
    try:
        # 转换为PIL Image
        img = Image.fromarray(frame)
        # 调整大小以减小API调用开销（最大边不超过1024）
        max_size = 1024
        img.thumbnail((max_size, max_size), Image.LANCZOS)
        
        # 保存为JPEG格式
        import io
        buffer = io.BytesIO()
        img.save(buffer, format='JPEG', quality=85)
        image_data = buffer.getvalue()
        base64_str = base64.b64encode(image_data).decode('utf-8')
        return base64_str
    except Exception as e:
        print(f"Error encoding frame: {e}")
        return None


def analyze_with_qwen_multiframe(frames, is_video=True):
    """
    使用Qwen API分析视频关键帧
    生成详细的动作描述，包括身体部位变化和周围环境
    """
    try:
        if not frames:
            return None

        # 构建提示词 - 针对动作识别任务优化
        prompt = f"""Analyze this video clip and provide a detailed action description (within 3 sentences).

Focus on:
1. **Body Part Movements**: Describe how each body part (head, torso, arms, hands, legs, feet) changes position and pose across frames. Include direction, range, and speed of movement.
2. **Spatial-Temporal Changes**: Track how the person's body configuration evolves - e.g., from standing to bending, arms swinging, head turning, etc.
3. **Environment Context**: Describe the surrounding environment - indoor/outdoor setting, background objects, props, terrain, lighting conditions.
4. **Action Quality**: Note the manner of movement - smooth, rapid, controlled, graceful, jerky, etc.

Use precise English words. Keep the description concise but informative, within 3 sentences maximum.

Example format: "A person [specific body movements in sequence] while [environmental context], with [action quality/pace] motion."

Provide only the analysis description:"""

        # 构建多帧内容
        content = []

        # 添加每帧图像
        for i, frame in enumerate(frames):
            frame_base64 = encode_frame_to_base64(frame)
            if frame_base64:
                content.append({
                    "type": "image_url",
                    "image_url": {
                        "url": f"data:image/jpeg;base64,{frame_base64}"
                    }
                })

        if not content:
            return None

        # 添加提示词
        content.append({
            "type": "text",
            "text": prompt
        })

        # 调用API
        headers = {
            "Authorization": f"Bearer {DASHSCOPE_API_KEY}",
            "Content-Type": "application/json"
        }

        data = {
            "model": "qwen2.5-vl-72b-instruct",
            "messages": [
                {
                    "role": "user",
                    "content": content
                }
            ],
            "max_tokens": 300,
            "temperature": 0.3
        }

        response = requests.post(
            "https://dashscope.aliyuncs.com/compatible-mode/v1/chat/completions",
            headers=headers,
            json=data,
            timeout=120
        )

        if response.status_code == 200:
            result = response.json()
            analysis = result['choices'][0]['message']['content'].strip()
            return analysis
        else:
            print(f"API调用失败: {response.status_code}, {response.text}")
            return None

    except Exception as e:
        print(f"API调用出错: {e}")
        print(f"{traceback.format_exc()}")
        return None


def main():
    parser = argparse.ArgumentParser(description="Analyze video datasets using Qwen API")
    parser.add_argument('--input-json', type=str, required=True,
                       help='Path to the input JSON file')
    parser.add_argument('--output-json', type=str, default=None,
                       help='Path to the output JSON file for analysis results (auto-generated from input if not provided)')
    parser.add_argument('--data-root', type=str, required=True,
                       help='Root directory containing the videos')
    parser.add_argument('--max-items', type=int, default=-1,
                       help='Maximum number of items to process (-1 for all)')
    parser.add_argument('--start-index', type=int, default=0,
                       help='Start index for processing (for resuming)')
    parser.add_argument('--file-name-key', type=str, default=None,
                       help='Custom key for file name field (e.g., video_name, img_id)')
    parser.add_argument('--debug', action='store_true',
                       help='Enable debug mode to print sample data structure')
    parser.add_argument('--num-frames', type=int, default=3,
                       help='Number of keyframes to extract from each video (default: 3)')

    args = parser.parse_args()

    # 自动生成输出路径（如果未提供）
    if args.output_json is None:
        input_path = args.input_json
        # 去掉 .json 扩展名，添加 _description.json
        if input_path.endswith('.json'):
            args.output_json = input_path[:-5] + '_description.json'
        else:
            args.output_json = input_path + '_description'
        print(f"自动生成输出路径: {args.output_json}")

    # 读取输入的JSON文件
    print(f"正在读取输入文件: {args.input_json}")
    with open(args.input_json, 'r', encoding='utf-8') as f:
        raw_annotations = json.load(f)

    # 处理可能的字典格式
    annotations = raw_annotations
    is_single_item = False

    # 检测JSON格式
    if args.file_name_key:
        field_map = {
            'file_name': args.file_name_key
        }
        print(f"使用用户指定的字段名: {field_map}")
    else:
        field_map = detect_json_format(raw_annotations)

        # 如果返回 is_single_item，说明整个字典就是一条记录
        if field_map and field_map.get('is_single_item'):
            annotations = [raw_annotations]
            is_single_item = True
            del field_map['is_single_name']
            print("检测到单条记录格式")

    if not field_map or 'file_name' not in field_map:
        print("错误: 无法识别JSON数据格式！")
        print("JSON数据示例:")
        # 显示原始数据的前几条记录
        if isinstance(raw_annotations, dict):
            sample_keys = list(raw_annotations.keys())[:3]
            sample_data = {k: raw_annotations[k] for k in sample_keys}
            print(json.dumps(sample_data, indent=2, ensure_ascii=False))
        else:
            print(json.dumps(raw_annotations[0] if raw_annotations else {}, indent=2, ensure_ascii=False))
        print("\n请使用 --file-name-key 参数指定文件名字段名")
        return

    print(f"共发现 {len(annotations)} 个条目")

    if args.debug:
        print("\nJSON数据结构示例:")
        print(json.dumps(annotations[0], indent=2, ensure_ascii=False))
        print(f"\n检测到的字段映射: {field_map}")
        return

    # 加载结果文件（用于恢复）
    results = []
    processed_files = set()
    if os.path.exists(args.output_json):
        print(f"加载已有结果文件: {args.output_json}")
        with open(args.output_json, 'r', encoding='utf-8') as f:
            old_results = json.load(f)

        # 兼容旧格式（字典）和新格式（列表）
        if isinstance(old_results, dict):
            # 旧格式: {"video_name": {"analysis": ...}} -> 转换为新格式
            print("检测到旧格式结果文件，正在转换...")
            for path, item in old_results.items():
                results.append({
                    'path': path,
                    'problem_id': item.get('problem_id', len(results) + 1),
                    'analysis': item.get('analysis', '')
                })
        else:
            results = old_results

        # 记录已处理的文件
        for item in results:
            if 'path' in item:
                processed_files.add(item['path'])
        print(f"已加载 {len(results)} 条已有结果")

    # 获取字段名
    file_name_key = field_map.get('file_name', 'file_name')

    # 处理每个条目
    processed = 0
    skipped = 0
    failed = 0

    for idx, anno in enumerate(tqdm(annotations, desc="处理中")):
        current_file_name = anno.get(file_name_key)
        if current_file_name is None:
            print(f"警告: 第 {idx} 条数据缺少文件名字段，使用索引作为key")
            current_file_name = str(idx)

        # 跳过已处理的项目
        if current_file_name in processed_files:
            skipped += 1
            continue

        if idx < args.start_index:
            continue

        if args.max_items > 0 and processed >= args.max_items:
            break

        file_path = os.path.join(args.data_root, current_file_name)

        if not os.path.exists(file_path):
            if os.path.exists(current_file_name):
                file_path = current_file_name

        if not os.path.exists(file_path):
            print(f"文件不存在: {file_path}")
            failed += 1
            continue

        # 提取视频关键帧
        frames = extract_keyframes_from_video(file_path, args.num_frames)
        if frames is None:
            print(f"无法提取视频帧: {file_path}")
            failed += 1
            continue
        analysis = analyze_with_qwen_multiframe(frames, is_video=True)

        if analysis:
            results.append({
                'path': current_file_name,
                'problem_id': idx + 1,
                'analysis': analysis
            })
            processed_files.add(current_file_name)
            processed += 1

            if processed % 10 == 0:
                with open(args.output_json, 'w', encoding='utf-8') as f:
                    json.dump(results, f, ensure_ascii=False, indent=2)
                print(f"\n已保存中间结果（已处理 {processed} 个）")
        else:
            failed += 1
            print(f"分析失败: {current_file_name}")

    print(f"\n正在保存最终结果到: {args.output_json}")
    with open(args.output_json, 'w', encoding='utf-8') as f:
        json.dump(results, f, ensure_ascii=False, indent=2)

    print("\n完成！")
    print(f"  总共: {len(annotations)} 个")
    print(f"  已处理: {processed} 个")
    print(f"  已跳过: {skipped} 个（已有结果）")
    print(f"  失败: {failed} 个")
    print(f"  总计结果: {len(results)} 条")

    print("\n示例分析结果:")
    for item in results[:3]:
        print(f"\n  path: {item['path']}")
        print(f"  problem_id: {item['problem_id']}")
        print(f"  analysis: {item['analysis']}")

if __name__ == "__main__":
    main()
