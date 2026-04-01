from ultralytics import YOLO
import cv2
import os
import argparse

# 加载模型
model = YOLO("/mnt/xmap_nas_alg/yzl/Recognition/AScripts/pose/yolo11x-pose.pt")

input_root = "/mnt/xmap_nas_alg/yzl/Recognition/mmaction2/data/Kinetics-600-val/share/common/VideoDatasets/Kinetics-600/videos/val_256"
output_root = "/mnt/xmap_nas_alg/yzl/Recognition/mmaction2/data/Kinetics-600-val/share/common/VideoDatasets/Kinetics-600/videos/po"

# 命令行参数解析
parser = argparse.ArgumentParser(description='处理HMDB51数据集中的指定类别')
parser.add_argument('--category', required=True, 
                    help='要处理的类别名称（示例：brush_hair, clap, run 等）')
args = parser.parse_args()

# 获取指定类别
category_name = args.category

# 构造实际存在的目录路径（注意双category_name）
input_category_dir = os.path.join(input_root, category_name)
if not os.path.isdir(input_category_dir):
    raise ValueError(f"指定的类别目录不存在: {input_category_dir}")

# 创建对应的输出目录（保持相同结构）
output_category_dir = os.path.join(output_root, category_name)
os.makedirs(output_category_dir, exist_ok=True)

# 处理该类别下的所有视频
print(f"开始处理类别: {category_name}")
for filename in os.listdir(input_category_dir):
    if not filename.endswith('.mp4'):
        continue
        
    # 构造完整路径
    video_path = os.path.join(input_category_dir, filename)
    output_path = os.path.join(output_category_dir, filename)
    
    # 处理视频
    try:
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            print(f"无法打开视频: {video_path}")
            continue
            
        fps = int(cap.get(cv2.CAP_PROP_FPS))
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        
        fourcc = cv2.VideoWriter_fourcc(*'MJPG')
        out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
        
        results = model(video_path, stream=True)
        
        for result in results:
            annotated_frame = result.plot(
                boxes=False,         # 关闭边界框
                labels=False, 
                conf=False,
                kpt_radius=3,         # 关键点圆圈半径
                line_width=3          # 连接线宽度
            )
            out.write(annotated_frame)
        
        cap.release()
        out.release()
        print(f"成功处理: {video_path} -> {output_path}")
        
    except Exception as e:
        print(f"处理视频时出错: {video_path}")
        print(f"错误详情: {str(e)}")
        continue

print(f"类别 {category_name} 处理完成")