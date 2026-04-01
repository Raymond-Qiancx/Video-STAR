from ultralytics import YOLO
import cv2
import os

# 加载模型
model = YOLO("/mnt/xmap_nas_alg/yzl/Recognition/AScripts/pose/yolo11x-pose.pt")

# 输入输出目录
input_root = "/mnt/xmap_nas_alg/yzl/Recognition/mmaction2/data/SSv2/SSv2/20bn-something-something-v2"
output_root = "/mnt/xmap_nas_alg/yzl/Recognition/mmaction2/data/SSv2/po"

# 检查输入目录是否存在
if not os.path.isdir(input_root):
    raise ValueError(f"输入目录不存在: {input_root}")

# 创建输出目录
os.makedirs(output_root, exist_ok=True)

print("开始处理所有视频文件...")

# 遍历输入目录下的所有视频文件
for filename in os.listdir(input_root):
    # 支持多种视频格式
    if not filename.lower().endswith(('.mp4', '.avi', '.mkv', '.webm')):
        continue
        
    # 构造完整路径
    video_path = os.path.join(input_root, filename)
    output_path = os.path.join(output_root, filename)
    
    # 处理视频
    try:
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            print(f"无法打开视频: {video_path}")
            continue
            
        fps = int(cap.get(cv2.CAP_PROP_FPS))
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        
        # 动态获取原始视频的编码格式
        fourcc = int(cap.get(cv2.CAP_PROP_FOURCC))
        # 如果无法获取有效编码格式，则使用 MJPG 作为默认
        if fourcc == 0:
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

print("所有视频处理完成")
