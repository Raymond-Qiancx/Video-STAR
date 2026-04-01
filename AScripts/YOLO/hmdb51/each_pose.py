from ultralytics import YOLO
import cv2
import os
import argparse

# 加载模型
model = YOLO("yolo11x-pose.pt")

# 数据集配置
# input_root = "/mnt/xmap_nas_alg/yzl/Recognition/mmaction2/data/hmdb51/videos"
# output_root = "/mnt/xmap_nas_alg/yzl/Recognition/mmaction2/data/hmdb51/tool-videos"

input_root = "/mnt/xmap_nas_alg/yzl/Recognition/AFile/datasets/hmdb51"
output_root = "/mnt/xmap_nas_alg/yzl/Recognition/AFile/datasets/hmdb51_pose"

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
    if not filename.endswith('.avi'):
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

# eval "$(/mnt/xmap_nas_alg/yzl/micromamba shell hook -s posix)"
# export MAMBA_ROOT_PREFIX=/mnt/xmap_nas_alg/yzl/bin
# micromamba activate action-r3

# brush_hair  clap          draw_sword  fall_floor  handstand  kick       pick    push        run          shoot_gun  smoke           sword           turn
# cartwheel   climb         dribble     fencing     hit        kick_ball  pour    pushup      shake_hands  sit        somersault      sword_exercise  walk
# catch       climb_stairs  drink       flic_flac   hug        kiss       pullup  ride_bike   shoot_ball   situp      stand           talk            wave
# chew        dive          eat         golf        jump       laugh      punch   ride_horse  shoot_bow    smile      swing_baseball  throw

# python each_pose.py --category brush_hair
# python each_pose.py --category clap
# python each_pose.py --category draw_sword
# python each_pose.py --category fall_floor
# python each_pose.py --category handstand

# python each_pose.py --category kick
# python each_pose.py --category pick
# python each_pose.py --category push
# python each_pose.py --category run
# python each_pose.py --category shoot_gun

# python each_pose.py --category smoke
# python each_pose.py --category sword
# python each_pose.py --category turn
# python each_pose.py --category cartwheel
# python each_pose.py --category climb

# python each_pose.py --category dribble
# python each_pose.py --category fencing
# python each_pose.py --category hit
# python each_pose.py --category kick_ball
# python each_pose.py --category pour

# python each_pose.py --category pushup
# python each_pose.py --category shake_hands
# python each_pose.py --category sit
# python each_pose.py --category somersault
# python each_pose.py --category sword_exercise

# python each_pose.py --category walk
# python each_pose.py --category catch
# python each_pose.py --category climb_stairs
# python each_pose.py --category drink
# python each_pose.py --category flic_flac

# python each_pose.py --category hug
# python each_pose.py --category kiss
# python each_pose.py --category pullup
# python each_pose.py --category ride_bike
# python each_pose.py --category shoot_ball

# python each_pose.py --category situp
# python each_pose.py --category stand
# python each_pose.py --category talk
# python each_pose.py --category chew
# python each_pose.py --category dive

# python each_pose.py --category eat
# python each_pose.py --category golf
# python each_pose.py --category jump
# python each_pose.py --category laugh
# python each_pose.py --category punch

# python each_pose.py --category ride_horse
# python each_pose.py --category shoot_bow
# python each_pose.py --category smile
# python each_pose.py --category swing_baseball
# python each_pose.py --category throw