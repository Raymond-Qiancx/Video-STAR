import subprocess
import concurrent.futures
import os
import logging

# 配置日志
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# 指定路径
DATA_DIR = "/mnt/xmap_nas_alg/yzl/Recognition/mmaction2/data/Kinetics-600-val/share/common/VideoDatasets/Kinetics-600/videos/val_256"

# 获取所有子目录名称（作为 category）
def get_categories_from_path(data_dir):
    if not os.path.exists(data_dir):
        logging.error(f"路径不存在: {data_dir}")
        return []
    try:
        entries = os.listdir(data_dir)
        categories = [entry for entry in entries if os.path.isdir(os.path.join(data_dir, entry))]
        return categories
    except Exception as e:
        logging.error(f"读取路径失败: {e}")
        return []

# 每个 category 的执行命令
def run_category(category):
    cmd = ["python", "each_pose.py", "--category", category]
    try:
        logging.info(f"开始执行: {' '.join(cmd)}")
        result = subprocess.run(cmd, check=True, text=True, capture_output=True)
        logging.info(f"成功完成: {category}")
    except subprocess.CalledProcessError as e:
        logging.error(f"执行失败: {category}, 错误信息: {e.stderr}")

# 设置最大线程数（根据你的 CPU 核心数调整）
MAX_THREADS = 8  # 可调整为 4~8

# 主函数
def main():
    categories = get_categories_from_path(DATA_DIR)
    if not categories:
        logging.warning("未找到任何 category，任务已跳过。")
        return

    with concurrent.futures.ThreadPoolExecutor(max_workers=MAX_THREADS) as executor:
        futures = [executor.submit(run_category, category) for category in categories]
        concurrent.futures.wait(futures)

if __name__ == "__main__":
    main()
