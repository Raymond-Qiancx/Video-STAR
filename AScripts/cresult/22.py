import json
import re
import sys
import os
import glob
import argparse

def extract_answer(xml_string):
    if not isinstance(xml_string, str):
        return ""
    match = re.search(r"<answer>(.*?)</answer>", xml_string, re.IGNORECASE | re.DOTALL)
    if match:
        return match.group(1)
    return ""

def normalize(text):
    if not isinstance(text, str):
        return ""
    return text.lower().replace(" ", "").replace("_", "").replace("\n", "").replace("\r", "")

def calculate_accuracy_for_file(json_file_path):
    try:
        with open(json_file_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
    except Exception as e:
        print(f"Failed to load JSON file {json_file_path}: {e}")
        return None

    correct = 0
    total = 0

    for item in data:
        solution = extract_answer(item.get("solution", ""))
        predict = extract_answer(item.get("predict", ""))

        sol_normalized = normalize(solution)
        pred_normalized = normalize(predict)

        # 如果 predict 是 error，则跳过该条目
        if pred_normalized == "error":
            continue

        # 仅对非 error 的条目进行统计
        total += 1
        if sol_normalized == pred_normalized:
            correct += 1

    # 防止除以零错误
    accuracy = correct / total if total > 0 else 0.0

    # 输出顺序：Accuracy -> Correct count -> Total count -> Filename（只保留最后的文件名）
    print(f"Filename: {os.path.basename(json_file_path)}")
    print(f"Accuracy: {accuracy:.4f}    Correct count: {correct}    Total count: {total}")
    print()  # 空行分隔不同文件的输出

    return accuracy

def find_json_files(path):
    if os.path.isfile(path) and path.lower().endswith('.json'):
        return [path]
    elif os.path.isdir(path):
        files = glob.glob(os.path.join(path, '*.json'))
        if files:
            return sorted(files)
        files = glob.glob(os.path.join(path, '**', '*.json'), recursive=True)
        return sorted(files)
    else:
        return []

def main():
    parser = argparse.ArgumentParser(description="计算目录下所有 JSON 文件的准确率（基于 <answer> 标签）。")
    parser.add_argument("path", help="JSON 文件或包含 JSON 文件的目录路径")
    args = parser.parse_args()

    path = args.path
    json_files = find_json_files(path)
    if not json_files:
        print(f"No JSON files found at: {path}")
        sys.exit(1)

    accuracies = []
    for jf in json_files:
        acc = calculate_accuracy_for_file(jf)
        if acc is None:
            # 出错文件以 0 代替，或根据需要跳过；这里我们用 0 来占位
            accuracies.append(0.0)
        else:
            accuracies.append(acc)

    # 最后一行：所有 accuracy * 100，数字之间用空格分隔，保留两位小数，仅数字
    accuracies_pct = [f"{(a * 100):.1f}" for a in accuracies]
    print(" ".join(accuracies_pct))

if __name__ == "__main__":
    main()
