import json
import re
import sys

def extract_answer(xml_string):
    """
    从字符串中提取 <answer> 标签内的内容。
    如果找不到 <answer> 标签，则返回原始的整个字符串。
    """
    # 增加 re.DOTALL 标志以匹配包括换行符在内的任意字符
    match = re.search(r"<answer>(.*?)</answer>", xml_string, re.IGNORECASE | re.DOTALL)
    if match:
        # 提取匹配到的内容并去除首尾空白
        return match.group(1).strip()
    else:
        # 如果没有找到 <answer> 标签，则返回原始字符串并去除首尾空白
        return xml_string.strip()

def normalize(text):
    """
    标准化字符串：小写、去除空格、下划线和换行符
    """
    return text.lower().replace(" ", "").replace("_", "").replace("\n", "").replace("*", "")

def calculate_accuracy(json_file_path):
    with open(json_file_path, 'r') as f:
        data = json.load(f)

    correct = 0
    total = len(data)

    for item in data:
        solution = extract_answer(item.get("solution", ""))
        predict = extract_answer(item.get("predict", ""))

        sol_normalized = normalize(solution)
        pred_normalized = normalize(predict)

        if sol_normalized == pred_normalized:
            correct += 1

    accuracy = correct / total
    print(f"Correct count: {correct}")
    print(f"Total count: {total}")
    print(f"Accuracy: {accuracy:.4f}")
    return accuracy

if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: python script.py <json_file_path>")
        sys.exit(1)

    json_file_path = sys.argv[1]
    calculate_accuracy(json_file_path)

# python final.py Qwen2.5-VL-7B-Instruct.json

