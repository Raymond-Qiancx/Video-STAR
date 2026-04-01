import json
import re
import sys

def extract_answer(xml_string):
    """
    从 XML 格式的字符串中提取 <answer> 标签内的内容
    """
    match = re.search(r"<answer>(.*?)</answer>", xml_string, re.IGNORECASE)
    if match:
        return match.group(1)
    else:
        return ""

def normalize(text):
    """
    标准化字符串：小写、去除空格、下划线和换行符
    """
    return text.lower().replace(" ", "").replace("_", "").replace("\n", "")

def calculate_accuracy(json_file_path):
    with open(json_file_path, 'r') as f:
        data = json.load(f)

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
