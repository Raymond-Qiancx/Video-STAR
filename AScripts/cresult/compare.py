import json
import re

def normalize(text):
    """
    标准化字符串：小写、去除空格、下划线和换行符
    """
    if text is None:
        return ""
    return text.lower().replace(" ", "").replace("_", "").replace("\n", "")

def extract_answer(text):
    """从文本中提取<answer>标签内的内容"""
    match = re.search(r'<answer>(.*?)</answer>', text)
    return match.group(1) if match else None

def analyze_json_files():
    """分析两个JSON文件，找出predict不一致但其中一个与solution匹配的情况"""
    
    # 读取第一个JSON文件
    with open('/mnt/xmap_nas_alg/yzl/Recognition/AScripts/ceval3oss/output_tab1/grpo_3B_4subtol/checkpoint-580/hmdb51_val.json', 'r', encoding='utf-8') as f:
        data1 = json.load(f)
    
    # 读取第二个JSON文件
    with open('/mnt/xmap_nas_alg/yzl/Recognition/AScripts/ceval1/coutput/Qwen2.5-VL-3B-Instruct/hmdb51_val.json', 'r', encoding='utf-8') as f:
        data2 = json.load(f)
    
    # 创建字典以便快速查找
    data1_dict = {item['problem_id']: item for item in data1}
    data2_dict = {item['problem_id']: item for item in data2}
    
    # 存储结果
    first_correct_items = []  # 第一个文件对，第二个文件错
    second_correct_items = []  # 第一个文件错，第二个文件对
    
    # 遍历所有problem_id
    all_ids = set(data1_dict.keys()) | set(data2_dict.keys())
    
    for problem_id in sorted(all_ids):
        if problem_id in data1_dict and problem_id in data2_dict:
            item1 = data1_dict[problem_id]
            item2 = data2_dict[problem_id]
            
            # 提取solution和predict
            solution1 = extract_answer(item1.get('solution', ''))
            solution2 = extract_answer(item2.get('solution', ''))
            predict1 = extract_answer(item1.get('predict', ''))
            predict2 = extract_answer(item2.get('predict', ''))
            
            # 检查solution是否一致
            if solution1 == solution2:
                # 使用标准化字符串进行比较
                normalized_solution = normalize(solution1)
                normalized_predict1 = normalize(predict1)
                normalized_predict2 = normalize(predict2)
                
                # 检查标准化后的predict是否不一致
                if normalized_predict1 != normalized_predict2:
                    # 检查其中一个predict是否与solution匹配（使用标准化比较）
                    if normalized_predict1 == normalized_solution and normalized_predict2 != normalized_solution:
                        # 第一个文件对，第二个文件错
                        first_correct_items.append({
                            'problem_id': problem_id,
                            'solution': solution1,
                            'predict1': predict1,
                            'predict2': predict2,
                            'normalized_solution': normalized_solution,
                            'normalized_predict1': normalized_predict1,
                            'normalized_predict2': normalized_predict2,
                            'item1': item1,
                            'item2': item2
                        })
                    elif normalized_predict2 == normalized_solution and normalized_predict1 != normalized_solution:
                        # 第一个文件错，第二个文件对
                        second_correct_items.append({
                            'problem_id': problem_id,
                            'solution': solution1,
                            'predict1': predict1,
                            'predict2': predict2,
                            'normalized_solution': normalized_solution,
                            'normalized_predict1': normalized_predict1,
                            'normalized_predict2': normalized_predict2,
                            'item1': item1,
                            'item2': item2
                        })
    
    return first_correct_items, second_correct_items

def save_to_files(first_correct_items, second_correct_items):
    """将结果保存到两个文件中"""
    
    # 保存第一个文件对而第二个错的内容
    with open('first_correct_second_wrong.json', 'w', encoding='utf-8') as f:
        json.dump(first_correct_items, f, indent=2, ensure_ascii=False)
    
    # 保存第一个文件错而第二个对的内容
    with open('first_wrong_second_correct.json', 'w', encoding='utf-8') as f:
        json.dump(second_correct_items, f, indent=2, ensure_ascii=False)
    
    print(f"已保存到两个文件：")
    print(f"1. first_correct_second_wrong.json (第一个对第二个错): {len(first_correct_items)} 个项目")
    print(f"2. first_wrong_second_correct.json (第一个错第二个对): {len(second_correct_items)} 个项目")

def main():
    """主函数"""
    print("正在分析JSON文件...")
    print("使用标准化字符串比较（小写、去除空格和下划线）")
    
    try:
        first_correct_items, second_correct_items = analyze_json_files()
        
        total_items = len(first_correct_items) + len(second_correct_items)
        
        if total_items == 0:
            print("没有找到符合条件的项目。")
            return
        
        print(f"\n找到 {total_items} 个符合条件的项目：")
        print(f"- 第一个文件对而第二个错: {len(first_correct_items)} 个")
        print(f"- 第一个文件错而第二个对: {len(second_correct_items)} 个")
        
        # 保存到文件
        save_to_files(first_correct_items, second_correct_items)
        
        # 显示详细结果
        print("\n" + "=" * 80)
        
        if first_correct_items:
            print(f"\n第一个文件对而第二个错的项目 (共{len(first_correct_items)}个):")
            for i, result in enumerate(first_correct_items, 1):
                print(f"\n项目 {i}:")
                print(f"Problem ID: {result['problem_id']}")
                print(f"Solution: {result['solution']}")
                print(f"第一个文件predict: {result['predict1']} ✅")
                print(f"第二个文件predict: {result['predict2']} ❌")
                print(f"标准化比较: {result['normalized_solution']} vs {result['normalized_predict1']} vs {result['normalized_predict2']}")
        
        if second_correct_items:
            print(f"\n第一个文件错而第二个对的项目 (共{len(second_correct_items)}个):")
            for i, result in enumerate(second_correct_items, 1):
                print(f"\n项目 {i}:")
                print(f"Problem ID: {result['problem_id']}")
                print(f"Solution: {result['solution']}")
                print(f"第一个文件predict: {result['predict1']} ❌")
                print(f"第二个文件predict: {result['predict2']} ✅")
                print(f"标准化比较: {result['normalized_solution']} vs {result['normalized_predict1']} vs {result['normalized_predict2']}")
    
    except FileNotFoundError as e:
        print(f"文件未找到: {e}")
    except json.JSONDecodeError as e:
        print(f"JSON解析错误: {e}")
    except Exception as e:
        print(f"发生错误: {e}")

if __name__ == "__main__":
    main()
