import os
import json
import requests
import traceback
import argparse
from tqdm import tqdm

# API配置
DASHSCOPE_API_KEY = 'sk-23fab9350acd4ba5bf99e3b0ce1a23b8'

# python generate_action_descriptions.py --input-json /mnt/xmap_nas_alg/yzl/Recognition/FROSTER/zs_label_db/B2N_hmdb/test_rephrased.json --output-json /mnt/xmap_nas_alg/yzl/Recognition/FROSTER/zs_label_db/B2N_hmdb/test_explained_2.json
# python generate_action_descriptions.py \ 
# --input-json /mnt/xmap_nas_alg/yzl/Recognition/FROSTER/zs_label_db/B2N_hmdb/test_rephrased.json \
# --output-json /mnt/xmap_nas_alg/yzl/Recognition/FROSTER/zs_label_db/B2N_hmdb/test_explained_2.json

def call_qwen_api(action_name, action_description):
    """
    调用通义千问API为动作类型生成独特解释
    """
    try:
        # 构建prompt 4
#         prompt = f"""Please describe the action by explicitly detailing the movement changes for each body part in the following order: head, shoulders, arms, hands, torso, hips, legs, feet. Each body part description should be slightly enriched but limited to 10 words.

# Format:
# Head: [movement], Shoulders: [movement], Arms: [movement], Hands: [movement], Torso: [movement], Hips: [movement], Legs: [movement], Feet: [movement]. 

# Action: {action_name}
# Original Description: {action_description}

# Output explanation only:"""

        # 构建prompt 5
        prompt = f"""Please describe the action by focusing on the main body parts that undergo movement. Use the format:

[Body Part 1]: [movement], [Body Part 2]: [movement], ...

Each body part description should be slightly enriched but limited to 5 words.

All description should be limited to 15 words.

You can choose from the following body parts: Head, Shoulders, Arms, Hands, Torso, Hips, Legs, Feet

Example (for "kicking a ball"):
Leg: Swings forward forcefully, Foot: Kicks the ball with precision

Action: {action_name}
Original Description: {action_description}

Output explanation only:
"""

        
        # 调用API
        headers = {
            "Authorization": f"Bearer {DASHSCOPE_API_KEY}",
            "Content-Type": "application/json"
        }
        
        data = {
            "model": "qwen2.5-vl-7b-instruct",
            "messages": [
                {
                    "role": "user",
                    "content": prompt
                }
            ],
            "max_tokens": 200,
            "temperature": 0.7
        }
        
        response = requests.post(
            "https://dashscope.aliyuncs.com/compatible-mode/v1/chat/completions",
            headers=headers,
            json=data,
            timeout=30
        )
        
        if response.status_code == 200:
            result = response.json()
            enhanced_description = result['choices'][0]['message']['content'].strip()
            return enhanced_description
        else:
            print(f"API调用失败: {response.status_code}")
            return action_description
            
    except Exception as e:
        print(f"API调用出错: {e}")
        print(f"{traceback.format_exc()}")
        return action_description

def enhance_action_descriptions(actions):
    """
    为所有动作类型生成增强的描述
    """
    enhanced_actions = {}
    print("正在为动作类型生成增强描述...")
    
    for key, value in tqdm(actions.items(), desc="处理动作类型"):
        # 提取动作名称（冒号前的部分）
        if ':' in value:
            action_name = value.split(':')[0].strip()
            action_description = value.split(':', 1)[1].strip()
        else:
            action_name = value
            action_description = value
        
        # 调用API生成增强描述
        enhanced_description = call_qwen_api(action_name, action_description)
        
        # 构建新的描述格式 - 将增强描述添加到原描述后面
        enhanced_actions[key] = f"{action_name}: {action_description} {enhanced_description}"
        
        print(f"已处理: {action_name}")
    
    return enhanced_actions

def main():
    parser = argparse.ArgumentParser(description="Generate enhanced action descriptions using Qwen API")
    parser.add_argument('--input-json', type=str, required=True, help='Path to the input JSON file containing action types')
    parser.add_argument('--output-json', type=str, required=True, help='Path to the output JSON file for enhanced descriptions')
    
    args = parser.parse_args()
    
    # 读取输入的JSON文件
    print(f"正在读取输入文件: {args.input_json}")
    with open(args.input_json, 'r', encoding='utf-8') as f:
        actions = json.load(f)
    
    print(f"共发现 {len(actions)} 个动作类型")
    
    # 为动作类型生成增强描述
    enhanced_actions = enhance_action_descriptions(actions)
    
    # 保存增强描述到输出文件
    print(f"正在保存增强描述到: {args.output_json}")
    with open(args.output_json, 'w', encoding='utf-8') as f:
        json.dump(enhanced_actions, f, ensure_ascii=False, indent=4)
    
    print("完成！增强描述已保存到输出文件。")
    print("\n示例增强描述:")
    for i, (key, value) in enumerate(list(enhanced_actions.items())[:3]):
        print(f"  {key}: {value}")

if __name__ == "__main__":
    main()


# python generate_action_descriptions.py \
#   --input-json /mnt/xmap_nas_alg/yzl/Recognition/FROSTER/zs_label_db/B2N_hmdb/train_rephrased.json \
#   --output-json /mnt/xmap_nas_alg/yzl/Recognition/FROSTER/zs_label_db/B2N_hmdb/train_explained.json
