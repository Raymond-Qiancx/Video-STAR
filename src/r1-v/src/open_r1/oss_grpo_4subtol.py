# Copyright 2025 The HuggingFace Team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import os
import json
import re
from datetime import datetime
from dataclasses import dataclass, field
from typing import Optional

from datasets import load_dataset, load_from_disk
from transformers import Qwen2VLForConditionalGeneration

from trainer import Qwen2VLGRPOTrainer, Qwen2VLGRPOVLLMTrainerModified
from trl import GRPOConfig, GRPOTrainer, ModelConfig, ScriptArguments, TrlParser, get_peft_config

from datasets import Dataset, DatasetDict

from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction
from rouge_score import rouge_scorer

def get_allowed_answers_str(actions, use_noun_explanation=True):
    """构建允许答案的字符串（根据是否需要名词解释决定格式）"""
    if use_noun_explanation:
        # 构建允许答案的字符串（格式：Brush hair: ... \n Cartwheel: ...）
        allowed_answers = [value for key, value in actions.items()]
        return "\n".join(allowed_answers)
    else:
        # 只提取动作类型，不包含解释
        allowed_answers = []
        for key, value in actions.items():
            # 提取冒号前的动作类型
            action_type = value.split(':')[0].strip()
            allowed_answers.append(action_type)
        return "\n".join(allowed_answers)

def get_video_path(x_path, pose_estimation, person_detection, original_video_path="/mnt/xmap_nas_alg/yzl/Recognition/AFile/datasets/hmdb51"):
    """根据工具决策选择对应的视频路径"""
    
    if pose_estimation == 'yes' and person_detection == 'no':
        # 只有姿态估计
        base_path = "/mnt/xmap_nas_alg/yzl/Recognition/AFile/datasets/hmdb51_po"
        return os.path.join(base_path, x_path)
    elif pose_estimation == 'yes' and person_detection == 'yes':
        # 姿态估计 + 人员检测
        base_path = "/mnt/xmap_nas_alg/yzl/Recognition/AFile/datasets/hmdb51_pose"
        return os.path.join(base_path, x_path)
    elif pose_estimation == 'no' and person_detection == 'yes':
        # 只有人员检测
        base_path = "/mnt/xmap_nas_alg/yzl/Recognition/AFile/datasets/hmdb51_bound"
        return os.path.join(base_path, x_path)
    else:
        # 都不需要，使用原始视频
        return os.path.join(original_video_path, x_path)

def extract_tool_decisions(first_turn_output: str) -> dict:
    """从first_turn_output中提取工具决策"""
    decisions = {
        'pose_estimation': 'no',
        'person_detection': 'no', 
        'noun_explanation': 'no'
    }
    
    # 提取姿态估计决策
    pose_pattern = r'<pose_estimation>\s*(yes|no)\s*</pose_estimation>'
    pose_match = re.search(pose_pattern, first_turn_output, re.IGNORECASE)
    if pose_match:
        decisions['pose_estimation'] = pose_match.group(1).lower()
    
    # 提取人员检测决策
    person_pattern = r'<person_detection>\s*(yes|no)\s*</person_detection>'
    person_match = re.search(person_pattern, first_turn_output, re.IGNORECASE)
    if person_match:
        decisions['person_detection'] = person_match.group(1).lower()
    
    # 提取名词解释决策
    noun_pattern = r'<noun_explanation>\s*(yes|no)\s*</noun_explanation>'
    noun_match = re.search(noun_pattern, first_turn_output, re.IGNORECASE)
    if noun_match:
        decisions['noun_explanation'] = noun_match.group(1).lower()
    
    return decisions


@dataclass
class GRPOScriptArguments(ScriptArguments):
    """
    Script arguments for the GRPO training script.

    Args:
        reward_funcs (`list[str]`):
            List of reward functions. Possible values: 'accuracy', 'format'.
    """

    reward_funcs: list[str] = field(
        default_factory=lambda: ["accuracy", "format"],
        metadata={"help": "List of reward functions. Possible values: 'accuracy', 'format'"},
    )
    max_pixels: Optional[int] = field(
        default=12845056,
        metadata={"help": "Maximum number of pixels for the image"},
    )
    min_pixels: Optional[int] = field(
        default=3136,
        metadata={"help": "Minimum number of pixels for the image"},
    )
    temporal: Optional[bool] = field(
        default=True,
        metadata={"help": "whether using temporal GRPO"},
    )
    len_control: Optional[bool] = field(
        default=True,
        metadata={"help": "whether using length reward"},
    )



def accuracy_reward(completions, solution, **kwargs):
    
    def extract_answer(text):
        # 处理嵌套的answer标签
        pattern = r'<answer>\s*<answer>(.*?)</answer>\s*</answer>'
        match = re.search(pattern, text, re.DOTALL)
        if match:
            return match.group(1).strip()
        
        # 如果没有嵌套，使用原来的模式
        pattern = r'<answer>\s*(.*?)\s*</answer>'
        match = re.search(pattern, text, re.DOTALL)
        if match:
            return match.group(1).strip()
        return ""
    
    def normalize(text):
        """
        标准化字符串：小写、去除空格、下划线和换行符
        """
        return text.lower().replace(" ", "").replace("_", "").replace("\n", "")
    
    def extract_body_parts_from_reasoning(text):
        """
        从推理过程中提取识别的身体部位，鲁棒处理无换行的紧凑文本
        返回：(body_parts_list, total_count)
        """
        # 定位 [1] 段落
        section_pattern = r"\[1\]\s*Observed body parts and movement characteristics:\s*(.*?)(?=\[2\]|$)"
        m = re.search(section_pattern, text, re.DOTALL | re.IGNORECASE)
        if not m:
            return [], 0
        section = m.group(1)

        # 在段落中直接用正则提取所有 "- Name:" 形式的名称，忽略是否换行
        # 捕获首字母大写/包含空格的名称，如 "Contact Points"
        name_pattern = r"-\s*([A-Za-z][A-Za-z ]*?):"
        parts = re.findall(name_pattern, section)
        # 去重并保持顺序
        seen = set()
        body_parts = []
        for p in parts:
            if p not in seen:
                seen.add(p)
                body_parts.append(p)
        return body_parts, len(body_parts)
    
    def extract_predicted_action_description(text, predicted_answer):
        """
        从content中提取预测动作的详细描述，只截取当前候选的描述，直到下一个"- Candidate:"或段落结束。
        """
        # 定位 [3] 段落
        section_pattern = r"\[3\]\s*Pattern comparison for each candidate:\s*(.*?)(?=\[4\]|$)"
        m = re.search(section_pattern, text, re.DOTALL | re.IGNORECASE)
        if not m:
            return ""
        section = m.group(1)

        # 构建候选项描述的正则，仅匹配当前预测答案，非贪婪直到下一个候选或结尾
        candidate = re.escape(predicted_answer)
        desc_pattern = rf"-\s*{candidate}\s*:\s*(.*?)(?=(?:\s+-\s*[A-Za-z][A-Za-z ]*:\s)|$)"
        m2 = re.search(desc_pattern, section, re.IGNORECASE | re.DOTALL)
        if not m2:
            return ""
        return m2.group(1).strip().lower()
    
    def calculate_body_parts_reward(body_parts, correct_answer, content):
        """
        计算身体部位识别的奖励
        """
        if not body_parts:
            return 0.0
        
        # 从content中提取预测动作的描述
        action_description = extract_predicted_action_description(content, correct_answer)
        
        if not action_description:
            return 0.0
        
        # 计算权重（按识别顺序：第一个=3，第二个=2，第三个=1）
        total_parts = len(body_parts)
        if total_parts == 0:
            return 0.0
        
        # 计算权重分母
        weight_denominator = sum(range(1, total_parts + 1))  # 1+2+3+...+n
        
        # 计算匹配的身体部位奖励
        matched_reward = 0.0
        for i, body_part in enumerate(body_parts):
            weight = (total_parts - i) / weight_denominator  # 第一个权重最大
            
            # 检查这个身体部位是否在预测动作的描述中
            if body_part.lower() in action_description:
                matched_reward += weight
        
        return matched_reward

    def normalize_number(num_str):
        try:
            num_str = num_str.replace(',', '')
            return float(num_str)
        except Exception as e:
            print(f"Error converting '{num_str}' to float: {e}")
            return None

    def wer(reference, hypothesis):
        ref_words = reference.split()
        hyp_words = hypothesis.split()
        m = len(ref_words)
        n = len(hyp_words)
        d = [[0]*(n+1) for _ in range(m+1)]
        for i in range(m+1):
            d[i][0] = i
        for j in range(n+1):
            d[0][j] = j
        for i in range(1, m+1):
            for j in range(1, n+1):
                if ref_words[i-1] == hyp_words[j-1]:
                    d[i][j] = d[i-1][j-1]
                else:
                    d[i][j] = 1 + min(d[i-1][j], d[i][j-1], d[i-1][j-1])
        return d[m][n] / max(1, m)


    def compute_rouge_score(reference, hypothesis, use_stemmer=True):
        scorer = rouge_scorer.RougeScorer(['rouge1', 'rouge2', 'rougeL'], use_stemmer=use_stemmer)
        scores = scorer.score(reference, hypothesis)
        average_fmeasure = (scores['rouge1'].fmeasure + scores['rouge2'].fmeasure + scores['rougeL'].fmeasure) / 3
        return average_fmeasure
    

    question_type = kwargs['problem_type'][0]
    
    contents = [completion[0]["content"] for completion in completions]
    current_time = datetime.now().strftime("%d-%H-%M-%S-%f")
    rewards = []

    first_turn_outputs = kwargs.get('first_turn_outputs', [""] * len(contents))
    for content, sol, first_turn_output in zip(contents, solution, first_turn_outputs):
    
        try:
            output_ans = extract_answer(content)
            gt_ans = extract_answer(sol)
            
            if question_type == "action-detection":
                # 基础奖励：答案正确性
                base_reward = 0.0
                if normalize(gt_ans) == normalize(output_ans):
                    base_reward = 1.0
                
                # 精细奖励：身体部位识别质量
                body_parts, total_parts = extract_body_parts_from_reasoning(content)
                body_parts_reward = calculate_body_parts_reward(body_parts, gt_ans, content)
                
                # 工具加分：答案正确且第一轮选择了任意工具（yes）=> +1
                tool_bonus = 0.0
                if base_reward > 0.0:
                    if re.search(r'<pose_estimation>\s*yes\s*</pose_estimation>', first_turn_output, re.IGNORECASE) or \
                       re.search(r'<person_detection>\s*yes\s*</person_detection>', first_turn_output, re.IGNORECASE) or \
                       re.search(r'<noun_explanation>\s*yes\s*</noun_explanation>', first_turn_output, re.IGNORECASE):
                        tool_bonus = 1.0

                # 总奖励 = 基础奖励 + 精细奖励 + 工具加分
                reward = base_reward + body_parts_reward + tool_bonus
                
                # 调试信息
                if os.getenv("DEBUG_MODE") == "true":
                    print(f"DEBUG Reward Calculation:")
                    print(f"  Base reward: {base_reward}")
                    print(f"  Body parts: {body_parts}")
                    print(f"  Body parts reward: {body_parts_reward}")
                    print(f"  Tool bonus: {tool_bonus}")
                    print(f"  Total reward: {reward}")
            else:
                reward = 0.0
        except Exception as e:
            print(f"Error in reward_fn for question_type '{question_type}': {e}")
            reward = 0.0
    
        rewards.append(reward)
        
        if os.getenv("DEBUG_MODE") == "true":
            log_path = os.getenv("LOG_PATH")
            # local_rank = int(os.getenv("LOCAL_RANK", 0))
            if log_path is not None:
                with open(log_path, "a", encoding="utf-8") as f:
                    f.write(f"------------- {current_time} Accuracy reward: {reward} -------------\n")
                    f.write(f"Content: {content}\n")
                    f.write(f"Solution: {sol}\n")
            
    return rewards


def format_reward(completions, **kwargs):
    """Reward function that checks if the completion has a specific format."""
    pattern = r"<think>.*?</think>\s*<answer>.*?</answer>"
    completion_contents = [completion[0]["content"] for completion in completions]
    matches = [re.fullmatch(pattern, content, re.DOTALL) for content in completion_contents]
    return [1.0 if match else 0.0 for match in matches]


reward_funcs_registry = {
    "accuracy": accuracy_reward,
    "format": format_reward,
}

SYSTEM_PROMPT = (
    "A conversation between User and Assistant. The user asks a question, and the Assistant solves it. The assistant "
    "first thinks about the reasoning process in the mind and then provides the user with the answer. The reasoning "
    "process and answer are enclosed within <think> </think> and <answer> </answer> tags, respectively, i.e., "
    "<think> reasoning process here </think><answer> answer here </answer>"
)


def main(script_args, training_args, model_args):
    # Get reward functions
    reward_funcs = [reward_funcs_registry[func] for func in script_args.reward_funcs]

    if script_args.dataset_name.endswith('.json') or script_args.dataset_name.endswith('.jsonl'):
        dataset =  DatasetDict({"train": Dataset.from_json(script_args.dataset_name)})
    else:
        # Load the dataset
        dataset = load_dataset(script_args.dataset_name, name=script_args.dataset_config)

    # 定义动作解释文件路径
    json_file_path = "/mnt/xmap_nas_alg/yzl/Recognition/FROSTER/zs_label_db/B2N_hmdb/train_rephrased.json"

    # 读取 JSON 文件
    with open(json_file_path, 'r') as f:
        actions = json.load(f)

    # 第一轮对话模板：判断需要哪些工具
    FIRST_TURN_TEMPLATE = """
    I will show you a video of human action. Your task is to determine which visual analysis tools would help you better identify the action. You have three independent tools available:

    1. **Pose Estimation Tool**: Adds skeleton keypoints and connections to show body pose and joint movements
    2. **Person Detection Tool**: Adds bounding boxes around detected persons to highlight human subjects
    3. **Noun Explanation Tool**: Provides detailed explanations of action types to help with classification

    Please analyze the video using step-by-step reasoning and decide for each tool independently:

    Analysis Requirements:
    1. **For Pose Estimation**: Assess whether body joint movements and pose details are crucial for identifying this action
    2. **For Person Detection**: Evaluate whether clearly identifying and localizing the person(s) would help with action recognition
    3. **For Noun Explanation**: Consider whether you need detailed explanations of action categories to make accurate classification

    Output Format:
    <think>step-by-step reasoning process:
    [1] Video content analysis: [describe what you observe in the video]
    [2] Pose estimation evaluation: [assess whether joint/skeleton info would help]
    [3] Person detection evaluation: [assess whether person localization would help]
    [4] Noun explanation evaluation: [assess whether action category details would help]
    [5] Final tool selection reasoning: [explain your decisions for each tool]
    </think>
    <pose_estimation>yes or no</pose_estimation>
    <person_detection>yes or no</person_detection>
    <noun_explanation>yes or no</noun_explanation>
    """

    # 第二轮对话模板：实际的动作识别
    SECOND_TURN_TEMPLATE = """
    I will provide you with a video and ask you to identify the human action shown. 
    Note: There is also an annotated version of this video available at: {ANNOTATED_VIDEO_PATH}
    The annotated video contains the same content but with additional annotations:
    - Blue bounding boxes around detected persons
    - Pose estimation points and skeleton lines showing body keypoints

    Your task is to follow this two-stage reasoning process:

    Input Format:
    Original Question: What kind of human action is shown in the video?
    All action types and their explanations are shown as follows:
    {ALLOWED_ANSWERS}.

    Analysis Requirements:

    **Stage 1: Body Part Mapping to Candidate Actions**
    1. Identify all body parts showing significant motion in the video (e.g., arms, legs, torso)
    2. For each identified body part:
    - Note its movement direction (up/down, rotational, etc.)
    - Record contact points with objects (if any)
    3. Match these observations to the action definitions in given action types
    4. Generate 2-3 candidate actions that best match the observed body part movements

    **Stage 2: Movement Pattern Refinement**
    For each candidate action:
    1. Extract the key movement patterns from its description in given action types
    2. Compare with the video's observed:
    - Temporal sequence of movements (which body part moves first)
    - Interaction patterns between body parts
    - Force application points (e.g., hand gripping vs. pulling)
    3. Score each candidate based on:
    - Body part involvement precision
    - Movement pattern similarity
    - Object interaction consistency

    Output Format:
    <think>step-by-step reasoning process:
    [1] Observed body parts and movement characteristics:
    - [Body Part 1]: [Direction/Contact Description]
    - [Body Part 2]: [Direction/Contact Description]
    ...
    [2] Matching candidate actions:
    - [Candidate 1]: Matches [Body Part A] [Movement Type]
    - [Candidate 2]: Matches [Body Part B] [Movement Type]
    ...
    [3] Pattern comparison for each candidate:
    - [Candidate 1]: [Score] - [Matching Details]
    - [Candidate 2]: [Score] - [Matching Details]
    </think>
    <answer>action-type</answer>
    """
        
    def make_conversation_image_and_video(example):
        """创建两轮对话的训练数据"""

        # 第一轮对话：工具选择
        first_turn_prompt = [{
            "role": "user",
            "content": [
                {
                    "type": example['data_type'],
                },
                {
                    "type": "text",
                    "text": FIRST_TURN_TEMPLATE
                }
            ]
        }]

        # 第二轮对话：动作识别
        second_turn_prompt = [{
            "role": "user",
            "content": [
                {
                    "type": example['data_type'],
                },
                {
                    "type": "text",
                    "text": SECOND_TURN_TEMPLATE.format(
                        ALLOWED_ANSWERS=get_allowed_answers_str(actions, True),
                        ANNOTATED_VIDEO_PATH=os.path.join("/mnt/xmap_nas_alg/yzl/Recognition/AFile/datasets/hmdb51_pose", example['path'])
                    )
                }
            ]
        }]

        # 注意：这里只提供第一轮用户消息和第二轮用户消息
        # GRPO训练器会动态生成第一轮助手回复，然后基于它构建第二轮完整对话
        # 最终训练的是第二轮的助手回复
        msg = {
            "prompt": first_turn_prompt + second_turn_prompt,
            "data_type": example['data_type'],
            "path": example['path'],
            "problem_type": example['problem_type'],
            "solution": example['solution'],
            "problem_id": example.get('problem_id', 0),
            "actions": actions,  # 传递动作定义用于动态选择
            "first_turn_template": FIRST_TURN_TEMPLATE,
            "second_turn_template": SECOND_TURN_TEMPLATE
        }

        return msg

    
    dataset = dataset.map(make_conversation_image_and_video)

    
    trainer_cls = Qwen2VLGRPOTrainer if not training_args.use_vllm else Qwen2VLGRPOVLLMTrainerModified
    print("using: ", trainer_cls)

    # Initialize the GRPO trainer
    trainer = trainer_cls(
        model=model_args.model_name_or_path,
        reward_funcs=reward_funcs,
        args=training_args,
        script_args=script_args,
        train_dataset=dataset[script_args.dataset_train_split],
        eval_dataset=dataset[script_args.dataset_test_split] if training_args.eval_strategy != "no" else None,
        peft_config=get_peft_config(model_args),
        attn_implementation=model_args.attn_implementation,
        max_pixels=script_args.max_pixels,
        min_pixels=script_args.min_pixels,
    )
    
    if training_args.resume_from_checkpoint is not None:
        checkpoint = training_args.resume_from_checkpoint
        trainer.train(resume_from_checkpoint=checkpoint)
    else:
        trainer.train()

    # Save and push to hub
    trainer.save_model(training_args.output_dir)
    if training_args.push_to_hub:
        trainer.push_to_hub(dataset_name=script_args.dataset_name)


if __name__ == "__main__":
    parser = TrlParser((GRPOScriptArguments, GRPOConfig, ModelConfig))
    script_args, training_args, model_args = parser.parse_args_and_config()
    main(script_args, training_args, model_args)
