# Copyright 2024. All rights reserved.
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
"""
Example usage:
accelerate launch \
    --config_file=deepspeed_zero2.yaml \
    train_video_llm.py \
    --dataset_name mfarre/simplevideoshorts \
    --model_name_or_path Qwen/Qwen2-VL-7B-Instruct \
    --per_device_train_batch_size 1 \
    --gradient_accumulation_steps 4 \
    --output_dir video-llm-output \
    --bf16 \
    --torch_dtype bfloat16 \
    --gradient_checkpointing
"""

import os
import json
import random
import requests
import torch
from datasets import load_dataset
from transformers import (
    AutoModelForVision2Seq,
    AutoProcessor,
    BitsAndBytesConfig,
    Qwen2VLProcessor,
    Qwen2VLForConditionalGeneration,
    Qwen2_5_VLForConditionalGeneration
)
from trl import (
    ModelConfig,
    ScriptArguments,
    SFTConfig,
    SFTTrainer,
    TrlParser,
    get_kbit_device_map,
    get_peft_config,
)
from accelerate import Accelerator
from qwen_vl_utils import process_vision_info

from datasets import Dataset, DatasetDict

from typing import List, Dict, Any
import argparse
import re

def extract_tool_decisions(first_turn_output: str) -> Dict[str, str]:
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

def get_allowed_answers_str(actions, use_noun_explanation=False):
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

def get_video_path(x_path, pose_estimation, person_detection, original_video_path="/mnt/xmap_nas_alg/yzl/Recognition/mmaction2/data/ucf101/UCF-101"):
    """根据工具决策选择对应的视频路径"""
    
    if pose_estimation == 'yes' and person_detection == 'no':
        # 只有姿态估计
        base_path = "/mnt/xmap_nas_alg/yzl/Recognition/mmaction2/data/ucf101/UCF-101_po"
        return os.path.join(base_path, x_path)
    elif pose_estimation == 'yes' and person_detection == 'yes':
        # 姿态估计 + 人员检测
        base_path = "/mnt/xmap_nas_alg/yzl/Recognition/mmaction2/data/ucf101/UCF-101_pose"
        return os.path.join(base_path, x_path)
    elif pose_estimation == 'no' and person_detection == 'yes':
        # 只有人员检测
        base_path = "/mnt/xmap_nas_alg/yzl/Recognition/mmaction2/data/ucf101/UCF-101_bound"
        return os.path.join(base_path, x_path)
    else:
        # 都不需要，使用原始视频
        return os.path.join(original_video_path, x_path)

def get_current_device():
    """Get the current device. For GPU we return the local process index to enable multiple GPU training."""
    return Accelerator().local_process_index if torch.cuda.is_available() else "cpu"

def download_video(url: str, folder: str = '/tmp/videos/') -> str:
    """Download video if not already present locally."""
    filename = url.split("/")[-1]
    local_path = os.path.join(folder, filename)

    if os.path.exists(local_path):
        return local_path

    try:
        with requests.get(url, stream=True) as r:
            r.raise_for_status()
            with open(local_path, 'wb') as f:
                for chunk in r.iter_content(chunk_size=8192):
                    if chunk:
                        f.write(chunk)
        return local_path
    except requests.RequestException as e:
        raise Exception(f"Failed to download video: {e}")

def prepare_dataset(example: Dict[str, Any]) -> Dict[str, List[Dict[str, Any]]]:
    """Prepare dataset example for training with two-turn dialogue mechanism."""

    system_message = "You are a helpful assistant"
    
    # 定义动作解释文件路径
    json_file_path = "/mnt/xmap_nas_alg/yzl/Recognition/AFile/json/ucf101/train_rephrased.json"

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

    # 从first_turn_output中提取工具决策信息
    first_turn_output = example.get('first_turn_output', '')
    tool_decisions = extract_tool_decisions(first_turn_output)
    pose_estimation = tool_decisions['pose_estimation']
    person_detection = tool_decisions['person_detection']
    noun_explanation = tool_decisions['noun_explanation']
    
    # 根据工具决策选择视频路径
    video_path = get_video_path(example['path'], pose_estimation, person_detection)
    
    # 根据是否需要名词解释选择答案格式
    need_noun_explanation = noun_explanation == 'yes'
    allowed_answers_str = get_allowed_answers_str(actions, need_noun_explanation)

    # 构建两轮对话的消息
    messages = [
        {
            "role": "system",
            "content": [{"type": "text", "text": system_message}]
        },
        # 第一轮对话：工具选择
        {
            "role": "user",
            "content": [
                {
                    "type": example['data_type'],
                    example['data_type']: os.path.join("/mnt/xmap_nas_alg/yzl/Recognition/mmaction2/data/ucf101/UCF-101", example['path'])
                },
                {
                    "type": "text",
                    "text": FIRST_TURN_TEMPLATE
                }
            ]
        },
        {
            "role": "assistant",
            "content": [{"type": "text", "text": first_turn_output}]
        },
        # 第二轮对话：动作识别
        {
            "role": "user",
            "content": [
                {
                    "type": example['data_type'],
                    example['data_type']: video_path
                },
                {
                    "type": "text",
                    "text": SECOND_TURN_TEMPLATE.format(
                        ALLOWED_ANSWERS=allowed_answers_str,
                        ANNOTATED_VIDEO_PATH=os.path.join("/mnt/xmap_nas_alg/yzl/Recognition/mmaction2/data/ucf101/UCF-101_pose", example['path'])
                    )
                }
            ]
        },
        {
            "role": "assistant",
            "content": [{"type": "text", "text": example.get('process', '') + "\n" + example['solution']}]
        }
    ]
    
    return {"messages": messages}

def collate_fn(examples: List[Dict[str, Any]]) -> Dict[str, torch.Tensor]:
    """Collate batch of examples for training."""
    texts = []
    # video_inputs = []
    # image_inputs = []

    for i, example in enumerate(examples):
        try:

            texts.append(processor.apply_chat_template(example["messages"], tokenize=False))
            image_inputs, video_inputs, video_kwargs = process_vision_info(example["messages"], return_video_kwargs=True)
            
        except Exception as e:
            raise ValueError(f"Failed to process example {i}: {e}")

    inputs = processor(
        text=texts,
        images=image_inputs,
        videos=video_inputs,
        return_tensors="pt",
        padding=True
    )

    labels = inputs["input_ids"].clone()
    labels[labels == processor.tokenizer.pad_token_id] = -100

    # Handle visual tokens based on processor type
    visual_tokens = [151652, 151653, 151656] if isinstance(processor, Qwen2VLProcessor) else [
        processor.tokenizer.convert_tokens_to_ids(processor.image_token)
    ]

    for visual_token_id in visual_tokens:
        labels[labels == visual_token_id] = -100

    inputs["labels"] = labels
    return inputs

if __name__ == "__main__":
    # Parse arguments
    parser = TrlParser((ScriptArguments, SFTConfig, ModelConfig))
    script_args, training_args, model_config = parser.parse_args_and_config()
    
    # Configure training args
    training_args.gradient_checkpointing_kwargs = dict(use_reentrant=False)
    training_args.remove_unused_columns = False
    training_args.dataset_kwargs = {"skip_prepare_dataset": True}

    # Load dataset
    if script_args.dataset_name.endswith('.json') or script_args.dataset_name.endswith('.jsonl'):
        dataset =  DatasetDict({"train": Dataset.from_json(script_args.dataset_name)})
    else:
        # Load the dataset
        dataset = load_dataset(script_args.dataset_name, name=script_args.dataset_config)

    # Setup model
    torch_dtype = (
        model_config.torch_dtype
        if model_config.torch_dtype in ["auto", None]
        else getattr(torch, model_config.torch_dtype)
    )

    # # Quantization configuration for 4-bit training
    # bnb_config = BitsAndBytesConfig(
    #     load_in_4bit=True,
    #     bnb_4bit_use_double_quant=True,
    #     bnb_4bit_quant_type="nf4",
    #     bnb_4bit_compute_dtype=torch.bfloat16
    # )

    # Model initialization
    model_kwargs = dict(
        revision=model_config.model_revision,
        trust_remote_code=model_config.trust_remote_code,
        torch_dtype=torch_dtype,
        device_map=get_kbit_device_map(),
        # quantization_config=bnb_config,
    )
    
    
    if "Qwen2-VL" in model_config.model_name_or_path:
        model = Qwen2VLForConditionalGeneration.from_pretrained(model_config.model_name_or_path, **model_kwargs)
    elif "Qwen2.5-VL" in model_config.model_name_or_path:
        model = Qwen2_5_VLForConditionalGeneration.from_pretrained(model_config.model_name_or_path, **model_kwargs)
    else:
        model = AutoModelForVision2Seq.from_pretrained(model_config.model_name_or_path, **model_kwargs)

    processor = AutoProcessor.from_pretrained(
        model_config.model_name_or_path,
        trust_remote_code=model_config.trust_remote_code
    )

    # Prepare dataset
    prepared_dataset = [prepare_dataset(example) for example in dataset['train']]

    # Initialize trainer
    trainer = SFTTrainer(
        model=model,
        args=training_args,
        train_dataset=prepared_dataset,
        data_collator=collate_fn,
        peft_config=get_peft_config(model_config),
        # tokenizer=processor.tokenizer
    )

    # Train model
    trainer.train()

    # Save final model

    trainer.save_model(training_args.output_dir)
    processor.save_pretrained(training_args.output_dir)

    if trainer.accelerator.is_main_process:
        # Restore k,v cache for fast inference
        trainer.model.config.use_cache = True
        trainer.model.config.save_pretrained(training_args.output_dir)

    # Cleanup
    del model
    del trainer
    torch.cuda.empty_cache()
