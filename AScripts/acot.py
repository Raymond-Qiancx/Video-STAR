import os
import json
import re
from tqdm import tqdm
from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction
from rouge_score import rouge_scorer
import torch
import argparse

from transformers import AutoProcessor, AutoTokenizer
from vllm import LLM, SamplingParams
from qwen_vl_utils import process_vision_info

# 添加命令行参数支持
parser = argparse.ArgumentParser(description="Generate CoT training data")
parser.add_argument('--model-path', type=str, default="/mnt/xmap_nas_alg/yzl/Amodel/Qwen2.5-VL-72B-Instruct", help='Path to the pre-trained model directory')
parser.add_argument('--json-file-path', type=str, default="/mnt/xmap_nas_alg/yzl/Recognition/FROSTER/zs_label_db/B2N_hmdb/train_rephrased.json", help='Path to action definitions JSON file')
parser.add_argument('--original-video-path', type=str, default="/mnt/xmap_nas_alg/yzl/Recognition/AFile/datasets/hmdb51", help='Path to original video directory')
parser.add_argument('--annotated-video-path', type=str, default="/mnt/xmap_nas_alg/yzl/Recognition/AFile/datasets/hmdb51_pose", help='Path to annotated video directory')
args = parser.parse_args()

MODEL_PATH = args.model_path
json_file_path = args.json_file_path
original_video_path = args.original_video_path
annotated_video_path = args.annotated_video_path
BSZ = 100

llm = LLM(
    model=MODEL_PATH,
    tensor_parallel_size=torch.cuda.device_count(),
    max_model_len=8192 * 2,
    gpu_memory_utilization=0.9,
    limit_mm_per_prompt={"image": 2, "video": 2},  # 每个prompt最多1个视频
)

# Add stop tokens for structured output
sampling_params = SamplingParams(
    temperature=0.6,
    top_p=0.8,
    max_tokens=4096,
    stop_token_ids=[]
)

processor = AutoProcessor.from_pretrained(MODEL_PATH)
tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH)
tokenizer.padding_side = "left"
processor.tokenizer = tokenizer

for dataset_name in ['your_data_name']:

    OUTPUT_PATH = "/mnt/xmap_nas_alg/yzl/Recognition/AScripts/acot/train_4subtol_cot_final.json"
    PROMPT_PATH = "/mnt/xmap_nas_alg/yzl/Recognition/AScripts/acot/train_final.json"
    
    data = []
    if PROMPT_PATH.endswith('.jsonl'):
        with open(PROMPT_PATH, "r", encoding="utf-8") as f:
            for line in f:
                data.append(json.loads(line))
    elif PROMPT_PATH.endswith('.json'):
        with open(PROMPT_PATH, "r", encoding="utf-8") as f:
            data = json.load(f)
    else:
        raise ValueError("Input file must be .json or .jsonl")

    # 读取 JSON 文件
    with open(json_file_path, 'r') as f:
        actions = json.load(f)

    # 构建允许答案的字符串（根据是否需要名词解释决定格式）
    def get_allowed_answers_str(use_noun_explanation=False):
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

    # 第一轮对话模板：判断需要哪些工具
    FIRST_TURN_TEMPLATE = (
    """
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
    )

    # 第二轮对话模板：生成CoT训练数据
    SECOND_TURN_TEMPLATE = (
    """
    I will provide you with a video and ask you to generate a comprehensive Chain-of-Thought (CoT) reasoning process for identifying the human action shown.
    
    Note: There is also an annotated version of this video available at: {ANNOTATED_VIDEO_PATH}
    The annotated video contains the same content but with additional annotations:
    - Blue bounding boxes around detected persons
    - Pose estimation points and skeleton lines showing body keypoints

    Your task is to create detailed step-by-step reasoning that demonstrates how to identify this action.

    Input Format:
    Original Question: What kind of human action is shown in the video?
    All action types and their explanations are shown as follows:
    {ALLOWED_ANSWERS}.
    Ground Truth Answer: {original_answer}

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
    [4] Final decision reasoning: [Explain why the ground truth answer is correct]
    </think>
    <answer>{original_answer}</answer>
    """
    )

    def extract_first_turn_decisions(text):
        """提取第一轮对话的三个工具决策"""
        decisions = {
            'pose_estimation': 'no',
            'person_detection': 'no', 
            'noun_explanation': 'no'
        }
        
        # 提取姿态估计决策
        pose_pattern = r'<pose_estimation>\s*(yes|no)\s*</pose_estimation>'
        pose_match = re.search(pose_pattern, text, re.IGNORECASE)
        if pose_match:
            decisions['pose_estimation'] = pose_match.group(1).lower()
        
        # 提取人员检测决策
        person_pattern = r'<person_detection>\s*(yes|no)\s*</person_detection>'
        person_match = re.search(person_pattern, text, re.IGNORECASE)
        if person_match:
            decisions['person_detection'] = person_match.group(1).lower()
        
        # 提取名词解释决策
        noun_pattern = r'<noun_explanation>\s*(yes|no)\s*</noun_explanation>'
        noun_match = re.search(noun_pattern, text, re.IGNORECASE)
        if noun_match:
            decisions['noun_explanation'] = noun_match.group(1).lower()
        
        return decisions

    def get_video_path(x_path, pose_estimation, person_detection):
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

    # 构建第一轮对话消息（判断是否需要工具）
    first_turn_messages = []
    for x in data:
        original_video_path_spe = os.path.join(original_video_path, x['path'])
        
        msg = [{
            "role": "user",
            "content": [
                {
                    "type": x['data_type'],
                    x['data_type']: original_video_path_spe
                },
                {
                    "type": "text",
                    "text": FIRST_TURN_TEMPLATE
                }
            ]
        }]
        first_turn_messages.append(msg)

    # 第一轮对话：判断需要哪些工具
    print("Starting first turn: Tool decision...")
    first_turn_decisions = []  # 存储字典格式的决策
    first_turn_outputs = []  # 存储第一轮的完整输出

    for i in tqdm(range(0, len(first_turn_messages), BSZ), desc="First turn - Tool decision"):
        batch_messages = first_turn_messages[i:i + BSZ]
        current_batch_size = len(batch_messages)

        prompts = [processor.apply_chat_template(msg, tokenize=False, add_generation_prompt=True) for msg in batch_messages]
        
        try:
            image_inputs, video_inputs, video_kwargs = process_vision_info(batch_messages, return_video_kwargs=True)
            
            image_idx = 0
            video_idx = 0
            llm_inputs = []
            
            for idx, prompt in enumerate(prompts):
                mm_type = batch_messages[idx][0]['content'][0]['type']
                sample_mm_data = {}
                sample_video_kw = {}
                if mm_type == 'image':
                    sample_mm_data["image"] = image_inputs[image_idx]
                    image_idx += 1
                elif mm_type == 'video':
                    sample_mm_data["video"] = video_inputs[video_idx]
                    for key, value in video_kwargs.items():
                        sample_video_kw[key] = value[video_idx]
                    video_idx += 1
                        
                llm_inputs.append({
                    "prompt": prompt,
                    "multi_modal_data": sample_mm_data,
                    "mm_processor_kwargs": sample_video_kw,
                })

            outputs = llm.generate(llm_inputs, sampling_params=sampling_params)
            batch_output_text = [out.outputs[0].text for out in outputs]
            
            # 提取决策结果并保存完整输出
            for output_text in batch_output_text:
                decisions = extract_first_turn_decisions(output_text)
                first_turn_decisions.append(decisions)
                first_turn_outputs.append(output_text.replace("\n", "").strip())
            
        except Exception as e:
            print('First turn error at batch:', i)
            print('Exception:', e)
            # 如果出错，默认不使用任何工具
            default_decisions = {'pose_estimation': 'no', 'person_detection': 'no', 'noun_explanation': 'no'}
            first_turn_decisions.extend([default_decisions] * current_batch_size)
            first_turn_outputs.extend(['error'] * current_batch_size)

    print(f"First turn completed. Decisions: {len(first_turn_decisions)}")

    # 第二轮对话：基于决策结果生成CoT数据
    print("Starting second turn: CoT generation...")
    messages = []
    for i, x in enumerate(data):
        question = x['problem']
        answer = x['solution']
        
        # 根据第一轮决策选择视频路径和答案格式
        decisions = first_turn_decisions[i] if i < len(first_turn_decisions) else {'pose_estimation': 'no', 'person_detection': 'no', 'noun_explanation': 'no'}
        
        # 确定是否需要名词解释
        need_noun_explanation = decisions['noun_explanation'] == 'yes'
        
        # 根据工具决策选择视频路径
        video_path = get_video_path(x['path'], decisions['pose_estimation'], decisions['person_detection'])
        
        # 根据是否需要名词解释选择答案格式
        allowed_answers_str = get_allowed_answers_str(need_noun_explanation)
        
        # 构建注释视频路径
        annotated_video_path_spe = os.path.join(annotated_video_path, x['path'])

        msg = [{
            "role": "user",
            "content": [
                {
                    "type": x['data_type'],
                    x['data_type']: video_path
                },
                {
                    "type": "text",
                    "text": SECOND_TURN_TEMPLATE.format(
                        original_answer=answer, 
                        ALLOWED_ANSWERS=allowed_answers_str,
                        ANNOTATED_VIDEO_PATH=annotated_video_path_spe
                    )
                }
            ]
        }]
        messages.append(msg)

    # For resume
    final_output = []
    start_idx = 0
    if os.path.exists(OUTPUT_PATH):
        try:
            with open(OUTPUT_PATH, "r", encoding="utf-8") as f:
                existing = json.load(f)
                final_output = existing.get("results", [])
                start_idx = len(final_output)
                print(f"Resuming from sample index {start_idx}")
        except Exception as e:
            print(f"Error reading existing output file: {e}")

    def extract_think(output_str):
        pattern = r'<think>\s*(.*?)\s*</think>'
        match = re.search(pattern, output_str, re.DOTALL)
        if match:
            return match.group(1).strip()
        return ""

    def extract_answer(text):
        pattern = r'<answer>\s*(.*?)\s*</answer>'
        match = re.search(pattern, text, re.DOTALL)
        if match:
            return match.group(1).strip()
        return ""

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

    for i in tqdm(range(start_idx, len(messages), BSZ), desc="Processing batches"):
        batch_messages = messages[i:i + BSZ]

        prompts = [processor.apply_chat_template(msg, tokenize=False, add_generation_prompt=True) for msg in batch_messages]
        
        try:
            image_inputs, video_inputs, video_kwargs = process_vision_info(batch_messages, return_video_kwargs=True)
            
            image_idx = 0
            video_idx = 0

            llm_inputs = []

            
            for idx, prompt in enumerate(prompts):
                mm_type = batch_messages[idx][0]['content'][0]['type']
                sample_mm_data = {}
                sample_video_kw = {}
                if mm_type == 'image':
                    sample_mm_data["image"] = image_inputs[image_idx]
                    image_idx += 1
                elif mm_type == 'video':
                    sample_mm_data["video"] = video_inputs[video_idx]
                    for key, value in video_kwargs.items():
                        sample_video_kw[key] = value[video_idx]
                    video_idx += 1
                        
                
                llm_inputs.append({
                    "prompt": prompt,
                    "multi_modal_data": sample_mm_data,
                    "mm_processor_kwargs": sample_video_kw,
                })
                

            outputs = llm.generate(llm_inputs, sampling_params=sampling_params)
            batch_output_text = [out.outputs[0].text for out in outputs]
            
        except Exception as e:
            print('error:', data[i]['path'])
            batch_output_text = ['<answer>error</answer>'] * BSZ
            

        for j, (sample, model_output) in enumerate(zip(data[i:i+BSZ], batch_output_text), start=i):
            think_chain = extract_think(model_output)
            final_ans = extract_answer(model_output)
            
            # 添加第一轮决策信息
            sample_idx = i + j
            tool_decisions = first_turn_decisions[sample_idx] if sample_idx < len(first_turn_decisions) else {'pose_estimation': 'no', 'person_detection': 'no', 'noun_explanation': 'no'}
            first_turn_output = first_turn_outputs[sample_idx] if sample_idx < len(first_turn_outputs) else 'error'
            
            # 添加工具决策到样本中
            sample["pose_estimation"] = tool_decisions['pose_estimation']
            sample["person_detection"] = tool_decisions['person_detection']
            sample["noun_explanation"] = tool_decisions['noun_explanation']
            sample["first_turn_output"] = first_turn_output
            sample["second_turn_output"] = model_output.replace("\n", "").strip()
            sample["process"] = model_output
            final_output.append(sample)
            
        
        try:
            with open(OUTPUT_PATH, "w", encoding="utf-8") as f:
                json.dump({"results": final_output}, f, indent=2, ensure_ascii=False)
            print(f"Processed batch {(i - start_idx)//BSZ + 1}, saved {len(final_output)} samples.")
        except Exception as e:
            print(f"Error writing to output file: {e}")

    print(f"Results saved to {OUTPUT_PATH}")
