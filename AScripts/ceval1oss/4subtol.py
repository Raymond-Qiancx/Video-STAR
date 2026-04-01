import os
import json
import re
import time
from tqdm import tqdm
from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction
from rouge_score import rouge_scorer
import torch

from transformers import AutoProcessor, AutoTokenizer
from vllm import LLM, SamplingParams
from qwen_vl_utils import process_vision_info
import argparse

BSZ = 50

# --- 修改开始：使用命令行参数传入路径 ---
parser = argparse.ArgumentParser(description="Evaluation benchmark")
parser.add_argument('--prompt-path', type=str, required=True, help='Path to the input JSON/JSONL prompt file')
parser.add_argument('--model-path', type=str, required=True, help='Path to the pre-trained model directory')
parser.add_argument('--output-path', type=str, required=True, help='Path to the output JSON file')
parser.add_argument('--json-file-path', type=str, required=True, help='Path to the output JSON file')
parser.add_argument('--original-video-path', type=str, required=True)
parser.add_argument('--annotated-video-path', type=str, required=True)
parser.add_argument('--measure-gflops', default=True, action='store_true', help='Profile batches with torch.profiler to estimate GFLOPS')
args = parser.parse_args()

PROMPT_PATH = args.prompt_path
MODEL_PATH = args.model_path
OUTPUT_PATH = args.output_path
json_file_path = args.json_file_path
original_video_path = args.original_video_path
annotated_video_path = args.annotated_video_path

# --- 修改结束 ---

llm = LLM(
    model=MODEL_PATH,
    tensor_parallel_size=1,
    max_model_len=8192 * 2,
    gpu_memory_utilization=0.9,
)

sampling_params = SamplingParams(
    temperature=0.8,
    top_p=0.001,
    max_tokens=4096,
    stop_token_ids=[],
)

processor = AutoProcessor.from_pretrained(MODEL_PATH)
tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH)
tokenizer.padding_side = "left"
processor.tokenizer = tokenizer

gflops_summary = {
    "total_flops": 0.0,
    "total_time": 0.0,
    "batch_count": 0,
}


def maybe_measure_gflops(description, callable_fn):
    """Optionally profile FLOPs and wall time for a callable."""
    if not args.measure_gflops:
        return callable_fn()

    profiler_kwargs = {
        "activities": [
            torch.profiler.ProfilerActivity.CPU,
            torch.profiler.ProfilerActivity.CUDA,
        ]
    }
    with_flops_kw = {"with_flops": True}
    profiler_kwargs.update(with_flops_kw)

    torch.cuda.synchronize() if torch.cuda.is_available() else None
    start = time.perf_counter()
    with torch.profiler.profile(**profiler_kwargs) as prof:
        result = callable_fn()
    torch.cuda.synchronize() if torch.cuda.is_available() else None
    elapsed = time.perf_counter() - start

    total_flops = 0.0
    for evt in prof.key_averages():
        evt_flops = getattr(evt, "flops", None)
        if evt_flops is not None:
            total_flops += evt_flops

    total_gflop = total_flops / 1e9
    gflops = total_gflop / elapsed if elapsed > 0 else 0.0
    print(f"[GFLOPS] {description}: {gflops:.2f} GFLOPS ({total_gflop:.2f} GFLOP over {elapsed:.3f}s)")

    gflops_summary["total_flops"] += total_flops
    gflops_summary["total_time"] += elapsed
    gflops_summary["batch_count"] += 1
    return result

if PROMPT_PATH.endswith('.jsonl'):
    data = []
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

# 第二轮对话模板：实际的动作识别
SECOND_TURN_TEMPLATE = (
"""
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
)

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

final_output = []
start_idx = 0
# if os.path.exists(OUTPUT_PATH):
#     try:
#         with open(OUTPUT_PATH, "r", encoding="utf-8") as f:
#             existing = json.load(f)
#             final_output = existing.get("results", [])
#             start_idx = len(final_output)
#             print(f"Resuming from sample index {start_idx}")
#     except Exception as e:
#         print(f"Error reading existing output file: {e}")


def extract_think(output_str):
    pattern = r'<think>\s*(.*?)\s*</think>'
    match = re.search(pattern, output_str, re.DOTALL)
    if match:
        return match.group(1).strip()
    return ""

def extract_answer(text):
    pattern = r'<answer>(.*?)</answer>'
    match = re.search(pattern, text, re.DOTALL)
    if match:
        return match.group(0).strip()  # 保留标签
    return ""

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
        base_path = original_video_path
        return os.path.join(base_path, x_path)
    elif pose_estimation == 'yes' and person_detection == 'yes':
        # 姿态估计 + 人员检测
        base_path = original_video_path
        return os.path.join(base_path, x_path)
    elif pose_estimation == 'no' and person_detection == 'yes':
        # 只有人员检测
        base_path = original_video_path
        return os.path.join(base_path, x_path)
    else:
        # 都不需要，使用原始视频
        base_path = original_video_path
        return os.path.join(base_path, x_path)


# 第一轮对话：判断需要哪些工具
print("Starting first turn: Tool decision...")
first_turn_decisions = []  # 现在存储字典格式的决策
first_turn_outputs = []  # 存储第一轮的完整输出

for i in tqdm(range(start_idx, len(first_turn_messages), BSZ), desc="First turn - Tool decision"):
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

        outputs = maybe_measure_gflops(
            description=f"First turn batch {i // BSZ}",
            callable_fn=lambda: llm.generate(llm_inputs, sampling_params=sampling_params)
        )
        batch_output_text = [out.outputs[0].text for out in outputs]
        
        # 提取决策结果并保存完整输出
        for output_text in batch_output_text:
            decisions = extract_first_turn_decisions(output_text)
            first_turn_decisions.append(decisions)
            first_turn_outputs.append(output_text.replace("\n", "").strip())
            # print(f"Tool decisions: Pose={decisions['pose_estimation']}, Person={decisions['person_detection']}, Noun={decisions['noun_explanation']}")
        
    except Exception as e:
        print('First turn error at batch:', i)
        print('Exception:', e)
        # 如果出错，默认不使用任何工具
        default_decisions = {'pose_estimation': 'no', 'person_detection': 'no', 'noun_explanation': 'no'}
        first_turn_decisions.extend([default_decisions] * current_batch_size)
        first_turn_outputs.extend(['error'] * current_batch_size)

print(f"First turn completed. Decisions: {len(first_turn_decisions)}")

# 第二轮对话：基于决策结果进行动作识别
print("Starting second turn: Action recognition...")
second_turn_messages = []

for local_idx, x in enumerate(data[start_idx:]):
    global_idx = start_idx + local_idx
    question = x['problem']
    original_video_path_spe = os.path.join(original_video_path, x['path'])
    annotated_video_path_spe = os.path.join(annotated_video_path, x['path'])
    
    # 根据第一轮决策选择视频路径和答案格式
    decisions = first_turn_decisions[local_idx] if local_idx < len(first_turn_decisions) else {'pose_estimation': 'no', 'person_detection': 'no', 'noun_explanation': 'no'}
    
    # 确定是否需要名词解释
    need_noun_explanation = decisions['noun_explanation'] == 'yes'
    
    # 根据工具决策选择视频路径
    video_path = get_video_path(x['path'], decisions['pose_estimation'], decisions['person_detection'])
    # print(f"Using video for sample {global_idx}: {video_path}")
    
    # 根据是否需要名词解释选择答案格式
    allowed_answers_str = get_allowed_answers_str(need_noun_explanation)
    
    msg = [{
        "role": "user",
        "content": [
            {
                "type": x['data_type'],
                x['data_type']: video_path
            },
            {
                "type": "text",
                "text": SECOND_TURN_TEMPLATE.format(ALLOWED_ANSWERS=allowed_answers_str, ANNOTATED_VIDEO_PATH = annotated_video_path_spe)
            }
        ]
    }]
    second_turn_messages.append(msg)

# 处理第二轮对话
for i in tqdm(range(0, len(second_turn_messages), BSZ), desc="Second turn - Action recognition"):
    batch_messages = second_turn_messages[i:i + BSZ]
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

        outputs = maybe_measure_gflops(
            description=f"Second turn batch {i // BSZ}",
            callable_fn=lambda: llm.generate(llm_inputs, sampling_params=sampling_params)
        )
        batch_output_text = [out.outputs[0].text for out in outputs]
        
    except Exception as e:
        print('Second turn error at batch:', i)
        print('Exception:', e)
        batch_output_text = ['<answer>error</answer>'] * current_batch_size

    # 保存结果
    for j, (sample, model_output) in enumerate(zip(data[start_idx + i:start_idx + i + current_batch_size], batch_output_text)):
        think_chain = extract_think(model_output)
        final_ans = extract_answer(model_output)
        
        # 添加第一轮决策信息和输出
        local_sample_idx = i + j  # 在当前处理范围内的索引
        global_sample_idx = start_idx + i + j  # 全局索引
        tool_decisions = first_turn_decisions[local_sample_idx] if local_sample_idx < len(first_turn_decisions) else {'pose_estimation': 'no', 'person_detection': 'no', 'noun_explanation': 'no'}
        first_turn_output = first_turn_outputs[local_sample_idx] if local_sample_idx < len(first_turn_outputs) else 'error'
        
        sample["pose_estimation"] = tool_decisions['pose_estimation']
        sample["person_detection"] = tool_decisions['person_detection']
        sample["noun_explanation"] = tool_decisions['noun_explanation']
        sample["first_turn_output"] = first_turn_output
        sample["second_turn_output"] = model_output.replace("\n", "").strip()
        sample["predict"] = final_ans.replace("\n", "").strip()
        final_output.append(sample)
        print(f"Sample {global_sample_idx}: Pose={tool_decisions['pose_estimation']}, Person={tool_decisions['person_detection']}, Noun={tool_decisions['noun_explanation']}, Predict={final_ans}")

# 保存最终输出
with open(OUTPUT_PATH, "w", encoding="utf-8") as f:
    json.dump(final_output, f, ensure_ascii=False, indent=4)

if args.measure_gflops and gflops_summary["total_time"] > 0:
    total_gflop = gflops_summary["total_flops"] / 1e9
    overall_gflops = total_gflop / gflops_summary["total_time"]
    print(f"[GFLOPS] Overall: {overall_gflops:.2f} GFLOPS ({total_gflop:.2f} GFLOP over {gflops_summary['total_time']:.3f}s)")
    
    # 保存 GFLOPS 统计结果到文件
    output_dir = os.path.dirname(OUTPUT_PATH)
    output_basename = os.path.basename(OUTPUT_PATH)
    output_name, output_ext = os.path.splitext(output_basename)
    gflops_output_path = os.path.join(output_dir, f"{output_name}_gpu{output_ext}")
    
    gflops_result = {
        "overall_gflops": overall_gflops,
        "total_gflop": total_gflop,
        "total_time_seconds": gflops_summary["total_time"],
        "total_flops": gflops_summary["total_flops"],
        "batch_count": gflops_summary.get("batch_count", 0)
    }
    
    with open(gflops_output_path, "w", encoding="utf-8") as f:
        json.dump(gflops_result, f, ensure_ascii=False, indent=4)
    
    print(f"[GFLOPS] Statistics saved to: {gflops_output_path}")

