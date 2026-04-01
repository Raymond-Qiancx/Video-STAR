def compute_loss(self, model, inputs, return_outputs=False, num_items_in_batch=None):
            import time
            import datetime
            
            # --- [Debug 工具] ---
            rank = dist.get_rank() if dist.is_available() and dist.is_initialized() else 0
            def dprint(msg):
                # 加上时间戳，方便定位卡死位置
                timestamp = datetime.datetime.now().strftime("%H:%M:%S")
                print(f"[{timestamp}][Rank {rank}] {msg}", flush=True)

            if return_outputs:
                raise ValueError("GRPOTrainer does not support returning outputs")

            assert self.TOOL_SELECTION_TEMPLATE is not None
            assert self.ANSWER_TEMPLATE is not None

            device = self.accelerator.device
            B = len(inputs)
            G = self.num_generations 
            step = int(getattr(self.state, "global_step", 0))
            LOG_EVERY = 1      
            MAX_SHOW = 2        
            DO_LOG = (step % LOG_EVERY == 0)

            # 确保所有 rank 的 batch size 一致
            self._assert_per_rank_B_same(B, device)

            # ============================================================
            # Step 1: Turn 1 (Tool Selection) Generation
            # ============================================================
            dprint(f"🚀 Start compute_loss (Step {step}).")
            turn1_messages = self._build_turn1_messages(inputs)
            turn1_prompts = [
                self.processing_class.apply_chat_template(m, tokenize=False, add_generation_prompt=True)
                for m in turn1_messages
            ]

            try:
                image_inputs, video_inputs, video_kwargs = process_vision_info(
                    turn1_messages, return_video_kwargs=True
                )
            except Exception as e:
                dprint(f"❌ process_vision_info (Turn 1) Error: {e}")
                raise e

            t1_inputs = self.processing_class(
                text=turn1_prompts,
                images=image_inputs,
                videos=video_inputs,
                return_tensors="pt",
                padding=True,
                padding_side="left",
                add_special_tokens=False,
                **video_kwargs,
            )
            t1_inputs = super()._prepare_inputs(t1_inputs)

            for k, v in list(t1_inputs.items()):
                if torch.is_tensor(v):
                    t1_inputs[k] = v.to(device, non_blocking=True)

            gen1_cfg = copy.deepcopy(self.generation_config)
            gen1_cfg.num_return_sequences = G

            # Generate Turn 1
            with unwrap_model_for_generation(model, self.accelerator) as unwrapped:
                t1_ids = unwrapped.generate(**t1_inputs, generation_config=gen1_cfg)

            prompt_len = t1_inputs["input_ids"].size(1)
            t1_comp_ids = t1_ids[:, prompt_len:]
            t1_texts = self.processing_class.batch_decode(t1_comp_ids, skip_special_tokens=True)

            if DO_LOG:
                self._rank0_print(f"\n[step={step}] Turn1 outputs (first {min(MAX_SHOW, len(t1_texts))})")
                for j in range(min(MAX_SHOW, len(t1_texts))):
                    self._rank0_print(f"--- t1[{j}] ---\n{t1_texts[j]}")

            # 清理 Turn 1 显存
            del t1_inputs, t1_ids, t1_comp_ids
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

            # ============================================================
            # Step 2: Execute Tools (B*G)
            # ============================================================
            calls_list = [extract_tool_calls(t) for t in t1_texts]
            expanded_inputs = [ex for ex in inputs for _ in range(G)]
            
            tool_reports = []
            clip_images = []

            # Sync Barrier 1: 确保所有 Rank 准备好运行工具
            if dist.is_available() and dist.is_initialized():
                dist.barrier()
            
            for idx, (calls, ex) in enumerate(zip(calls_list, expanded_inputs)):
                try:
                    # 调用工具
                    report, clip_path = self.tool_runner.execute_tools_and_report(calls, ex)
                    tool_reports.append(report)
                    clip_images.append(clip_path)
                except Exception as e:
                    dprint(f"CRITICAL TOOL FAILURE: {e}")
                    tool_reports.append(f"Tool Error: {e}")
                    clip_images.append(None)

            # Sync Barrier 2: 等待所有 Rank 跑完工具，防止 NCCL 超时
            if dist.is_available() and dist.is_initialized():
                dist.barrier()

            if DO_LOG:
                self._rank0_print(f"\n[step={step}] Tool Reports")
                for j in range(min(MAX_SHOW, len(tool_reports))):
                    rep = tool_reports[j]
                    if len(rep) > 200: rep = rep[:200] + "..."
                    self._rank0_print(f"--- Report[{j}] ---\n{rep}")

            # ============================================================
            # Step 3: Turn 2 (Final Answer) Generation
            # ============================================================
            turn2_messages = []
            for i in range(B * G):
                bidx = i // G

                input_item = inputs[bidx]
                media_path = self._fix_media_path(input_item)
                dataset_name = input_item.get("dataset_name", "") # "fakeVV"
                question = input_item["problem"]
                data_type = input_item.get("data_type", "video") # "video"

                msgs = []

                user_content = []

                if dataset_name == 'fakeVV':
                    # FakeVV: 必须再次稀疏读取 (保证和 Step 1 一致且不爆内存)
                    frames = self.sparse_sample_video(media_path, num_frames=8)
                    user_content.append({
                        "type": "video",
                        "video": frames
                    })
                else:
                    # FakeSV: 保持原样
                    user_content.append({
                        "type": data_type, 
                        data_type: media_path, 
                    })
                
                user_content.append({
                    "type": "text", 
                    "text": self.TOOL_SELECTION_TEMPLATE.format(Question=question)
                })

                msgs.append({
                    "role": "user",
                    "content": user_content,
                })

                msgs.append({"role": "assistant", "content": t1_texts[i]})
                
                # Construct Turn 2 User Input (Tool Report + Clip)
                content = [{"type": "text", "text": tool_reports[i] + "\n\n"}]

                # 创建一个极小的黑色 Dummy Image (1x1 像素)
                # 只要让它过一遍 Vision Encoder，保证计算图路径一致即可
                dummy_image_path = os.path.abspath(os.path.join(self.args.output_dir, "dummy_32x32.png"))
            
                # 2. 只有 Rank 0 负责创建文件
                if rank == 0:
                    if not os.path.exists(dummy_image_path):
                        # 创建一个稍大一点的图(32x32)，防止某些库对1x1处理异常
                        # 显式保存为 PNG 格式
                        dprint(f"Creating dummy image at {dummy_image_path}...")
                        Image.fromarray(np.zeros((32, 32, 3), dtype=np.uint8)).save(dummy_image_path, format='PNG')
                
                # 3. 🚨 关键同步点：所有 Rank 必须在此等待 Rank 0 写完文件！
                if dist.is_available() and dist.is_initialized():
                    dist.barrier()

                final_image_to_use = None

                # 1. 尝试获取工具生成的图
                raw_clip = clip_images[i]
                
                if raw_clip is not None:
                    if isinstance(raw_clip, list) and len(raw_clip) > 0:
                        # 如果工具返回了多张图的列表，只取第一张！
                        final_image_to_use = raw_clip[0]
                    elif isinstance(raw_clip, str):
                        # 如果是单个路径字符串
                        final_image_to_use = raw_clip
                    # (如果有其他对象类型，视情况处理)

                # 2. 如果没拿到图，或者文件不存在，就用 Dummy
                if final_image_to_use is None or not os.path.exists(final_image_to_use):
                    final_image_to_use = dummy_image_path
                
                if final_image_to_use != dummy_image_path:
                    content.append({"type": "text", "text": "Frames extracted:\n"})
                
                # ✅ 核心：这里永远只 append 一次 image
                content.append({"type": "image", "image": final_image_to_use})
                
                content.append({"type": "text", "text": self.ANSWER_TEMPLATE})

                msgs.append({"role": "user", "content": content})
                turn2_messages.append(msgs)

            turn2_prompts = [
                self.processing_class.apply_chat_template(m, tokenize=False, add_generation_prompt=True)
                for m in turn2_messages
            ]

            try:
                image_inputs2, video_inputs2, video_kwargs2 = process_vision_info(
                    turn2_messages, return_video_kwargs=True
                )
            except Exception as e:
                dprint(f"❌ process_vision_info (Turn 2) Error: {e}")
                raise e

            t2_inputs = self.processing_class(
                text=turn2_prompts,
                images=image_inputs2,
                videos=video_inputs2,
                return_tensors="pt",
                padding=True,
                padding_side="left",
                add_special_tokens=False,
                **video_kwargs2,
            )
            t2_inputs = super()._prepare_inputs(t2_inputs)

            for k, v in list(t2_inputs.items()):
                if torch.is_tensor(v):
                    t2_inputs[k] = v.to(device, non_blocking=True)

            gen2_cfg = copy.deepcopy(self.generation_config)
            gen2_cfg.num_return_sequences = 1

            with unwrap_model_for_generation(model, self.accelerator) as unwrapped:
                t2_ids = unwrapped.generate(**t2_inputs, generation_config=gen2_cfg)

            t2_prompt_len = t2_inputs["input_ids"].size(1)
            t2_comp_ids = t2_ids[:, t2_prompt_len:]
            t2_texts = self.processing_class.batch_decode(t2_comp_ids, skip_special_tokens=True)

            if DO_LOG:
                self._rank0_print(f"\n[step={step}] Turn2 Final Answer")
                for j in range(min(MAX_SHOW, len(t2_texts))):
                    self._rank0_print(f"--- t2[{j}] ---\n{t2_texts[j]}")

            # ============================================================
            # 🚨 [最终修正] Global Padding 对齐 (完美处理 Left+Right Padding)
            # ============================================================
            
            # 1. 保存当前 Rank 的原始长度
            original_seq_len = t2_ids.size(1)
            
            # 2. 计算所有 Rank 中的最大长度
            if dist.is_available() and dist.is_initialized():
                local_max = torch.tensor(original_seq_len, device=device)
                dist.all_reduce(local_max, op=dist.ReduceOp.MAX)
                global_max_len = local_max.item()
            else:
                global_max_len = original_seq_len


            # 3. 对齐 Input IDs
            if original_seq_len < global_max_len:
                pad_len = global_max_len - original_seq_len
                # 右侧填充 pad_token
                padding_ids = torch.full(
                    (t2_ids.size(0), pad_len), 
                    self.processing_class.pad_token_id, 
                    dtype=t2_ids.dtype, 
                    device=device
                )
                t2_ids_padded = torch.cat([t2_ids, padding_ids], dim=1)
            else:
                t2_ids_padded = t2_ids

            # 4. [关键修改] 动态重构 Attention Mask
            # 直接根据 ID 值是否是 Pad Token 来判断。
            # 这样既能把左边的 Pad (Prompt产生的) 标为 0，也能把右边的 Pad (我们加的) 标为 0。
            t2_mask_padded = (t2_ids_padded != self.processing_class.pad_token_id).long()

            # ============================================================
            # Rewards & Loss Calculation
            # ============================================================
            completions = [[{"role": "assistant", "content": t}] for t in t2_texts]
            prompts = [inputs[i // G]["prompt"] for i in range(B * G)]
            reward_kwargs = {}
            for k in inputs[0].keys():
                if k not in ["prompt", "completion"]:
                    reward_kwargs[k] = [inputs[i // G][k] for i in range(B * G)]

            rewards_per_func = torch.zeros((B * G, len(self.reward_funcs)), device=device)
            for i, reward_func in enumerate(self.reward_funcs):
                try:
                    out = reward_func(prompts=prompts, completions=completions, **reward_kwargs)
                    assert len(out) == B*G, (len(out), B*G)
                    rewards_per_func[:, i] = torch.tensor(out, dtype=torch.float32, device=device)
                except Exception as e:
                    rewards_per_func[:, i] = 0.0

            rewards = rewards_per_func.sum(dim=1)

            # tool_bonus_rewards = torch.zeros_like(rewards)
            # BONUS_CORRECT_WITH_TOOL = 0.2

            # PENALTY_NOISE = -0.5  # 新增: 如果用了工具还答错，倒扣分 (惩罚噪声)

            # has_tool_use = len(calls) > 0 and calls[0]['name'] != 'None'
            # is_correct = rewards[idx].item() > 1.0 # 假设 >= 1.0 算答对 (Accuracy=1.0)

            # if has_tool_use:
            #     if is_correct:
            #         # ✅ Case 1: 答对 + 用了工具 -> 给奖励
            #         tool_bonus_rewards[idx] += BONUS_CORRECT_WITH_TOOL
            #     else:
            #         # ❌ Case 2: 答错 + 用了工具 -> 倒扣分
            #         # 说明工具引入了噪声，或者模型瞎用工具
            #         tool_bonus_rewards[idx] += PENALTY_NOISE
                
            # total_rewards = rewards + tool_bonus_rewards

            # rewards = total_rewards
            # 准备 Completion Mask (使用原始未 Pad 的数据)
            is_eos = t2_comp_ids == self.processing_class.eos_token_id
            eos_idx = torch.full((is_eos.size(0),), is_eos.size(1), dtype=torch.long, device=device)
            eos_idx[is_eos.any(dim=1)] = is_eos.int().argmax(dim=1)[is_eos.any(dim=1)]

            seq_idx = torch.arange(is_eos.size(1), device=device).expand(is_eos.size(0), -1)
            completion_mask = (seq_idx <= eos_idx.unsqueeze(1)).int()

            # 准备 Forward Inputs
            fwd_inputs = dict(t2_inputs)
            fwd_inputs.pop("input_ids", None)
            
            # ⚠️ 显式传入我们动态生成的完美 Mask
            fwd_inputs["attention_mask"] = t2_mask_padded 
            fwd_inputs.pop("second_per_grid_ts", None)

            # 显存清理
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

            # ------------------------------------------------------------------
            # 计算 Current Policy LogProbs (使用 Padded Input)
            # ------------------------------------------------------------------
            
            per_token_logps_padded = self._get_per_token_logps(model, t2_ids_padded, **fwd_inputs)
            
            # ✂️ [切片还原]：只取回原始长度部分，丢弃 Padding 的 LogProb
            # _get_per_token_logps 返回形状是 (B, L-1)，所以取 :original_seq_len-1
            per_token_logps = per_token_logps_padded[:, :original_seq_len - 1]
            per_token_logps = per_token_logps[:, t2_prompt_len - 1 :]

            # ------------------------------------------------------------------
            # 计算 Ref Policy LogProbs (使用 Padded Input)
            # ------------------------------------------------------------------
            with torch.inference_mode():
                if self.ref_model is not None:
                    ref_logps_padded = self._get_per_token_logps(self.ref_model, t2_ids_padded, **fwd_inputs)
                else:
                    with self.accelerator.unwrap_model(model).disable_adapter():
                        ref_logps_padded = self._get_per_token_logps(model, t2_ids_padded, **fwd_inputs)
                
                # ✂️ 同样切片还原
                ref_logps = ref_logps_padded[:, :original_seq_len - 1]
                ref_logps = ref_logps[:, t2_prompt_len - 1 :]

            # 计算 KL & Loss
            x = torch.clamp(ref_logps - per_token_logps, -10, 10)
            per_token_kl = torch.exp(x) - x - 1

            mean_r = rewards.view(-1, G).mean(dim=1).repeat_interleave(G)
            std_r = rewards.view(-1, G).std(dim=1).repeat_interleave(G)
            advantages = (rewards - mean_r) / (std_r + 1e-4)

            pg_loss = torch.exp(per_token_logps - per_token_logps.detach()) * advantages.unsqueeze(1)
            per_token_loss = -(pg_loss - self.beta * per_token_kl)

            loss = ((per_token_loss * completion_mask).sum(dim=1) / completion_mask.sum(dim=1)).mean()

            # Log metrics
            if self._metrics is None: self._metrics = defaultdict(list)
            self._metrics["reward"].append(self.accelerator.gather_for_metrics(rewards).mean().item())
            self._metrics["kl"].append(self.accelerator.gather_for_metrics(per_token_kl.mean()).mean().item())
            self._metrics["reward_std"].append(self.accelerator.gather_for_metrics(std_r).mean().item())

            completion_length = self.accelerator.gather_for_metrics(completion_mask.sum(1)).float().mean().item()
            self._metrics["completion_length"].append(completion_length)

            # 3. 各 Reward Function 的得分详情
            reward_per_func = self.accelerator.gather_for_metrics(rewards_per_func).mean(0)
            for i, reward_func in enumerate(self.reward_funcs):
                if isinstance(reward_func, PreTrainedModel):
                    reward_func_name = reward_func.config._name_or_path.split("/")[-1]
                else:
                    reward_func_name = reward_func.__name__
                self._metrics[f"rewards/{reward_func_name}"].append(reward_per_func[i].item())

            # 4. 准确率统计 (基于 Threshold 判断)
            # 假设: Total Reward >= 1.0 算 Correct (例如 Accuracy=1.0)
            gathered_rewards = self.accelerator.gather_for_metrics(rewards)
            num_devices = gathered_rewards.size(0) // self.num_generations 

            # self._metrics["reward_tool_bonus"].append(self.accelerator.gather_for_metrics(tool_bonus_rewards).mean().item())
            return loss
