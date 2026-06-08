# RFT (Rejection sampling Fine-Tuning) Pipeline

催收对话话术辅助生成项目的 RFT 流程：用 Qwen3-4B + 分类头作 reward model（"对话→次日还款"概率），让 SFT 模型自我提升。

参考论文：[Scaling Relationship on Learning Mathematical Reasoning with LLMs (arXiv:2308.01825)](https://arxiv.org/abs/2308.01825)

---

## 算法

```
M0 = SFT(base, sft_set)
for round in 1..N:
    candidates = M_{round-1}.sample(rft_prompts, n=N_SAMPLE)   # vLLM
    scored     = reward_model.score(candidates)                # Qwen3-4B + cls head
    top1       = select_top1(scored, group_by=account_id)
    M_round    = SFT(M_{round-1} adapter, top1)                # LLaMA-Factory 续训
```

---

## 文件清单

| 文件 | 作用 |
|---|---|
| `sample_candidates.py` | vLLM 采样：每个 prompt 出 N 条候选回复（输出 `response` 字段） |
| `score_candidates.py`  | 打分：reward model 给每条候选加 `score` 字段（accelerate 多卡） |
| `select_top1.py`       | 按 group_key（默认 account_id）分组，留 score 最高那条；输出 LLaMA-Factory 训练格式 |
| `qwen3_qlora_mini_sft.yaml` | mini 验证 SFT 配置 |
| `qwen3_qlora_mini_rft.yaml` | mini 验证 RFT 续训模板（占位字段由 shell 命令行注入） |
| `qwen3_qlora_full_sft.yaml` | 正式 SFT 配置（5w 数据、5 epoch、load_best_model_at_end） |
| `qwen3_qlora_full_rft.yaml` | 正式 RFT 续训模板 |
| `run_mini_rft.sh` | 端到端 pipeline（mini，100 条 × 2 轮，验证脚本是否跑通） |
| `run_full_rft.sh` | 端到端 pipeline（full，5w SFT × 3 轮 RFT） |

---

## 环境分工

| 用途 | env | 路径（3.4 机器） |
|---|---|---|
| 训练 + 打分 | `llm` | `/home/gerald.hu/.conda/envs/llm/bin/python` |
| vLLM 采样 | `qwen3coder` | `/home/gerald.hu/.conda/envs/qwen3coder/bin/python` |

shell 脚本里写死路径，不依赖 `conda activate`。

---

## 关键约定

### 数据格式

**SFT / RFT prompt 集**（list[dict]）：每条至少含
- `account_id`：分组键（一个账户一个 prompt）
- `input`：完整 prompt 文本，以 `### 对话片段：` 开头并以"客户：xxx"结尾
- `target`：客服下一句（SFT 训练用；RFT prompt 集可有可无，会被采样产物覆盖）

由 `rc_conversations/demo/build_conv_s1ast_202502_202602.ipynb` 生成。

### 候选打分时的字段适配

`score_candidates.py` 兼容两种输入：
- 已有 `message` 字段（直接 SFT 数据）→ 直接打分
- 仅有 `input` + `response`（采样候选）→ 自动拼成 `message = [<对话历史> + <候选客服 turn>]`

### LoRA 续训机制

LLaMA-Factory 的 `adapter_name_or_path=<上一轮 LoRA>` 是 **resume 模式**：
- 加载 base 模型 + 上一轮 LoRA 权重作为可训练起点
- 继续 forward + backward 更新这个 LoRA
- 保存的 round R LoRA = "M_{R-1} 起点 + R 轮更新量"
- 推理时 **只需加载最后一轮 LoRA 即可**，不需要 stack 所有历史

详见源码 `src/llamafactory/model/adapter.py:150-189`。

---

## Mini 验证（100 条 × 2 轮，端到端跑通）

```bash
bash /home/gerald.hu/llama-factory/examples/train_rft/run_mini_rft.sh
```

产物：
```
/ldata/share_data/cuishou_conv/tmp/rft_mini/$RUN_TAG/round{1,2}/{candidates,scored,top1}.json
/ldata/share_data/cuishou_conv/models/lora_models/qwen3_32b_rft_$RUN_TAG/{m0_sft,round1,round2}/checkpoint-*/
```

注意：mini 数据 100 条 + 1 epoch + bs=1×ga=8 ≈ **1-2 个 optimizer step**，权重几乎没动。验证目的是 **链路通**，不是模型质量。

---

## 正式跑（5w SFT × 3 轮 RFT）

```bash
bash /home/gerald.hu/llama-factory/examples/train_rft/run_full_rft.sh
```

参数：
- M0 SFT：50000 条，5 epoch，lr=1e-5，`load_best_model_at_end=true`
- 每轮 RFT：12500 prompt × n=8 候选，2 epoch，lr=5e-6
- 总耗时估计：15-20 小时

产物结构：
```
/ldata/share_data/cuishou_conv/models/lora_models/qwen3_32b_rft_<RUN_TAG>/
  ├── m0_sft/         (M0 SFT)
  ├── round1/         (Round 1 RFT)
  ├── round2/         (Round 2 RFT)
  └── round3/         (Round 3 RFT，最终模型)
```

---

## wandb 配置

每次 bash 跑一次 = 一个 group，组内 4 个独立 run（M0_sft / round1 / round2 / round3）。

```bash
WANDB_PROJECT=rc_conv
WANDB_RUN_GROUP=$RUN_TAG    # 默认时间戳，可在 shell 顶部改成有意义的名字
```

本地 wandb 模式建议：

```bash
export WANDB_MODE=offline
```

---

## 单脚本调试

### 单独打分（验证 reward model）

```bash
CUDA_VISIBLE_DEVICES=0,1,2,3 /home/gerald.hu/.conda/envs/llm/bin/accelerate launch \
    --num_processes 4 --main_process_port 29555 \
    /home/gerald.hu/llama-factory/examples/train_rft/score_candidates.py \
    --model_path /ldata/share_data/rucui/model/<reward_model>/checkpoint-XXXXX \
    --input_json /path/to/data.json \
    --output_json /path/to/scored.json \
    --max_samples 10
```

### 单独采样（验证 vLLM 多样性）

```bash
CUDA_VISIBLE_DEVICES=0,1,2,3 /home/gerald.hu/.conda/envs/qwen3coder/bin/python \
    /home/gerald.hu/llama-factory/examples/train_rft/sample_candidates.py \
    --base_model /ldata/Qwen/Qwen3-32B \
    --lora_path /path/to/lora \
    --input_json /path/to/prompts.json \
    --output_json /tmp/candidates.json \
    --max_samples 2 --n 8 --tensor_parallel 4
```

### 单独选 top-1

```bash
python select_top1.py --scored_json scored.json --output_json top1.json --group_key account_id
```

---

## 几个坑

1. **数据集要预先注册到 `data/dataset_info.json`**：mini 和 full 的 sft 数据已注册；RFT 每轮的 top1 数据由 shell 自动注册（key 名 `mini_rft_round{R}_top1` / `full_rft_<RUN_TAG>_round{R}_top1`）。
2. **save_strategy 要设 epoch**：mini 数据 step 数太少，按 step 保存会 0 个 ckpt。
3. **`latest_checkpoint` 函数**：每轮训练完后从 `output_dir/checkpoint-*` 找最新一个含 `adapter_config.json` 的子目录，作为下一轮 `adapter_name_or_path`。LLaMA-Factory 默认把 LoRA 文件保存在 `checkpoint-XX/` 子目录里，`output_dir` 根目录的 LoRA 是训练完的 best 副本。
4. **vLLM TP 与 GPU 数对齐**：`--tensor_parallel N` 必须与 `CUDA_VISIBLE_DEVICES` 里 GPU 数一致。
5. **本地 wandb**：跑前 `export WANDB_MODE=offline`，否则会反复尝试连云端报错（不影响训练但刷红字）。
