# LLaMA-Factory — 项目使用经验（针对催收话术 RFT pipeline）

> 这份是本地仓库给 Claude 看的笔记，记录 `examples/train_rft/` 这一套 RFT pipeline 实际跑出来的关键经验。
> 完整的实现细节、reward model 行为洞察、跨 session 接力，在用户的 memory 里：`[[project-rft-pipeline-v1]]` + `[[project-collection-script-assist]]`。

## 这套 pipeline 在做什么

`examples/train_rft/` 是给催收话术生成模型做 **RFT（Rejection sampling Fine-Tuning）** 的一套脚本：

```
M0 SFT (Qwen3-32B QLoRA)
  → vLLM 采样 N 条候选 (sample_candidates.py)
  → reward model 打分 (score_candidates.py)
  → 按 account_id 选 top-1 (select_top1.py)
  → 训 round1 LoRA (adapter_name_or_path=m0 接力训练)
```

3.4 机器上跑：训练 + 打分用 `llm` env，vLLM 采样用 `qwen3coder` env。

## 跑下来踩过的坑（按重要度排）

### ★★★ Reward model 输出语义反了 _(2026-06-11)_

**问题**：reward model（`/ldata/share_data/rucui/model/.../checkpoint-11000`）`probs[:, 1]` 是**逾期概率**（越高越差），不是"还款概率"。第一次跑 RFT 时按"越高越好"取 top-1，把模型训反了 —— round1 输出风格变差（更敷衍 + 偶尔编造性别）。

**修复**：`score_candidates.py` 和 `eval_score.py` 里 score 都用 `1 - probs[:, 1]`，让"越高越好"的下游逻辑（`select_top1` 取最大）保持不变。

**Why**：避免改 `select_top1` 改全链路，反转一次就完事。

### ★★★ 训练-推理 prompt 拼接必须严格一致 _(2026-06-10)_

**问题**：score_candidates 拼 message 是一种风格，eval_score 拼 message 是另一种风格 → 训练目标 ≠ 评估目标，训了白训。

**正确做法**（已固化在脚本里）：
- 去掉 `### 对话片段：` 之前的指令段
- context 内 `\n\n` → 单空格
- response 内 `\n\n` → 单空格
- context 与 response 之间也用单空格连接
- 最终 `message = [context + " " + response]` 单元素列表

### ★★★ wrap_chat 不要带 `<think></think>` _(2026-06-10)_

**问题**：照搬 LLaMA-Factory `qwen3_nothink` template 给 prompt 末尾加 `<think>\n\n</think>\n\n` 后，模型输出变成"instruct 答题风格"（带引号/括号说明），不像训练数据里的"客服：xxx"自然语气。

**正确**：`wrap_chat` 末尾停在 `<|im_start|>assistant\n`，**不**加 `<think></think>`。`no_think` 通过在 user 末尾追加 `\n</no_think>` 实现（content 内部标记）。

### ★★★ select_top1 必须做质量过滤（最重要的实战经验）_(2026-06-12)_

**问题**：reward model 在 OOD 输入上行为不稳定，会给一些垃圾候选打高分（0.6-0.75）。这些候选被 `select_top1` 选中后污染训练集 → round1 学坏（输出复读、过短、带 think 占位）。

**实测污染情况**（round1 sample 阶段 12500 组）：

| 污染类型 | 比例 | 例子 |
|---|---|---|
| 复读机长 target | 12% target > 200 字 | `"客服：啊主管主管主管...×500"` (2046 字)、`"...反正反正反正..."` |
| `<think>` 占位残留 | 1% | `"客服：xxx<think>\n\n</think>\n\nxxx"` |
| 极短无信息 | 1.4% | `"客服：喂"`、`"客服：啊"` (score 0.75) |

**修复**：`select_top1.py` 加 4 重过滤，组内按 score 降序遍历，找第一个通过过滤的取为 top-1（不是丢整组）：

```python
def is_valid_target(t, min_len=10, max_len=400):
    if len(t) < min_len: return False, "too_short"
    if len(t) > max_len: return False, "too_long"          # 防复读机长尾
    if '<think>' in t: return False, "has_think"           # 防 m0 残留先验扩散
    if re.search(r'(.{2,8})\1{5,}', t): return False, "repetition"  # 2-8 字 ngram 连续重复 >5 次
    return True, ""
```

**实测效果**：
- 12500 组 → 12496 组（仅 4 组 8 条全废，几乎无损）
- kept score median: 0.55 → **0.81**（留下的 top-1 reward 普遍更高）
- 单候选过滤数：too_short 283 / has_think 266 / too_long 356 / repetition 2 —— 全是 reward model 抽风给的高分垃圾

**Why**：reward model 学的是"对话最终是否回收"的概率，遇到训练集没见过的复读 / 极短 OOD 回复时 softmax 行为随机，可能恰好打高分。**不过滤就是用 reward model 的脏分数当训练标签**。

**优先级**：这是 RFT 链路里**最大的单点提升**。同一份 scored.json 重跑过滤后的 select_top1 + 重训 round1，比任何调参 / 改 prompt / 改采样温度都有效。新 RFT 项目 day 1 就要带这个过滤，不要等训出问题才补。

### ★★ M0 SFT 会让模型残留 `<think>` 占位先验 _(2026-06-11)_

**现象**：m0 训完后，无论 prompt 拼接是否带 `<think></think>`，模型有 ~30% 概率会自己吐出 `<think>\n\n</think>\n\n` 开头。

**原因**：LLaMA-Factory 训练时 assistant 段自带 think 占位，模型学到了"assistant 应该这样开头"。

**两种应对**：
1. **eval_score 端兜底**（已实施）：检测 response 是否以 `<think>\n\n</think>\n\n` 开头，是则替换成 `"客服："` 再走拼接打分。不影响 m0/round1 相对比较
2. **从源头消除**：m0 重训时改 template = `qwen`（不是 `qwen3_nothink`），让 assistant 段不带 think。代价是要重训

### ★★ vLLM 同时挂 LoRA 用 `--lora-modules` _(2026-06-10)_

```bash
vllm serve <BASE> --enable-lora --max-lora-rank 64 \
  --lora-modules m0=<m0_path> round1=<round1_path> \
  --served-model-name <BASE_ALIAS> ...
```

调用时 `model_name='m0'` / `'round1'` 切换；裸 base 用 `--served-model-name` 那个值。

**两个 LoRA 同时挂 ≠ 叠加生效**：每个请求只套一个 LoRA。round1 是从 m0 接力训出来的（B 模式），权重已经包含 m0 的全部信息，所以**只挂 round1 一个 = m0+round1 然后 model_name='round1'**，效果完全等价。同时挂只是为了方便切换对比。

验证 round1 是否 m0 接力训的：`diff` 两个 adapter_config.json 的 r/lora_alpha/target_modules 是否一致。

### ★ Reward model 是 conversation-level 信号，不是 turn-level 偏好 _(2026-06-11)_

抽看 sampling 阶段同 prompt 的 8 条候选打分，发现：

| 上下文位置 | 组内 score range（max-min）| reward 行为 |
|---|---|---|
| 开场 / 解释业务 | ~0.2 | 偏好长且解释性的回复 ✅ |
| 中段过渡 / 即将挂电话 | ~0.02-0.04 | 偏好"短句礼貌结束" |
| 客户已表态还款 | ~0.002 | 整组打 ~0.03（=逾期概率高），组内排序无意义 |

reward 学的是**"这通对话最终是否回收"**，不是"这条回复好不好"。所以"无关紧要的 turn"上 reward 无能为力，强行训这些数据反而把模型带偏。

**下一步要做但还没做**：`select_top1` 后加过滤 —— 丢掉组内 `max_score - min_score < 0.05` 的所有组。预计数据量 12500 → 2500-5000，但信号纯。

### ★ Pipeline 中默认参数对齐 notebook 调用，不是 LLaMA-Factory 训练默认 _(2026-06-10)_

整套对齐 `rc_conversations/tests/test_call_server/test_call_vllm_on_cuishou_conv.ipynb` cell 4 的 `completion(no_think=True)` 调用：

```
temperature=0.2, top_p=0.98, repetition_penalty=1.2,
presence_penalty=0.4, max_tokens=1024, no_think=True, stop=["<|im_end|>"]
```

采样阶段（要多样性）可以拉到 `temperature=0.8`，其他不动。

## 关键文件用途

| 文件 | 接口 | 注意点 |
|---|---|---|
| `sample_candidates.py` | argparse | wrap_chat 不带 `<think></think>`；`no_think` 默认 True |
| `score_candidates.py` | argparse + accelerate | `1 - probs[:, 1]` 反转；message 拼接同 eval_score |
| `select_top1.py` | argparse | 按 `--group_key account_id` 分组，取 score 最大 |
| `eval_generate.py` | argparse + vLLM | n=1，参数对齐 notebook |
| `eval_score.py` | **文件顶部 JOBS 常量** + accelerate | 一次跑多个 model 自动对比 delta；`<think></think>` 占位替换 `"客服："` |

## 远程文件读写

3.4 机器（10.131.3.4:6655）的所有训练产物 / 脚本，用 JupyterHub HTTP API 直接读写：

```bash
curl -sL -H "Authorization: token <TOKEN>" "http://10.131.3.4:6655/user/gerald.hu/api/contents/<PATH>"
```

token 在用户 memory 的 `[[reference_servers]]` 里。`/ldata/` 目录在 user home 外，API 够不到，需要让用户跑命令把内容贴回来。
