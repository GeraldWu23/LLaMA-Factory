# RFT 组件 — 打分器 score_candidates.py

把"对话候选 → 还款概率分数"作为 RFT 的 reward 组件。**只是一条独立的 shell 命令**，不嵌入训练进程。

---

## 文件清单

| 文件 | 角色 |
|---|---|
| `score_candidates.py` | 单卡打分 worker。输入 list[dict] JSON，输出每条加上 `score` 字段的 JSON |

---

## 输入 / 输出格式

**输入** `--input_json`：list[dict]，每条至少有 `message` 字段（list[str]，符合 `TextDataset` 约定）。
**输出** `--output_json`：原 list 每条加上 `score: float`（正类概率，0-1）。

---

## 单独跑（最小验证，10 条）

```bash
cd /home/gerald.hu/llama-factory/examples/train_rft

CUDA_VISIBLE_DEVICES=7 python score_candidates.py \
    --model_path  /ldata/share_data/rucui/model/model_qwen3_4B_emb_csdomain_1to10_JuntoSep_eith_conv_rm08_09_20260312_111329/checkpoint-11000 \
    --input_json  /ldata/share_data/rucui/evalset_s1ast_202508_202601_20260608.json \
    --output_json /ldata/share_data/rucui/_rft_demo_scored.json \
    --max_samples 10
```

预期输出最后几行：

```
Scoring: 100%|████████| 10/10 [01:xx<00:00, ...]
[score_candidates] saved to /ldata/share_data/rucui/_rft_demo_scored.json
[score_candidates] sample scores = [0.xx, 0.xx, 0.xx]
```

读一下 output json，确认每条都多了 `score` 字段：

```bash
python -c "import json; d=json.load(open('/ldata/share_data/rucui/_rft_demo_scored.json')); print(len(d), [x['score'] for x in d[:3]])"
```

---

## 全量跑（实际 RFT 用）

去掉 `--max_samples`，把 `--input_json` 换成 vLLM 采样产出的候选集即可：

```bash
CUDA_VISIBLE_DEVICES=7 python score_candidates.py \
    --model_path  /ldata/share_data/rucui/model/.../checkpoint-11000 \
    --input_json  /path/to/round1_candidates.json \
    --output_json /path/to/round1_scored.json
```

---

## 在 RFT pipeline 中的位置

```bash
# round 1
CUDA_VISIBLE_DEVICES=0,1,2,3 python <vllm_infer.py> ...      # A. 采样 N=8
CUDA_VISIBLE_DEVICES=7       python score_candidates.py ...  # B. 打分（本脚本）
                             python select_top1.py ...       # C. 选 top-1（待写）
CUDA_VISIBLE_DEVICES=4,5,6,7 llamafactory-cli train ...      # D. 训练 round 1 LoRA
# round 2 重复 A→D
```

三者是**同级 shell 命令，靠文件系统传数据**，不存在 Python 互相调用。
