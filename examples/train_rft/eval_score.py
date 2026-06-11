#!/usr/bin/env python
# eval_score.py
#
# 用 reward model 给 eval_generate.py 产出的 response 打分；支持多个模型批量打分 + 自动对比。
#
# 用法（多卡 accelerate）：
#   CUDA_VISIBLE_DEVICES=0,1,2,3 /home/gerald.hu/.conda/envs/llm/bin/accelerate launch \
#       --num_processes 4 --main_process_port 29555 \
#       /home/gerald.hu/llama-factory/examples/train_rft/eval_score.py
#
# 输入: list[dict]，每条至少含 'input' 和 'response' 字段
# 输出: 每个 job 一个纯分数列表 json（与 input 顺序一一对应）
#       跑完自动打印分布 + delta vs baseline（JOBS[0] 为 baseline）
#
# 注意：reward model 输出 probs[:,1] 是 "逾期概率"，越高越差；
# 这里取 1 - probs[:,1] 转成 "还款概率"，所以 score 越高越好。

import json
import os
import statistics
import sys


# =========================================================================
# 配置区
# =========================================================================
REWARD_MODEL = "/ldata/share_data/rucui/model/model_qwen3_4B_emb_csdomain_1to10_JuntoSep_eith_conv_rm08_09_20260312_111329/checkpoint-11000"

# JOBS: (display_name, response_json, output_scores_json)
# 第 1 个为 baseline，后续 job 会自动算 delta vs 它
JOBS = [
    ("m0",
     "/ldata/share_data/cuishou_conv/eval/m0_responses.json",
     "/ldata/share_data/cuishou_conv/eval/m0_scores.json"),
    ("round1",
     "/ldata/share_data/cuishou_conv/eval/round1_responses.json",
     "/ldata/share_data/cuishou_conv/eval/round1_scores.json"),
    ("round1_e3",
     "/ldata/share_data/cuishou_conv/eval/round1_e3_responses.json",
     "/ldata/share_data/cuishou_conv/eval/round1_e3_scores.json"),
]


# =========================================================================
def run_one(name, input_json, output_path, accelerator):
    import torch
    from torch.utils.data import DataLoader
    from tqdm import tqdm
    from transformers import AutoTokenizer

    RC_CONV_ROOT = "/home/gerald.hu/rc_conversations"
    if RC_CONV_ROOT not in sys.path:
        sys.path.insert(0, RC_CONV_ROOT)
    from rc_conversations.app.models.qwen_emb_classifier import Qwen3EmbeddingModel
    from rc_conversations.app.utils.dataset_utils import DatasetPreprocessor, TextDataset
    from rc_conversations.app.utils.utils import load_json
    from rc_conversations.conf.conf import MAX_LENGTH

    if accelerator.is_main_process:
        print(f"\n[eval_score] === JOB: {name} ===")
        print(f"[eval_score] input  = {input_json}")
        print(f"[eval_score] output = {output_path}")
    accelerator.wait_for_everyone()

    raw = load_json(input_json)
    SYS_MARKER = "### 对话片段："
    THINK_PREFIX = "<think>\n\n</think>\n\n"
    for d in raw:
        d.setdefault("label", 0)
        inp = d.get("input", "")
        resp = d.get("response", "")
        # 如果 response 以 think 占位开头，把占位替换成 "客服：" 再走拼接
        if resp.startswith(THINK_PREFIX):
            resp = "客服：" + resp[len(THINK_PREFIX):]
        pos = inp.find(SYS_MARKER)
        context = inp[pos + len(SYS_MARKER):] if pos >= 0 else inp
        context = context.replace("\n\n", " ").strip()
        resp = resp.replace("\n\n", " ").strip()
        d["message"] = [context + " " + resp]

    raw = DatasetPreprocessor.preprocess_with_maxlen(raw, max_length=MAX_LENGTH)
    dataset = TextDataset(raw)
    if accelerator.is_main_process:
        print(f"[eval_score] total samples = {len(dataset)}")

    model = Qwen3EmbeddingModel.load_peft_model(REWARD_MODEL)
    model.eval()
    model.to(dtype=torch.float32)

    tokenizer = AutoTokenizer.from_pretrained(
        model.base_model.model.config.pretrained_model_name_or_path,
        padding_side="left", use_fast=True,
    )
    task_description = model.base_model.model.config.task_description

    dataloader = DataLoader(
        dataset, batch_size=1, shuffle=False,
        num_workers=4, pin_memory=True,
        collate_fn=lambda batch: Qwen3EmbeddingModel.collate_fn_tokenize(
            batch, tokenizer=tokenizer, task_description=task_description),
    )
    model, dataloader = accelerator.prepare(model, dataloader)

    all_scores = []
    with torch.no_grad():
        with accelerator.autocast():
            for batch in tqdm(dataloader, desc=f"Scoring[{name}]",
                              disable=not accelerator.is_main_process):
                logits = model(input_ids=batch["input_ids"],
                               attention_mask=batch["attention_mask"])
                probs = torch.softmax(logits, dim=-1)
                # reward model 输出 probs[:,1] 是逾期概率（越高越差），取 1- 转成还款概率
                scores = 1 - probs[:, 1]
                all_scores.append(accelerator.gather_for_metrics(scores))

    if accelerator.is_main_process:
        all_scores = torch.cat(all_scores).cpu().tolist()
        assert len(all_scores) == len(raw), f"{len(all_scores)} vs {len(raw)}"
        os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
        with open(output_path, "w") as f:
            json.dump(all_scores, f)
        print(f"[eval_score] saved {len(all_scores)} scores to {output_path}")
        return all_scores
    return None


def _quant(values, q):
    s = sorted(values)
    return s[int(q * (len(s) - 1))]


def print_compare(results):
    """results: list of (name, scores_list)，第 1 个为 baseline"""
    print(f"\n{'='*80}\n打分对比\n{'='*80}")
    print(f"{'name':<14} {'n':>6} {'min':>8} {'p25':>8} {'median':>8} {'mean':>8} {'p75':>8} {'max':>8}")
    print("-" * 78)
    stats = []
    for name, scores in results:
        if not scores:
            continue
        row = (name, len(scores), min(scores), _quant(scores, 0.25),
               statistics.median(scores), statistics.mean(scores),
               _quant(scores, 0.75), max(scores))
        stats.append(row)
        print(f"{row[0]:<14} {row[1]:>6} {row[2]:>8.4f} {row[3]:>8.4f} {row[4]:>8.4f} "
              f"{row[5]:>8.4f} {row[6]:>8.4f} {row[7]:>8.4f}")

    if len(stats) >= 2:
        base_name, _, _, _, base_median, base_mean, _, _ = stats[0]
        print(f"\n--- delta vs {base_name} ---")
        for name, _, _, _, med, mean, _, _ in stats[1:]:
            print(f"  {name:<14} mean Δ = {mean - base_mean:+.4f}, median Δ = {med - base_median:+.4f}")


def main():
    from accelerate import Accelerator
    accelerator = Accelerator()
    if accelerator.is_main_process:
        print(f"[eval_score] num_processes = {accelerator.num_processes}")
        print(f"[eval_score] reward_model  = {REWARD_MODEL}")
        print(f"[eval_score] jobs:")
        for name, ip, op in JOBS:
            print(f"  - {name}: {ip} -> {op}")
    accelerator.wait_for_everyone()

    results = []
    for name, ip, op in JOBS:
        s = run_one(name, ip, op, accelerator)
        if accelerator.is_main_process:
            results.append((name, s))

    if accelerator.is_main_process:
        print_compare(results)


if __name__ == "__main__":
    main()
