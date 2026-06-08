#!/usr/bin/env python
# sample_candidates.py
#
# RFT 采样组件：用 vLLM + LoRA 给一批 prompt 各 sample N 条候选回复。
# 与 score_candidates.py 配对使用。
#
# 输入: --input_json   list[dict]，每条至少含 'input' 字段（SFT 训练时的 prompt 格式）
# 输出: --output_json  扁平化的 list，每条 = {原字段..., 'sample_idx': i, 'response': '...'}
#                      一个 input 对应 N 条输出，可直接喂 score_candidates.py
#
# 单卡示例:
# CUDA_VISIBLE_DEVICES=0 python sample_candidates.py \
#     --base_model /ldata/Qwen/Qwen3-32B \
#     --lora_path  /ldata/share_data/cuishou_conv/models/lora_models/qwen3_32b_gen_cuishou_long3k_0605/checkpoint-11200 \
#     --input_json /ldata/share_data/rucui/trainset_rft_s1ast_202508_202601_20260608.json \
#     --output_json /ldata/share_data/rucui/_rft_demo_candidates.json \
#     --max_samples 2 \
#     --n 8 \
#     --tensor_parallel 1

import argparse
import json
import os
import sys

from tqdm import tqdm
from vllm import LLM, SamplingParams
from vllm.lora.request import LoRARequest


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--base_model", default="/ldata/Qwen/Qwen3-32B",
                   help="vLLM 加载的基础模型路径")
    p.add_argument("--lora_path", default=None,
                   help="LoRA adapter 路径；不传则用 base 直接采样")
    p.add_argument("--input_json", required=True,
                   help="list[dict] JSON，每条含 'input' 字段")
    p.add_argument("--output_json", required=True)
    p.add_argument("--max_samples", type=int, default=None,
                   help="只采前 N 条 prompt；demo 用，不传则全量")
    p.add_argument("--n", type=int, default=8, help="每个 prompt 采 N 条候选")
    p.add_argument("--temperature", type=float, default=0.9)
    p.add_argument("--top_p", type=float, default=0.95)
    p.add_argument("--max_tokens", type=int, default=512,
                   help="单条候选最大新 token 数")
    p.add_argument("--max_model_len", type=int, default=5000,
                   help="vLLM context 长度上限")
    p.add_argument("--tensor_parallel", type=int, default=1)
    p.add_argument("--seed", type=int, default=42)
    return p.parse_args()


def main():
    args = parse_args()

    print(f"[sample_candidates] CUDA_VISIBLE_DEVICES={os.environ.get('CUDA_VISIBLE_DEVICES')}")
    print(f"[sample_candidates] base_model={args.base_model}")
    print(f"[sample_candidates] lora_path={args.lora_path}")
    print(f"[sample_candidates] input_json={args.input_json}")
    print(f"[sample_candidates] output_json={args.output_json}")
    print(f"[sample_candidates] n={args.n}, temperature={args.temperature}, top_p={args.top_p}")

    # ---- load prompts ----
    with open(args.input_json) as f:
        raw_data = json.load(f)
    if args.max_samples is not None:
        raw_data = raw_data[: args.max_samples]
    print(f"[sample_candidates] total prompts = {len(raw_data)}")

    # 兼容两种 input 字段：'input'（SFT 风格）或没有时拼 message
    prompts = []
    for d in raw_data:
        if "input" in d:
            prompts.append(d["input"])
        elif "message" in d:
            # message 是 list[str]，拼成对话历史；这里只是兜底，正式应该用 SFT 同款 prompt
            prompts.append("\n\n".join(m for m in d["message"] if m))
        else:
            raise ValueError(f"sample missing 'input' or 'message': keys={list(d.keys())}")

    # ---- vLLM init ----
    llm_kwargs = dict(
        model=args.base_model,
        trust_remote_code=True,
        dtype="bfloat16",
        max_model_len=args.max_model_len,
        tensor_parallel_size=args.tensor_parallel,
        disable_log_stats=True,
    )
    if args.lora_path:
        llm_kwargs["enable_lora"] = True
        llm_kwargs["max_lora_rank"] = 64  # 兼容到 r=64；用 r=16/32 也 OK

    llm = LLM(**llm_kwargs)

    sampling_params = SamplingParams(
        n=args.n,
        temperature=args.temperature,
        top_p=args.top_p,
        max_tokens=args.max_tokens,
        seed=args.seed,
    )

    lora_request = (
        LoRARequest("rft_lora", 1, args.lora_path) if args.lora_path else None
    )

    # ---- generate ----
    print(f"[sample_candidates] generating {args.n} candidates per prompt...")
    results = llm.generate(prompts, sampling_params, lora_request=lora_request)

    # ---- flatten ----
    flat = []
    for src, res in zip(raw_data, results):
        for i, out in enumerate(res.outputs):
            row = dict(src)  # 保留原字段（account_id / message / overdue_days...）
            row["sample_idx"] = i
            row["response"] = out.text
            flat.append(row)

    os.makedirs(os.path.dirname(args.output_json) or ".", exist_ok=True)
    with open(args.output_json, "w") as f:
        json.dump(flat, f, ensure_ascii=False, indent=1)
    print(f"[sample_candidates] saved {len(flat)} rows to {args.output_json}")
    print(f"[sample_candidates] (= {len(raw_data)} prompts × {args.n} samples)")

    # 打印前 1 条的全部 N 个候选，做多样性 sanity check
    print("\n[sample_candidates] sanity preview - prompt 0, all N candidates:")
    for i, out in enumerate(results[0].outputs):
        preview = out.text[:120].replace("\n", " ")
        print(f"  [{i}] {preview!r}")


if __name__ == "__main__":
    main()
