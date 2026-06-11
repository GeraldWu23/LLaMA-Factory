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
    # 完全对齐 notebook cell 4 的 completion(no_think=True) 调用参数
    p.add_argument("--temperature", type=float, default=0.2)
    p.add_argument("--top_p", type=float, default=0.98)
    p.add_argument("--repetition_penalty", type=float, default=1.2)
    p.add_argument("--presence_penalty", type=float, default=0.4)
    p.add_argument("--max_tokens", type=int, default=1024)
    p.add_argument("--max_model_len", type=int, default=5000,
                   help="vLLM context 长度上限")
    p.add_argument("--tensor_parallel", type=int, default=1)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--no_think", action="store_true", default=True,
                   help="注入 </no_think> 标记，关闭 reasoning（与 notebook 对齐，默认 True）")
    p.add_argument("--chunk_size", type=int, default=500,
                   help="分批跑：每 N 个 prompt 写一次 output 文件，方便中途查看")
    return p.parse_args()


def main():
    args = parse_args()

    print(f"[sample_candidates] CUDA_VISIBLE_DEVICES={os.environ.get('CUDA_VISIBLE_DEVICES')}")
    print(f"[sample_candidates] base_model={args.base_model}")
    print(f"[sample_candidates] lora_path={args.lora_path}")
    print(f"[sample_candidates] input_json={args.input_json}")
    print(f"[sample_candidates] output_json={args.output_json}")
    print(f"[sample_candidates] n={args.n}, temperature={args.temperature}, top_p={args.top_p}, "
          f"rep_penalty={args.repetition_penalty}, presence_penalty={args.presence_penalty}, max_tokens={args.max_tokens}")

    # ---- load prompts ----
    with open(args.input_json) as f:
        raw_data = json.load(f)
    if args.max_samples is not None:
        raw_data = raw_data[: args.max_samples]
    print(f"[sample_candidates] total prompts = {len(raw_data)}")

    # 完全对齐 notebook cell 4 的 completion(no_think=True) prompt 拼接：
    #   <|im_start|>system\nYou are a helpful assistant.<|im_end|>\n
    #   <|im_start|>user\n{input}\n</no_think><|im_end|>\n   (no_think=True 时)
    #   <|im_start|>assistant\n   ← 模型从这里开始生成
    def wrap_chat(raw_input: str) -> str:
        # 注入自定义 no_think 标记（此标记属于 content 内部）
        user_inner_content = f"{raw_input}\n</no_think>" if args.no_think else raw_input
        return (
            "<|im_start|>system\n"
            "You are a helpful assistant.<|im_end|>\n"
            "<|im_start|>user\n"
            f"{user_inner_content}<|im_end|>\n"
            "<|im_start|>assistant\n"
        )

    prompts = []
    for d in raw_data:
        if "input" in d:
            base = d["input"]
        elif "message" in d:
            base = "\n\n".join(m for m in d["message"] if m)
        else:
            raise ValueError(f"sample missing 'input' or 'message': keys={list(d.keys())}")
        prompts.append(wrap_chat(base))

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

    # 套了 qwen3_nothink chat template 后，<|im_end|> 是天然边界——
    # 模型训练时学到的就是"客服 turn 完了出 <|im_end|>"，stop 用它就够。
    sampling_params = SamplingParams(
        n=args.n,
        temperature=args.temperature,
        top_p=args.top_p,
        repetition_penalty=args.repetition_penalty,
        presence_penalty=args.presence_penalty,
        max_tokens=args.max_tokens,
        stop=["<|im_end|>"],
        seed=args.seed,
    )

    lora_request = (
        LoRARequest("rft_lora", 1, args.lora_path) if args.lora_path else None
    )

    # ---- generate by chunks（边跑边落盘，方便中途查看 output_json）----
    os.makedirs(os.path.dirname(args.output_json) or ".", exist_ok=True)
    flat = []
    n_total = len(prompts)
    chunk = max(1, args.chunk_size)
    print(f"[sample_candidates] generating with chunk_size={chunk}, total chunks={(n_total + chunk - 1) // chunk}")

    first_chunk_preview = None
    for start in range(0, n_total, chunk):
        end = min(start + chunk, n_total)
        sub_prompts = prompts[start:end]
        sub_raw = raw_data[start:end]
        print(f"[sample_candidates] chunk {start}:{end} / {n_total} ...")
        sub_results = llm.generate(sub_prompts, sampling_params, lora_request=lora_request)

        for src, res in zip(sub_raw, sub_results):
            for i, out in enumerate(res.outputs):
                row = dict(src)
                row["sample_idx"] = i
                row["response"] = out.text
                flat.append(row)

        if first_chunk_preview is None and sub_results:
            first_chunk_preview = sub_results[0]

        # 原子写：先写 .tmp 再 rename
        tmp_path = args.output_json + ".tmp"
        with open(tmp_path, "w") as f:
            json.dump(flat, f, ensure_ascii=False, indent=1)
        os.replace(tmp_path, args.output_json)
        print(f"[sample_candidates]   wrote {len(flat)} rows -> {args.output_json}")

    print(f"[sample_candidates] DONE: saved {len(flat)} rows to {args.output_json}")
    print(f"[sample_candidates] (= {n_total} prompts × {args.n} samples)")

    # 打印第 1 条 prompt 全部 N 个候选作为 sanity check
    if first_chunk_preview is not None:
        print("\n[sample_candidates] sanity preview - prompt 0, all N candidates:")
        for i, out in enumerate(first_chunk_preview.outputs):
            preview = out.text[:120].replace("\n", " ")
            print(f"  [{i}] {preview!r}")


if __name__ == "__main__":
    main()



