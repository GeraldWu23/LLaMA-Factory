#!/usr/bin/env python
# eval_generate.py
#
# 加载 base + LoRA，对 eval 集所有 prompt 各做 1 次生成。
# 完全对齐 notebook test_call_vllm_on_cuishou_conv.ipynb cell 4 的 completion(no_think=True) 调用：
#   - qwen3_nothink chat template wrap
#   - user 末尾追加 \n</no_think>
#   - temperature=0.2, top_p=0.98, repetition_penalty=1.2, presence_penalty=0.4
#   - stop=["<|im_end|>"], max_tokens=1024
#
# 用法:
#   conda activate qwen3coder
#   CUDA_VISIBLE_DEVICES=0,1,2,3 python eval_generate.py \
#       --base_model /ldata/Qwen/Qwen3-32B \
#       --lora_path  <LORA_PATH> \
#       --input_json /ldata/share_data/rucui/evalset_s1ast_202508_202601_20260608.json \
#       --output_json /ldata/share_data/cuishou_conv/eval/<MODEL_NAME>_responses.json \
#       --tensor_parallel 4
#
# 命名规则：每个模型只生成一个版本，文件名 = <MODEL_NAME>_responses.json (m0 / round1 / ...)，直接覆盖。
#
# 输出: 每条样本加 'response' 字段（模型生成的客服回复）

import argparse
import json
import os

from tqdm import tqdm
from vllm import LLM, SamplingParams
from vllm.lora.request import LoRARequest


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--base_model", default="/ldata/Qwen/Qwen3-32B")
    p.add_argument("--lora_path", required=True,
                   help="LoRA adapter 路径（必须含 adapter_config.json）")
    p.add_argument("--input_json", required=True,
                   help="eval 集 JSON，list[dict]，每条含 'input' 字段")
    p.add_argument("--output_json", required=True)
    p.add_argument("--max_samples", type=int, default=None,
                   help="只评前 N 条；不传则全量")
    # 完全对齐 notebook cell 4 的 completion(no_think=True) 调用参数
    p.add_argument("--max_tokens", type=int, default=1024)
    p.add_argument("--max_model_len", type=int, default=5000)
    p.add_argument("--tensor_parallel", type=int, default=4)
    p.add_argument("--temperature", type=float, default=0.2)
    p.add_argument("--top_p", type=float, default=0.98)
    p.add_argument("--repetition_penalty", type=float, default=1.2)
    p.add_argument("--presence_penalty", type=float, default=0.4)
    p.add_argument("--no_think", action="store_true", default=True,
                   help="注入 </no_think> 标记，关闭 reasoning（与 notebook 对齐，默认 True）")
    return p.parse_args()


def main():
    args = parse_args()

    print(f"[eval_generate] CUDA_VISIBLE_DEVICES={os.environ.get('CUDA_VISIBLE_DEVICES')}")
    print(f"[eval_generate] base_model = {args.base_model}")
    print(f"[eval_generate] lora_path  = {args.lora_path}")
    print(f"[eval_generate] input_json = {args.input_json}")
    print(f"[eval_generate] output_json= {args.output_json}")

    if not os.path.isfile(os.path.join(args.lora_path, "adapter_config.json")):
        raise SystemExit(f"[eval_generate] adapter_config.json not found in {args.lora_path}")

    with open(args.input_json) as f:
        raw = json.load(f)
    if args.max_samples is not None:
        raw = raw[: args.max_samples]
    print(f"[eval_generate] total prompts = {len(raw)}")

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
    for d in raw:
        if "input" not in d:
            raise ValueError(f"sample missing 'input': keys={list(d.keys())}")
        prompts.append(wrap_chat(d["input"]))

    llm = LLM(
        model=args.base_model,
        trust_remote_code=True,
        dtype="bfloat16",
        max_model_len=args.max_model_len,
        tensor_parallel_size=args.tensor_parallel,
        enable_lora=True,
        max_lora_rank=64,
        disable_log_stats=True,
    )

    sampling = SamplingParams(
        n=1,
        temperature=args.temperature,
        top_p=args.top_p,
        repetition_penalty=args.repetition_penalty,
        presence_penalty=args.presence_penalty,
        max_tokens=args.max_tokens,
        stop=["<|im_end|>"],
        seed=42,
    )

    print(f"[eval_generate] generating (n=1, T={args.temperature}, top_p={args.top_p}, "
          f"rep_penalty={args.repetition_penalty}, presence_penalty={args.presence_penalty}, "
          f"no_think={args.no_think})...")
    results = llm.generate(
        prompts, sampling,
        lora_request=LoRARequest("eval_lora", 1, args.lora_path),
    )

    out = []
    for src, res in zip(raw, results):
        row = dict(src)
        row["response"] = res.outputs[0].text
        out.append(row)

    os.makedirs(os.path.dirname(args.output_json) or ".", exist_ok=True)
    with open(args.output_json, "w") as f:
        json.dump(out, f, ensure_ascii=False, indent=1)
    print(f"[eval_generate] saved {len(out)} rows to {args.output_json}")

    # sanity preview
    print(f"\n[eval_generate] preview (first 3):")
    for i, r in enumerate(out[:3]):
        preview = r["response"][:120].replace("\n", " ")
        print(f"  [{i}] {preview!r}")


if __name__ == "__main__":
    main()




