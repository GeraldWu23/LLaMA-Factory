#!/usr/bin/env python
# score_candidates.py
#
# 打分 worker（accelerate 多卡版本，参考 rc_conversations/examples/eval_acc.py）。
# 启动方式: 通过 accelerate launch 调起，run_mini_rft.sh 已封装好。
# 也可手动: accelerate launch --num_processes N --main_process_port PORT score_candidates.py ...
#
# 输入: --input_json   list[dict]，每条至少含 'message' 字段
# 输出: --output_json  原列表每条加上 'score' (float, 正类概率)
#
# 与 eval_acc.py 的差异:
#   - 去掉 sleep_until / metrics / Gains_report（候选集没有 label）
#   - 缺 label 时塞 dummy=0，仅用于 collate，不参与计算

import argparse
import json
import os
import sys

import torch
from accelerate import Accelerator
from torch.utils.data import DataLoader
from tqdm import tqdm
from transformers import AutoTokenizer

RC_CONV_ROOT = "/home/gerald.hu/rc_conversations"
if RC_CONV_ROOT not in sys.path:
    sys.path.insert(0, RC_CONV_ROOT)

from rc_conversations.app.models.qwen_emb_classifier import Qwen3EmbeddingModel
from rc_conversations.app.utils.dataset_utils import DatasetPreprocessor, TextDataset
from rc_conversations.app.utils.utils import load_json, save_json
from rc_conversations.conf.conf import MAX_LENGTH


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument(
        "--model_path",
        default="/ldata/share_data/rucui/model/model_qwen3_4B_emb_csdomain_1to10_JuntoSep_eith_conv_rm08_09_20260312_111329/checkpoint-11000",
    )
    p.add_argument("--input_json", required=True)
    p.add_argument("--output_json", required=True)
    p.add_argument("--max_samples", type=int, default=None,
                   help="只评前 N 条；demo 用，不传则全量")
    p.add_argument("--batch_size", type=int, default=1)
    return p.parse_args()


def main():
    args = parse_args()

    accelerator = Accelerator()
    if accelerator.is_main_process:
        print(
            f"[score_candidates] num_processes={accelerator.num_processes}, "
            f"mixed_precision={accelerator.mixed_precision}"
        )
        print(f"[score_candidates] CUDA_VISIBLE_DEVICES={os.environ.get('CUDA_VISIBLE_DEVICES')}")
        print(f"[score_candidates] model_path={args.model_path}")
        print(f"[score_candidates] input_json={args.input_json}")
        print(f"[score_candidates] output_json={args.output_json}")
    accelerator.wait_for_everyone()

    # ----- 加载数据 -----
    raw_data = load_json(args.input_json)
    if args.max_samples is not None:
        raw_data = raw_data[: args.max_samples]
    for d in raw_data:
        d.setdefault("label", 0)  # dummy，不参与计算

    # 兼容两种输入：
    #   (1) 已带 'message' 字段（如直接读 trainset_*）→ 不动
    #   (2) RFT 候选集（带 'input' + 'response'，没 'message'）
    #       → 用 input 中 '### 对话片段：' 之后的部分 + response 拼成对话
    # 拼接规则与 eval_score.py 完全一致（严格对齐评估时的输入格式）：
    #   - 去掉 SYS_MARKER 之前的指令段
    #   - context 和 response 内部 '\n\n' → 单空格
    #   - context 与 response 之间也用单空格连接
    SYS_MARKER = "### 对话片段："
    n_built_from_response = 0
    for d in raw_data:
        if "message" in d and d["message"]:
            continue
        if "input" in d and "response" in d:
            inp = d["input"]
            pos = inp.find(SYS_MARKER)
            context = inp[pos + len(SYS_MARKER):] if pos >= 0 else inp
            context = context.replace("\n\n", " ").strip()
            resp = d["response"].replace("\n\n", " ").strip()
            d["message"] = [context + " " + resp]
            n_built_from_response += 1
        else:
            raise ValueError(
                f"sample missing 'message' and ('input'+'response'): keys={list(d.keys())}"
            )
    if n_built_from_response and accelerator.is_main_process:
        print(f"[score_candidates] built 'message' from input+response for "
              f"{n_built_from_response}/{len(raw_data)} samples")

    raw_data = DatasetPreprocessor.preprocess_with_maxlen(raw_data, max_length=MAX_LENGTH)
    dataset = TextDataset(raw_data)
    if accelerator.is_main_process:
        print(f"[score_candidates] total samples = {len(dataset)}")

    # ----- 加载模型 -----
    model = Qwen3EmbeddingModel.load_peft_model(args.model_path)
    model.eval()
    model.to(dtype=torch.float32)

    tokenizer = AutoTokenizer.from_pretrained(
        model.base_model.model.config.pretrained_model_name_or_path,
        padding_side="left",
        use_fast=True,
    )
    task_description = model.base_model.model.config.task_description

    dataloader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=4,
        pin_memory=True,
        collate_fn=lambda batch: Qwen3EmbeddingModel.collate_fn_tokenize(
            batch, tokenizer=tokenizer, task_description=task_description
        ),
    )

    model, dataloader = accelerator.prepare(model, dataloader)

    # ----- 推理 -----
    all_pos_probs = []
    with torch.no_grad():
        with accelerator.autocast():
            iterator = tqdm(
                dataloader,
                desc="Scoring",
                disable=not accelerator.is_main_process,
            )
            for batch in iterator:
                logits = model(
                    input_ids=batch["input_ids"],
                    attention_mask=batch["attention_mask"],
                )
                probs = torch.softmax(logits, dim=-1)
                # reward model 输出 probs[:,1] 是"逾期概率"（越高越差），
                # 取 1 - 之，转成"还款 / 不逾期概率"，让 select_top1 仍按"score 越高越好"工作
                pos_probs = 1 - probs[:, 1]
                all_pos_probs.append(accelerator.gather_for_metrics(pos_probs))

    if accelerator.is_main_process:
        all_pos_probs = torch.cat(all_pos_probs).cpu().tolist()
        assert len(all_pos_probs) == len(raw_data), (
            f"length mismatch: {len(all_pos_probs)} vs {len(raw_data)}"
        )
        for d, s in zip(raw_data, all_pos_probs):
            d["score"] = float(s)
            d.pop("label", None)

        os.makedirs(os.path.dirname(args.output_json) or ".", exist_ok=True)
        save_json(raw_data, args.output_json)
        print(f"[score_candidates] saved to {args.output_json}")
        print(f"[score_candidates] sample scores = {all_pos_probs[:3]}")


if __name__ == "__main__":
    main()
