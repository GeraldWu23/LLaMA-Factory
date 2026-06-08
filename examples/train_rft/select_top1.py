#!/usr/bin/env python
# select_top1.py
#
# 输入: --scored_json，由 score_candidates.py 产出，每行 = {prompt 原字段..., sample_idx, response, score}
#       同一 (account_id) 下有 N 条候选（N 个 sample_idx）
# 输出: --output_json，每个 account_id 留 1 条 score 最高的，转成 LLaMA-Factory 训练格式
#       字段：保留 input + 用 response 替换 target

import argparse
import json
import os
from collections import defaultdict


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--scored_json", required=True,
                   help="score_candidates.py 输出，flat list，每行有 score 字段")
    p.add_argument("--output_json", required=True,
                   help="LLaMA-Factory 训练用 json，每 prompt 留 1 条 top score")
    p.add_argument("--group_key", default="account_id",
                   help="按哪个字段分组（默认 account_id；如果同账户多 prompt 改成 input）")
    p.add_argument("--min_score", type=float, default=None,
                   help="可选：top-1 分数低于此阈值时丢弃这个 prompt（避免学烂的）")
    return p.parse_args()


def main():
    args = parse_args()

    with open(args.scored_json) as f:
        data = json.load(f)
    print(f"[select_top1] loaded {len(data)} scored rows")

    if not data:
        raise SystemExit("[select_top1] empty input")

    # 兼容性检查
    sample = data[0]
    for k in ("score", "response", args.group_key):
        if k not in sample:
            raise SystemExit(f"[select_top1] missing key {k!r} in input; keys={list(sample.keys())}")

    # 按 group_key 分组
    groups = defaultdict(list)
    for r in data:
        groups[r[args.group_key]].append(r)

    print(f"[select_top1] {len(groups)} unique groups (avg {len(data)/len(groups):.1f} candidates/group)")

    # 选 top-1
    selected = []
    n_dropped_by_score = 0
    for gk, items in groups.items():
        items.sort(key=lambda r: r["score"], reverse=True)
        top = items[0]
        if args.min_score is not None and top["score"] < args.min_score:
            n_dropped_by_score += 1
            continue

        # 转成 LLaMA-Factory 格式：input/target 风格 (与 dataset_info.json 现有 columns 一致)
        out = {k: v for k, v in top.items() if k not in ("response", "score", "sample_idx")}
        out["target"] = top["response"]   # 用模型生成的 top-1 替换原 target
        out["_rft_score"] = top["score"]  # 记录分数（不参与训练，方便日志查看）
        selected.append(out)

    print(f"[select_top1] selected {len(selected)}; dropped {n_dropped_by_score} by min_score")

    if args.min_score is not None:
        # 简单分数分布
        if selected:
            scores = sorted(r["_rft_score"] for r in selected)
            mid = len(scores) // 2
            print(f"[select_top1] kept score range: min={scores[0]:.4f} "
                  f"median={scores[mid]:.4f} max={scores[-1]:.4f}")

    os.makedirs(os.path.dirname(args.output_json) or ".", exist_ok=True)
    with open(args.output_json, "w") as f:
        json.dump(selected, f, ensure_ascii=False, indent=1)
    print(f"[select_top1] saved to {args.output_json}")


if __name__ == "__main__":
    main()
