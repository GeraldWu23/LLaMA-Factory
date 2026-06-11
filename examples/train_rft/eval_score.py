#!/usr/bin/env python
# eval_score.py
#
# 用 reward model 给 eval_generate.py 产出的 response 打分。
#
# 用法：
#   CUDA_VISIBLE_DEVICES=0 /home/gerald.hu/.conda/envs/llm/bin/python \
#       /home/gerald.hu/llama-factory/examples/train_rft/eval_score.py
#
# 输入: list[dict]，每条至少含 'input' 和 'response' 字段
# 输出: 纯分数列表 json，每条一个 float（pos_prob），顺序与输入一一对应

import json
import os
import sys


# =========================================================================
# 配置区
# =========================================================================
REWARD_MODEL = "/ldata/share_data/rucui/model/model_qwen3_4B_emb_csdomain_1to10_JuntoSep_eith_conv_rm08_09_20260312_111329/checkpoint-11000"
INPUT_JSON = "/ldata/share_data/cuishou_conv/eval/round1_responses.json"
OUTPUT_PATH = "/ldata/share_data/cuishou_conv/eval/round1_scores.json"


# =========================================================================
def main():
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

    print(f"[eval_score] reward_model = {REWARD_MODEL}")
    print(f"[eval_score] input_json   = {INPUT_JSON}")
    print(f"[eval_score] output_path  = {OUTPUT_PATH}")

    raw = load_json(INPUT_JSON)

    SYS_MARKER = "### 对话片段："
    for d in raw:
        d.setdefault("label", 0)
        inp = d.get("input", "")
        resp = d.get("response", "")
        # 去掉 SYS_MARKER 之前的指令
        pos = inp.find(SYS_MARKER)
        context = inp[pos + len(SYS_MARKER):] if pos >= 0 else inp
        # '\n\n' → 单空格
        context = context.replace("\n\n", " ").strip()
        resp = resp.replace("\n\n", " ").strip()
        # context 与 response 用单空格连接
        d["message"] = [context + " " + resp]

    raw = DatasetPreprocessor.preprocess_with_maxlen(raw, max_length=MAX_LENGTH)
    dataset = TextDataset(raw)
    print(f"[eval_score] total samples = {len(dataset)}")

    model = Qwen3EmbeddingModel.load_peft_model(REWARD_MODEL)
    model.eval()
    model.to(dtype=torch.float32)
    model.cuda()

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

    scores = []
    with torch.no_grad():
        for batch in tqdm(dataloader, desc="Scoring"):
            input_ids = batch["input_ids"].cuda()
            attention_mask = batch["attention_mask"].cuda()
            logits = model(input_ids=input_ids, attention_mask=attention_mask)
            probs = torch.softmax(logits, dim=-1)
            scores.extend(probs[:, 1].cpu().tolist())

    assert len(scores) == len(raw)
    os.makedirs(os.path.dirname(OUTPUT_PATH) or ".", exist_ok=True)
    with open(OUTPUT_PATH, "w") as f:
        json.dump(scores, f)
    print(f"[eval_score] saved {len(scores)} scores to {OUTPUT_PATH}")


if __name__ == "__main__":
    main()
