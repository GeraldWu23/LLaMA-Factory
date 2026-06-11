#!/usr/bin/env bash
# run_full_rft_from_sft.sh
#
# 跳过 M0 SFT，直接基于一个已有的 SFT LoRA 跑 3 轮 RFT。
# 用于：SFT 已经训过、希望复用结果重跑 / 续跑 RFT 的场景。
#
# 与 run_full_rft.sh 的唯一区别：
#   - 删掉 Round 0（SFT 训练），改为从 SFT_LORA 直接接 round 1
#   - RUN_TAG 默认带 from_sft 前缀，避免与 SFT 同跑的产物冲突
#
# 使用方式：
#   1. 改下面的 SFT_LORA 路径，指向已有的 SFT LoRA（含 adapter_config.json）
#   2. bash examples/train_rft/run_full_rft_from_sft.sh
#
# 环境分工:
#   - 训练 + 打分: llm env
#   - vLLM 采样: qwen3coder env

set -euo pipefail

# =========================================================================
# ⚠️ 必改：已有的 SFT LoRA 路径（必须包含 adapter_config.json）
# =========================================================================
SFT_LORA=/ldata/share_data/cuishou_conv/models/lora_models/qwen3_32b_rft_full_20260608_175236/m0_sft/checkpoint-3865

# =========================================================================
# wandb 实验名
# =========================================================================
RUN_TAG=from_sft_$(date +%Y%m%d_%H%M%S)
export WANDB_PROJECT=rc_conv
export WANDB_RUN_GROUP=$RUN_TAG
echo "[run_full_rft_from_sft] WANDB_RUN_GROUP=$WANDB_RUN_GROUP"

# =========================================================================
# 路径与参数
# =========================================================================
LLAMA_FACTORY=/home/gerald.hu/llama-factory
TRAIN_RFT_DIR=$LLAMA_FACTORY/examples/train_rft

PY_LLM=/home/gerald.hu/.conda/envs/llm/bin/python
PY_VLLM=/home/gerald.hu/.conda/envs/qwen3coder/bin/python
ACCEL_LLM=/home/gerald.hu/.conda/envs/llm/bin/accelerate

DATA_DIR=$LLAMA_FACTORY/data

TMP_ROOT=/ldata/share_data/cuishou_conv/tmp/rft_full/$RUN_TAG
mkdir -p "$TMP_ROOT"

LORA_ROOT=/ldata/share_data/cuishou_conv/models/lora_models/qwen3_32b_rft_${RUN_TAG}
mkdir -p "$LORA_ROOT"

echo "[run_full_rft_from_sft] SFT_LORA  = $SFT_LORA"
echo "[run_full_rft_from_sft] TMP_ROOT  = $TMP_ROOT"
echo "[run_full_rft_from_sft] LORA_ROOT = $LORA_ROOT"

# 数据集
RFT_PROMPT_DATASET=trainset_rft_s1ast_202508_202601_20260608
RFT_PROMPT_PATH=$DATA_DIR/${RFT_PROMPT_DATASET}.json

REWARD_MODEL=/ldata/share_data/rucui/model/model_qwen3_4B_emb_csdomain_1to10_JuntoSep_eith_conv_rm08_09_20260312_111329/checkpoint-11000
BASE_MODEL=/ldata/Qwen/Qwen3-32B

# GPU 分配
GPU_TRAIN="0,1,2,3,4,5,6,7"
GPU_SAMPLE="0,1,2,3,4,5,6,7"
GPU_SCORE="0,1,2,3,4,5,6,7"
NPROC=8
ACCEL_PORT=29555
VLLM_TP=8

# RFT 采样参数
RFT_N=8
RFT_TEMP=0.9
RFT_TOPP=0.95
RFT_MAXTOK=512

N_ROUNDS=3

# =========================================================================
# 工具函数
# =========================================================================
latest_checkpoint() {
    local d=$1
    local last=$(ls -d "$d"/checkpoint-* 2>/dev/null | sort -V | tail -n 1 || true)
    if [[ -n "$last" && -f "$last/adapter_config.json" ]]; then
        echo "$last"
    elif [[ -f "$d/adapter_config.json" ]]; then
        echo "$d"
    else
        echo "[run_full_rft_from_sft] ERROR: no adapter_config.json under $d" >&2
        return 1
    fi
}

# =========================================================================
# 校验 SFT_LORA 存在
# =========================================================================
if [[ ! -f "$SFT_LORA/adapter_config.json" ]]; then
    echo "[run_full_rft_from_sft] ERROR: SFT_LORA missing adapter_config.json:" >&2
    echo "  $SFT_LORA" >&2
    exit 1
fi
PREV_LORA=$SFT_LORA
echo "[run_full_rft_from_sft] PREV_LORA = $PREV_LORA"

# =========================================================================
# RFT 循环（直接从 round 1 开始）
# =========================================================================
for ROUND in $(seq 1 $N_ROUNDS); do
    echo
    echo "=========================================="
    echo "[run_full_rft_from_sft] Round $ROUND"
    echo "=========================================="
    ROUND_DIR=$TMP_ROOT/round${ROUND}
    mkdir -p "$ROUND_DIR"

    # ---- A. vLLM 采样 ----
    CAND_JSON=$ROUND_DIR/candidates.json
    echo "[round $ROUND] (A) vLLM sampling -> $CAND_JSON"
    CUDA_VISIBLE_DEVICES=$GPU_SAMPLE $PY_VLLM \
        "$TRAIN_RFT_DIR/sample_candidates.py" \
        --base_model "$BASE_MODEL" \
        --lora_path "$PREV_LORA" \
        --input_json "$RFT_PROMPT_PATH" \
        --output_json "$CAND_JSON" \
        --n $RFT_N \
        --temperature $RFT_TEMP \
        --top_p $RFT_TOPP \
        --max_tokens $RFT_MAXTOK \
        --tensor_parallel $VLLM_TP

    # ---- B. 打分 ----
    SCORED_JSON=$ROUND_DIR/scored.json
    echo "[round $ROUND] (B) scoring -> $SCORED_JSON"
    CUDA_VISIBLE_DEVICES=$GPU_SCORE $ACCEL_LLM launch \
        --num_processes $NPROC \
        --main_process_port $ACCEL_PORT \
        "$TRAIN_RFT_DIR/score_candidates.py" \
        --model_path "$REWARD_MODEL" \
        --input_json "$CAND_JSON" \
        --output_json "$SCORED_JSON"

    # ---- C. 选 top-1 ----
    TOP1_JSON=$ROUND_DIR/top1.json
    TOP1_DATASET_NAME=full_rft_${RUN_TAG}_round${ROUND}_top1
    TOP1_FOR_LF=$DATA_DIR/${TOP1_DATASET_NAME}.json

    echo "[round $ROUND] (C) selecting top-1 -> $TOP1_JSON"
    $PY_LLM "$TRAIN_RFT_DIR/select_top1.py" \
        --scored_json "$SCORED_JSON" \
        --output_json "$TOP1_JSON" \
        --group_key account_id

    cp "$TOP1_JSON" "$TOP1_FOR_LF"
    echo "[round $ROUND] copied to $TOP1_FOR_LF"

    $PY_LLM -c "
import json
p = '$DATA_DIR/dataset_info.json'
with open(p) as f: info = json.load(f)
key = '$TOP1_DATASET_NAME'
info[key] = {'file_name': '${TOP1_DATASET_NAME}.json', 'columns': {'prompt':'input','response':'target'}}
with open(p,'w') as f: json.dump(info, f, ensure_ascii=False, indent=2)
print('[dataset_info] registered/updated:', key)
"

    # ---- D. 训练 round R LoRA ----
    ROUND_LORA=$LORA_ROOT/round${ROUND}
    RFT_YAML=$TRAIN_RFT_DIR/qwen3_qlora_full_rft.yaml

    cd "$LLAMA_FACTORY"
    CUDA_VISIBLE_DEVICES=$GPU_TRAIN $PY_LLM -m llamafactory.cli train "$RFT_YAML" \
        output_dir="$ROUND_LORA" \
        dataset="$TOP1_DATASET_NAME" \
        adapter_name_or_path="$PREV_LORA" \
        run_name="${RUN_TAG}_round${ROUND}"
    echo "[round $ROUND] training done -> $ROUND_LORA"

    PREV_LORA=$(latest_checkpoint "$ROUND_LORA")
    echo "[round $ROUND] PREV_LORA = $PREV_LORA"
done

echo
echo "=========================================="
echo "[run_full_rft_from_sft] ALL DONE"
echo "Final LoRA: $PREV_LORA"
echo "=========================================="
