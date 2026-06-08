#!/usr/bin/env bash
# run_mini_rft.sh
#
# 端到端 mini RFT pipeline：M0 SFT → 2 轮 (vLLM 采样 + 打分 + 选 top-1 + LoRA 续训)
# 全部使用 0,1,2,3,4,5,6,7 八卡。脚本本身就是正式版的最小化数据测试版，正式跑只改数据即可。
#
# 环境分工:
#   - 训练 + 打分: llm env
#   - vLLM 采样: qwen3coder env
#
# 使用方式：
#   bash examples/train_rft/run_mini_rft.sh

set -euo pipefail

# =========================================================================
# wandb 实验名（每次跑前可改 RUN_TAG；本地 wandb，请确保已 wandb offline）
# 同一 RUN_TAG 下：M0/round1/round2 是 3 个独立 run，归到同一 group
# =========================================================================
RUN_TAG=mini_$(date +%Y%m%d_%H%M%S)
export WANDB_PROJECT=rc_conv
export WANDB_RUN_GROUP=$RUN_TAG
echo "[run_mini_rft] WANDB_RUN_GROUP=$WANDB_RUN_GROUP"

# =========================================================================
# 路径与参数
# =========================================================================
LLAMA_FACTORY=/home/gerald.hu/llama-factory
TRAIN_RFT_DIR=$LLAMA_FACTORY/examples/train_rft

PY_LLM=/home/gerald.hu/.conda/envs/llm/bin/python
PY_VLLM=/home/gerald.hu/.conda/envs/qwen3coder/bin/python
ACCEL_LLM=/home/gerald.hu/.conda/envs/llm/bin/accelerate

# 数据文件目录（LLaMA-Factory dataset 解析时会从这里读）
DATA_DIR=$LLAMA_FACTORY/data

# 中间产物目录（candidates / scored / top1 等 JSON）
TMP_ROOT=/ldata/share_data/cuishou_conv/tmp/rft_mini/$RUN_TAG
mkdir -p "$TMP_ROOT"

# 最终 LoRA 产物目录（与 qwen3_32b_gen_cuishou_long3k_0605 同级）
LORA_ROOT=/ldata/share_data/cuishou_conv/models/lora_models/qwen3_32b_rft_${RUN_TAG}
mkdir -p "$LORA_ROOT"

echo "[run_mini_rft] TMP_ROOT  = $TMP_ROOT"
echo "[run_mini_rft] LORA_ROOT = $LORA_ROOT"

MINI_DATASET_NAME=trainset_mini_s1ast_202508_202601_20260608
MINI_DATA_PATH=$DATA_DIR/${MINI_DATASET_NAME}.json

REWARD_MODEL=/ldata/share_data/rucui/model/model_qwen3_4B_emb_csdomain_1to10_JuntoSep_eith_conv_rm08_09_20260312_111329/checkpoint-11000
BASE_MODEL=/ldata/Qwen/Qwen3-32B

# GPU 分配（mini 用 8 卡）
GPU_TRAIN="0,1,2,3,4,5,6,7"
GPU_SAMPLE="0,1,2,3,4,5,6,7"
GPU_SCORE="0,1,2,3,4,5,6,7"
NPROC=8
ACCEL_PORT=29555
VLLM_TP=8

# RFT 采样参数
RFT_N=4
RFT_TEMP=0.9
RFT_TOPP=0.95
RFT_MAXTOK=512

N_ROUNDS=2

# =========================================================================
# 工具函数：找最新的 checkpoint-* 子目录（adapter 真实路径）
# =========================================================================
latest_checkpoint() {
    # $1 = output_dir
    local d=$1
    # 优先找 checkpoint-* 子目录里最大的；没有则用 d 本身
    local last=$(ls -d "$d"/checkpoint-* 2>/dev/null | sort -V | tail -n 1 || true)
    if [[ -n "$last" && -f "$last/adapter_config.json" ]]; then
        echo "$last"
    elif [[ -f "$d/adapter_config.json" ]]; then
        echo "$d"
    else
        echo "[run_mini_rft] ERROR: no adapter_config.json under $d" >&2
        return 1
    fi
}

# =========================================================================
# Round 0: M0 SFT
# =========================================================================
echo "=========================================="
echo "[run_mini_rft] Round 0 (SFT)"
echo "=========================================="
M0_DIR=$LORA_ROOT/m0_sft
SFT_YAML=$TRAIN_RFT_DIR/qwen3_qlora_mini_sft.yaml

cd "$LLAMA_FACTORY"
CUDA_VISIBLE_DEVICES=$GPU_TRAIN $PY_LLM -m llamafactory.cli train "$SFT_YAML" \
    output_dir="$M0_DIR" \
    dataset="$MINI_DATASET_NAME"
echo "[run_mini_rft] M0 SFT done at $M0_DIR"

PREV_LORA=$(latest_checkpoint "$M0_DIR")
echo "[run_mini_rft] PREV_LORA = $PREV_LORA"

# =========================================================================
# RFT 循环
# =========================================================================
for ROUND in $(seq 1 $N_ROUNDS); do
    echo
    echo "=========================================="
    echo "[run_mini_rft] Round $ROUND"
    echo "=========================================="
    ROUND_DIR=$TMP_ROOT/round${ROUND}
    mkdir -p "$ROUND_DIR"

    # ---- A. vLLM 采样（qwen3coder env）----
    CAND_JSON=$ROUND_DIR/candidates.json
    echo "[round $ROUND] (A) vLLM sampling -> $CAND_JSON"
    CUDA_VISIBLE_DEVICES=$GPU_SAMPLE $PY_VLLM \
        "$TRAIN_RFT_DIR/sample_candidates.py" \
        --base_model "$BASE_MODEL" \
        --lora_path "$PREV_LORA" \
        --input_json "$MINI_DATA_PATH" \
        --output_json "$CAND_JSON" \
        --n $RFT_N \
        --temperature $RFT_TEMP \
        --top_p $RFT_TOPP \
        --max_tokens $RFT_MAXTOK \
        --tensor_parallel $VLLM_TP

    # ---- B. 打分（llm env，accelerate launch 多卡）----
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
    TOP1_DATASET_NAME=mini_rft_round${ROUND}_top1
    TOP1_FOR_LF=$DATA_DIR/${TOP1_DATASET_NAME}.json

    echo "[round $ROUND] (C) selecting top-1 -> $TOP1_JSON"
    $PY_LLM "$TRAIN_RFT_DIR/select_top1.py" \
        --scored_json "$SCORED_JSON" \
        --output_json "$TOP1_JSON" \
        --group_key account_id

    cp "$TOP1_JSON" "$TOP1_FOR_LF"
    echo "[round $ROUND] copied to $TOP1_FOR_LF"

    # 注册 dataset_info.json
    $PY_LLM -c "
import json
p = '$DATA_DIR/dataset_info.json'
with open(p) as f: info = json.load(f)
key = '$TOP1_DATASET_NAME'
info[key] = {'file_name': '${TOP1_DATASET_NAME}.json', 'columns': {'prompt':'input','response':'target'}}
with open(p,'w') as f: json.dump(info, f, ensure_ascii=False, indent=2)
print('[dataset_info] registered/updated:', key)
"

    # ---- D. 训练 round R LoRA（所有动态字段命令行覆盖）----
    ROUND_LORA=$LORA_ROOT/round${ROUND}
    RFT_YAML=$TRAIN_RFT_DIR/qwen3_qlora_mini_rft.yaml

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
echo "[run_mini_rft] ALL DONE"
echo "Final LoRA: $PREV_LORA"
echo "=========================================="
