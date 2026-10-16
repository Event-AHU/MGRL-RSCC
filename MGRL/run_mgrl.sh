#!/usr/bin/env bash

set -euo pipefail

# =========================
# Basic paths
# =========================
PYTHON_BIN="${PYTHON_BIN:-python}"
PROJECT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
SCRIPT_PATH="${PROJECT_DIR}/work2_train_unified_reward_lrboost_once.py"

# =========================
# GPU / output
# =========================
GPU_ID="${GPU_ID:-0}"
SAVEPATH="${SAVEPATH:-${PROJECT_DIR}/checkpoints/levir_mgrl}"


# =========================
# Data paths
# =========================
DATA_FOLDER="${DATA_FOLDER:-${PROJECT_DIR}/datasets/LEVIR_CC/images}"
LIST_PATH="${LIST_PATH:-${PROJECT_DIR}/datasets/LEVIR_CC}"
TOKEN_FOLDER="${TOKEN_FOLDER:-${LIST_PATH%/}/tokens}"
VOCAB_FILE="${VOCAB_FILE:-vocab}"
KG_CSV="${KG_CSV:-${PROJECT_DIR}/datasets/LEVIR_CC/levir_kg4_three_cleaned.csv}"

# =========================
# Training base settings
# =========================
USE_RL=1
RL_START_EPOCH=8
CE_LAMBDA=0.9
RL_TOPK=3
RL_TEMPERATURE=0.8

NUM_EPOCHS=150
TRAIN_BATCHSIZE=32
VAL_BATCHSIZE=1
WORKERS=2
ENCODER_LR=1e-4
DECODER_LR=1e-5
DECODER_LR_BOOST_EPOCH=30
DECODER_LR_BOOST_VALUE=2e-5

# =========================
# Reward switches
# Set weight=0 to disable
# End epoch = -1 means never turn off
# =========================

# BLEU reward
BLEU_REWARD_WEIGHT=0.3
BLEU_REWARD_START_EPOCH=12
BLEU_REWARD_END_EPOCH=20

# CIDEr reward
CIDER_REWARD_WEIGHT=0.7
CIDER_REWARD_START_EPOCH=8
CIDER_REWARD_END_EPOCH=-1

# Change-state reward
CHANGE_REWARD_WEIGHT=0.3
CHANGE_REWARD_START_EPOCH=8
CHANGE_REWARD_END_EPOCH=10

# Detail reward
DETAIL_REWARD_WEIGHT=0.3
DETAIL_REWARD_START_EPOCH=21
DETAIL_REWARD_END_EPOCH=25
DETAIL_MIN_VOTE=1

# KG reward
KG_REWARD_WEIGHT=0.3
KG_REWARD_START_EPOCH=21
KG_REWARD_END_EPOCH=30

# =========================
# Change-state reward values
# =========================
REWARD_NOCHANGE_NOCHANGE=1.0
REWARD_NOCHANGE_CHANGE=-0.5
REWARD_CHANGE_CHANGE=1.0
REWARD_CHANGE_NOCHANGE=-1.0

# =========================
# KG internal weights
# =========================
KG_ENTITY_WEIGHT=0.5
KG_RELATION_WEIGHT=0.2
KG_TAIL_WEIGHT=0.2
KG_TRIPLET_WEIGHT=0.1
KG_SUPPORTED_BONUS_WEIGHT=0.0

# =========================
# Optional validation / logging
# =========================
VAL_STEP_FREQ=0
VAL_MAX_BATCHES=0
PRINT_FREQ=100

cd "${PROJECT_DIR}"

CMD=(
  "${PYTHON_BIN}"
  "${SCRIPT_PATH}"
  --gpu_id "${GPU_ID}"
  --savepath "${SAVEPATH}"
  --data_folder "${DATA_FOLDER}"
  --list_path "${LIST_PATH}"
  --token_folder "${TOKEN_FOLDER}"
  --vocab_file "${VOCAB_FILE}"
  --kg_csv "${KG_CSV}"
  --num_epochs "${NUM_EPOCHS}"
  --train_batchsize "${TRAIN_BATCHSIZE}"
  --val_batchsize "${VAL_BATCHSIZE}"
  --workers "${WORKERS}"
  --encoder_lr "${ENCODER_LR}"
  --decoder_lr "${DECODER_LR}"
  --decoder_lr_boost_epoch "${DECODER_LR_BOOST_EPOCH}"
  --decoder_lr_boost_value "${DECODER_LR_BOOST_VALUE}"
  --rl_start_epoch "${RL_START_EPOCH}"
  --ce_lambda "${CE_LAMBDA}"
  --rl_topk "${RL_TOPK}"
  --rl_temperature "${RL_TEMPERATURE}"
  --print_freq "${PRINT_FREQ}"
  --val_step_freq "${VAL_STEP_FREQ}"
  --val_max_batches "${VAL_MAX_BATCHES}"
  --reward_nochange_nochange "${REWARD_NOCHANGE_NOCHANGE}"
  --reward_nochange_change "${REWARD_NOCHANGE_CHANGE}"
  --reward_change_change "${REWARD_CHANGE_CHANGE}"
  --reward_change_nochange "${REWARD_CHANGE_NOCHANGE}"
  --detail_min_vote "${DETAIL_MIN_VOTE}"
  --bleu_reward_weight "${BLEU_REWARD_WEIGHT}"
  --bleu_reward_start_epoch "${BLEU_REWARD_START_EPOCH}"
  --bleu_reward_end_epoch "${BLEU_REWARD_END_EPOCH}"
  --cider_reward_weight "${CIDER_REWARD_WEIGHT}"
  --cider_reward_start_epoch "${CIDER_REWARD_START_EPOCH}"
  --cider_reward_end_epoch "${CIDER_REWARD_END_EPOCH}"
  --change_reward_weight "${CHANGE_REWARD_WEIGHT}"
  --change_reward_start_epoch "${CHANGE_REWARD_START_EPOCH}"
  --change_reward_end_epoch "${CHANGE_REWARD_END_EPOCH}"
  --detail_reward_weight "${DETAIL_REWARD_WEIGHT}"
  --detail_reward_start_epoch "${DETAIL_REWARD_START_EPOCH}"
  --detail_reward_end_epoch "${DETAIL_REWARD_END_EPOCH}"
  --kg_reward_weight "${KG_REWARD_WEIGHT}"
  --kg_reward_start_epoch "${KG_REWARD_START_EPOCH}"
  --kg_reward_end_epoch "${KG_REWARD_END_EPOCH}"
  --kg_entity_weight "${KG_ENTITY_WEIGHT}"
  --kg_relation_weight "${KG_RELATION_WEIGHT}"
  --kg_tail_weight "${KG_TAIL_WEIGHT}"
  --kg_triplet_weight "${KG_TRIPLET_WEIGHT}"
  --kg_supported_bonus_weight "${KG_SUPPORTED_BONUS_WEIGHT}"
)

if [[ "${USE_RL}" == "1" ]]; then
  CMD+=(--use_rl)
fi

echo "Running unified reward training with:"
printf ' %q' "${CMD[@]}"
echo

CUDA_VISIBLE_DEVICES="${GPU_ID}" "${CMD[@]}"
