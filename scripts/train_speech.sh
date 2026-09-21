#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/.."

# Environment must be activated by the caller. Relative paths use the project root.
profile="${SPEECH_PROFILE:-multi}"
export WANDB_START_METHOD="${WANDB_START_METHOD:-thread}"
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"
export TORCH_DISTRIBUTED_TIMEOUT_MINUTES="${TORCH_DISTRIBUTED_TIMEOUT_MINUTES:-120}"
export TORCH_NCCL_ASYNC_ERROR_HANDLING="${TORCH_NCCL_ASYNC_ERROR_HANDLING:-1}"
export NCCL_ASYNC_ERROR_HANDLING="${NCCL_ASYNC_ERROR_HANDLING:-1}"
export SPEECH_DNSMOS_DEVICE="${SPEECH_DNSMOS_DEVICE:-cpu}"
export VOICE_EVALUATION_ROOT="${VOICE_EVALUATION_ROOT:-$(dirname "$PWD")/voice_evaluation}"
export SPEECH_DATA_ROOT="${SPEECH_DATA_ROOT:-$(dirname "$PWD")/DNS_noreverb}"
export TRAIN_MANIFEST="${TRAIN_MANIFEST:-$PWD/dataset/speech/train_manifest.jsonl}"
export VAL_MANIFEST="${VAL_MANIFEST:-$PWD/dataset/speech/val_manifest.jsonl}"
export INIT_CHECKPOINT="${INIT_CHECKPOINT:-$PWD/flow_grpo/speech_flowse/ckpts/best.pt.tar}"
export SPEECHBERT_MODEL_PATH="${SPEECHBERT_MODEL_PATH:-$VOICE_EVALUATION_ROOT/hubert-base-ls960}"
export PYTHONPATH="$PWD${PYTHONPATH:+:$PYTHONPATH}"

case "$profile" in
  multi)
    NPROC_PER_NODE="${NPROC_PER_NODE:-3}"
    CONFIG="${CONFIG:-config/nft.py:speech_wotext_multi_reward_rms_scale_tau03_weight211}"
    export WANDB_MODE="${WANDB_MODE:-offline}"
    export SPEAKER_MODEL_TYPE="${SPEAKER_MODEL_TYPE:-eres2net}"
    export SPEAKER_MODEL_PATH="${SPEAKER_MODEL_PATH:-$VOICE_EVALUATION_ROOT/eres2net}"
    export SPEAKER_CODE_PATH="${SPEAKER_CODE_PATH:-$VOICE_EVALUATION_ROOT/3D-Speaker}"
    profile_args=()
    ;;
  dnsmos)
    NPROC_PER_NODE="${NPROC_PER_NODE:-4}"
    CONFIG="${CONFIG:-config/nft.py:speech_wotext_pure_nft}"
    export WANDB_MODE="${WANDB_MODE:-online}"
    NUM_CANDIDATES="${NUM_CANDIDATES:-24}"
    TIMESTEPS_PER_BATCH="${TIMESTEPS_PER_BATCH:-32}"
    TIMESTEP_FRACTION="${TIMESTEP_FRACTION:-1.0}"
    MAX_GRAD_NORM="${MAX_GRAD_NORM:-1.0}"
    SPEECH_REWARD_DEVICE="${SPEECH_REWARD_DEVICE:-cpu}"

    # Full-SDE rollout by default: no mixed SDE/ODE window, all sampling steps stochastic.
    MIXED_SAMPLING_ENABLED="${MIXED_SAMPLING_ENABLED:-False}"
    ROLLOUT_DETERMINISTIC="${ROLLOUT_DETERMINISTIC:-False}"
    ROLLOUT_NOISE_LEVEL="${ROLLOUT_NOISE_LEVEL:-0.4}"

    profile_args=(
      --config.speech.data.train_manifest="${TRAIN_MANIFEST}"
      --config.speech.data.val_manifest="${VAL_MANIFEST}"
      --config.speech.model.init_checkpoint="${INIT_CHECKPOINT}"
      --config.speech.rollout.num_candidates="${NUM_CANDIDATES}"
      --config.speech.rollout.mixed_sampling_enabled="${MIXED_SAMPLING_ENABLED}"
      --config.speech.rollout.deterministic="${ROLLOUT_DETERMINISTIC}"
      --config.speech.rollout.noise_level="${ROLLOUT_NOISE_LEVEL}"
      --config.speech.train.timesteps_per_batch="${TIMESTEPS_PER_BATCH}"
      --config.speech.train.timestep_fraction="${TIMESTEP_FRACTION}"
      --config.speech.train.max_grad_norm="${MAX_GRAD_NORM}"
      --config.speech.reward.weights.dnsmos=1.0
      --config.speech.reward.normalization=raw_linear
      --config.speech.reward.primary_keys.dnsmos=dnsmos_avg
      --config.speech.reward.voice_eval_root="${VOICE_EVALUATION_ROOT}"
      --config.speech.reward.device="${SPEECH_REWARD_DEVICE}"
    )
    ;;
  *) echo "Unknown SPEECH_PROFILE: $profile (multi, dnsmos)" >&2; exit 2 ;;
esac

command=(torchrun --nproc_per_node="$NPROC_PER_NODE" scripts/train_nft_speech.py
  --config "$CONFIG")
if (( ${#profile_args[@]} )); then
  command+=("${profile_args[@]}")
fi
command+=("$@")
if [[ "${DRY_RUN:-0}" == 1 ]]; then
  printf 'Profile: %s\n' "$profile"
  printf '%s\n' "SPEECH_DATA_ROOT=$SPEECH_DATA_ROOT" "VOICE_EVALUATION_ROOT=$VOICE_EVALUATION_ROOT" "TRAIN_MANIFEST=$TRAIN_MANIFEST" "VAL_MANIFEST=$VAL_MANIFEST" "INIT_CHECKPOINT=$INIT_CHECKPOINT" "SPEECHBERT_MODEL_PATH=$SPEECHBERT_MODEL_PATH"
  printf '%q ' "${command[@]}"
  printf '\n'
  exit 0
fi
exec "${command[@]}"
