#!/usr/bin/env bash
set -euo pipefail

# Official FlowSE-NFT paper training entry point.
# Main configuration: multi-reward RMS scaling, conflict-aware filtering
# with tau=0.3, reward weights 2:1:1, and the Early-Time Window setup.
export SPEECH_PROFILE="multi"
export CONFIG="${CONFIG:-config/nft.py:speech_wotext_multi_reward_rms_scale_tau03_weight211}"

exec bash "$(dirname "${BASH_SOURCE[0]}")/scripts/train_speech.sh" "$@"
