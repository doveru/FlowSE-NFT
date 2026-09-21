#!/usr/bin/env bash
set -euo pipefail
export SPEECH_PROFILE="dnsmos"
exec bash "$(dirname "${BASH_SOURCE[0]}")/scripts/train_speech.sh" "$@"
