from __future__ import annotations

from absl import app
from ml_collections import config_flags


import warnings
warnings.filterwarnings("ignore")

from flow_nft.speech_orchestrator import SpeechNFTOrchestrator
from flow_nft.speech_diagnostics import diagnostic_session


_CONFIG = config_flags.DEFINE_config_file(
    "config",
    "config/nft.py:speech_wotext_multi_reward_rms_scale_tau03_weight211",
    "Speech NFT training configuration.",
)


def main(_):
    config = _CONFIG.value
    orchestrator = SpeechNFTOrchestrator(config)
    with diagnostic_session():
        orchestrator.run()


if __name__ == "__main__":
    app.run(main)
