# FlowSE-NFT: Efficient Multi-Reward Post-Training for Flow-Matching Speech Enhancement

Official implementation of **FlowSE-NFT**, a likelihood-free online reinforcement learning framework for flow-matching speech enhancement.

FlowSE-NFT adapts Negative-aware Fine-Tuning (NFT) to the FlowSE backbone and optimizes multiple speech-quality objectives without explicit policy-likelihood estimation or full reverse-trajectory policy-gradient updates. The method combines:

- **Multi-reward alignment** with DNSMOS, speaker similarity, and SpeechBERTScore.
- **Conflict-Aware Filtering** to remove rollouts with severe disagreement across reward-wise advantages before aggregation.
- **Early-Time Window training** to concentrate NFT updates near the Gaussian-noise endpoint and reduce online training cost.

The default training entry point reproduces the paper configuration with 24 rollouts per utterance, 48 utterances per epoch, reward weights `0.5 / 0.25 / 0.25` (implemented as the equivalent ratio `2 / 1 / 1`), conflict threshold `tau = 0.3`, and Early-Time Window `[0, 0.25]`.

## Installation

Python 3.10 is recommended.

```bash
conda create -n flowse-nft python=3.10 -y
conda activate flowse-nft

# Install PyTorch for your CUDA version. The paper code was validated with PyTorch 2.6.0.
pip install torch==2.6.0 torchaudio==2.6.0 torchvision==0.21.0 \
  --index-url https://download.pytorch.org/whl/cu126

pip install -e .
```

For development/tests:

```bash
pip install -e '.[dev]'
```

## Pretrained Models & Reward Models

The code uses four external model/resource groups: the pretrained **FlowSE** backbone, **DNSMOS P.835**, **HuBERT** features for SpeechBERTScore, and **ERes2Net** for speaker similarity.

By default, the repository expects the following layout. `voice_evaluation` is a sibling directory of this repository because `VOICE_EVALUATION_ROOT` defaults to `../voice_evaluation`.

```text
<workspace>/
├── FlowSE-NFT/
│   ├── flow_grpo/
│   │   └── speech_flowse/
│   │       └── ckpts/
│   │           └── best.pt.tar
│   └── ...
│
└── voice_evaluation/
    ├── evaluation/
    │   ├── dnsmos.py
    │   └── DNSMOS/
    │       ├── sig_bak_ovr.onnx
    │       └── model_v8.onnx
    ├── hubert-base-ls960/
    ├── eres2net/
    │   └── pretrained_eres2net_aug.ckpt
    └── 3D-Speaker/
        └── speakerlab/
            └── ...
```

You may use different locations by setting `INIT_CHECKPOINT`, `VOICE_EVALUATION_ROOT`, `SPEECHBERT_MODEL_PATH`, `SPEAKER_MODEL_PATH`, and `SPEAKER_CODE_PATH`.

### 1. FlowSE backbone

FlowSE-NFT starts from the pretrained **FlowSE: Efficient and High-Quality Speech Enhancement via Flow Matching** model (Interspeech 2025):

- Official repository: https://github.com/honee-w/flowse
- Project page: https://honee-w.github.io/FlowSE/

Download the pretrained **w/o-text FlowSE checkpoint** provided by the FlowSE authors and place/copy it at:

```text
flow_grpo/speech_flowse/ckpts/best.pt.tar
```

For example, after downloading the checkpoint:

```bash
mkdir -p flow_grpo/speech_flowse/ckpts
cp /path/to/downloaded/best.pt.tar flow_grpo/speech_flowse/ckpts/best.pt.tar
```

Alternatively:

```bash
export INIT_CHECKPOINT=/absolute/path/to/best.pt.tar
```

> The FlowSE authors distribute the pretrained weights through their official project/repository. We intentionally do not mirror third-party checkpoints in this repository.

### 2. DNSMOS P.835

FlowSE-NFT uses the official Microsoft DNSMOS P.835 implementation and the following ONNX models:

```text
sig_bak_ovr.onnx
model_v8.onnx
```

Official source: https://github.com/microsoft/DNS-Challenge/tree/master/DNSMOS

Create the directory expected by the code:

```bash
export VOICE_EVALUATION_ROOT="$(dirname "$PWD")/voice_evaluation"
mkdir -p "$VOICE_EVALUATION_ROOT/evaluation/DNSMOS"
```

Download the official DNSMOS implementation and checkpoints:

```bash
wget -O "$VOICE_EVALUATION_ROOT/evaluation/dnsmos.py" \
  https://raw.githubusercontent.com/microsoft/DNS-Challenge/master/DNSMOS/dnsmos_local.py

wget -O "$VOICE_EVALUATION_ROOT/evaluation/DNSMOS/sig_bak_ovr.onnx" \
  https://raw.githubusercontent.com/microsoft/DNS-Challenge/master/DNSMOS/DNSMOS/sig_bak_ovr.onnx

wget -O "$VOICE_EVALUATION_ROOT/evaluation/DNSMOS/model_v8.onnx" \
  https://raw.githubusercontent.com/microsoft/DNS-Challenge/master/DNSMOS/DNSMOS/model_v8.onnx
```

The reward implementation imports `evaluation.dnsmos.ComputeScore`, so the official `dnsmos_local.py` is intentionally saved as `evaluation/dnsmos.py`.

### 3. SpeechBERTScore / HuBERT

SpeechBERTScore in this repository uses frame-level features from:

```text
facebook/hubert-base-ls960
hidden layer: 8
sampling rate: 16 kHz
```

Official model: https://huggingface.co/facebook/hubert-base-ls960

The model can be downloaded directly into the path expected by the default configuration:

```bash
pip install -U huggingface_hub

huggingface-cli download facebook/hubert-base-ls960 \
  --local-dir "$VOICE_EVALUATION_ROOT/hubert-base-ls960"
```

Then the default path is:

```bash
export SPEECHBERT_MODEL_PATH="$VOICE_EVALUATION_ROOT/hubert-base-ls960"
```

If no explicit local path is supplied and remote loading is enabled, the implementation can also resolve `facebook/hubert-base-ls960` through Transformers/Hugging Face.

### 4. ERes2Net speaker encoder

Speaker similarity is the cosine similarity between clean-reference and enhanced-speech embeddings extracted with the official 3D-Speaker **ERes2Net** model.

The exact model used by this code is:

```text
ModelScope ID: iic/speech_eres2net_sv_zh-cn_16k-common
revision:      v1.0.5
architecture:  speakerlab.models.eres2net.ERes2Net_huge.ERes2Net
feat_dim:      80
embedding:     192
checkpoint:    pretrained_eres2net_aug.ckpt
sampling rate: 16 kHz
```

Official 3D-Speaker repository: https://github.com/modelscope/3D-Speaker

Clone the model code into the default location:

```bash
git clone https://github.com/modelscope/3D-Speaker.git \
  "$VOICE_EVALUATION_ROOT/3D-Speaker"
```

Download the matching ModelScope snapshot:

```bash
pip install -U modelscope

python - <<'PY'
import os
from modelscope.hub.snapshot_download import snapshot_download

root = os.environ['VOICE_EVALUATION_ROOT']
snapshot_download(
    'iic/speech_eres2net_sv_zh-cn_16k-common',
    revision='v1.0.5',
    local_dir=os.path.join(root, 'eres2net'),
)
PY
```

Verify that the checkpoint exists at:

```text
$VOICE_EVALUATION_ROOT/eres2net/pretrained_eres2net_aug.ckpt
```

The corresponding defaults are:

```bash
export SPEAKER_MODEL_TYPE=eres2net
export SPEAKER_MODEL_PATH="$VOICE_EVALUATION_ROOT/eres2net"
export SPEAKER_CODE_PATH="$VOICE_EVALUATION_ROOT/3D-Speaker"
```

### Resource check

Before training, the important paths should resolve as follows:

```bash
ls flow_grpo/speech_flowse/ckpts/best.pt.tar
ls "$VOICE_EVALUATION_ROOT/evaluation/DNSMOS/sig_bak_ovr.onnx"
ls "$VOICE_EVALUATION_ROOT/evaluation/DNSMOS/model_v8.onnx"
ls "$VOICE_EVALUATION_ROOT/hubert-base-ls960"
ls "$VOICE_EVALUATION_ROOT/eres2net/pretrained_eres2net_aug.ckpt"
ls "$VOICE_EVALUATION_ROOT/3D-Speaker/speakerlab"
```

## Data Preparation

The paper uses DNS2020 noisy-clean pairs. This repository consumes JSONL manifests containing paired noisy/clean audio paths.

Set the root directory containing `train/noisy`, `train/clean`, `val/noisy`, and `val/clean`:

```bash
export SPEECH_DATA_ROOT=/path/to/DNS_data
```

Build a training manifest with audio paths relative to this root:

```bash
python scripts/build_speech_manifest.py \
  --noisy_dir "$SPEECH_DATA_ROOT/train/noisy" \
  --clean_dir "$SPEECH_DATA_ROOT/train/clean" \
  --data_root "$SPEECH_DATA_ROOT" \
  --output_manifest dataset/speech/train_manifest.jsonl \
  --split train
```

Build the validation manifest similarly:

```bash
python scripts/build_speech_manifest.py \
  --noisy_dir "$SPEECH_DATA_ROOT/val/noisy" \
  --clean_dir "$SPEECH_DATA_ROOT/val/clean" \
  --data_root "$SPEECH_DATA_ROOT" \
  --output_manifest dataset/speech/val_manifest.jsonl \
  --split val
```

The `--data_root` option stores relative paths such as `train/noisy/001.wav` in the manifests. During training and evaluation, these paths are resolved against `SPEECH_DATA_ROOT`. Set this variable in each new shell before running training or evaluation. When moving the dataset to another machine, update `SPEECH_DATA_ROOT` to its new location while keeping the same directory structure; the manifests do not need to be regenerated.

The manifests included in this repository already use relative audio paths. If you omit `--data_root` when generating new manifests, the script stores absolute paths instead; `SPEECH_DATA_ROOT` does not override absolute paths.

## Training

### FlowSE-NFT main experiment

The root-level entry point is configured for the paper's main multi-reward experiment:

```bash
bash train_flowse_nft.sh
```

It uses:

```text
config/nft.py:speech_wotext_multi_reward_rms_scale_tau03_weight211
```

Important defaults include:

```text
rollouts per utterance       K = 24
utterances per epoch             48
reward weights                   DNSMOS : SIM : SBS = 2 : 1 : 1
                                 (equivalent to 0.5 : 0.25 : 0.25)
Conflict-Aware threshold     tau = 0.3
Early-Time Window                [0, 0.25]
training timesteps               32
NFT beta                         1
rollout solver                   32-step SDE
rollout noise scale              0.4
```

The number of local GPUs can be overridden without editing the script:

```bash
NPROC_PER_NODE=8 bash train_flowse_nft.sh
```

You can inspect the final launch command without starting training:

```bash
DRY_RUN=1 bash train_flowse_nft.sh
```

### DNSMOS single-reward experiment

```bash
bash train_dnsmos.sh
```

## Evaluation

Evaluate a FlowSE-NFT checkpoint with the same default paper configuration:

```bash
torchrun --nproc_per_node=1 scripts/evaluation_speech.py \
  --config config/nft.py:speech_wotext_multi_reward_rms_scale_tau03_weight211 \
  --checkpoint /path/to/checkpoint.pt \
  --output_json /path/to/eval.json
```

## Method-to-Code Map

```text
config/nft.py
    Paper/default experiment configuration

flow_grpo/speech_flowse/rewards.py
    DNSMOS, SpeechBERTScore, ERes2Net speaker similarity,
    reward-wise normalization and multi-reward scoring

flow_grpo/speech_backend_adapter.py
    FlowSE/NFT speech backend integration

scripts/train_nft_speech.py
    Online FlowSE-NFT training loop

scripts/evaluation_speech.py
    Speech evaluation

train_flowse_nft.sh
    Main paper training entry point

train_dnsmos.sh
    DNSMOS single-reward training entry point
```


## Acknowledgements

This implementation builds on the ideas and codebases of:

- **DiffusionNFT: Online Diffusion Reinforcement with Forward Process** — https://github.com/NVlabs/DiffusionNFT
- **FlowSE: Efficient and High-Quality Speech Enhancement via Flow Matching** — https://github.com/honee-w/flowse
- **3D-Speaker** — https://github.com/modelscope/3D-Speaker
- **DNSMOS / DNS Challenge** — https://github.com/microsoft/DNS-Challenge

We thank the authors of these projects for releasing their work.

## Citation

If you find this repository useful, please cite the FlowSE-NFT paper. The final BibTeX entry will be added after the paper is publicly available.

```bibtex
@inproceedings{ge2027flowsenft,
  title     = {FlowSE-NFT: Efficient Multi-Reward Post-Training for Flow-Matching Speech Enhancement},
  author    = {Ge, Ziru and Yang, Liusha and Zhang, Junan},
  year      = {2027}
}
```

Please also consider citing the original FlowSE and DiffusionNFT works.

## License

See [`LICENSE`](LICENSE). Third-party models, checkpoints, datasets, and external repositories remain subject to their respective licenses and terms of use.
