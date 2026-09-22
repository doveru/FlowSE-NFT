import importlib.util
import os
from pathlib import Path

import ml_collections

from flow_nft.speech_paths import resolve_project_path

_PROJECT_ROOT = Path(__file__).resolve().parents[1]
_base_spec = importlib.util.spec_from_file_location("speech_nft_base", Path(__file__).with_name("base.py"))
base = importlib.util.module_from_spec(_base_spec)
_base_spec.loader.exec_module(base)


def _asset_path(env_name, default):
    return str(resolve_project_path(os.environ.get(env_name) or default))


def _speechbert_path(reward):
    return _asset_path("SPEECHBERT_MODEL_PATH", Path(reward.voice_eval_root) / "hubert-base-ls960")


def get_config(name):
    return globals()[name]()


def _resolve_speech_batch_layout(
    n_gpus: int,
    gradient_step_per_epoch: int,
    num_candidates: int,
    num_groups: int = 48,
    max_train_batch_size: int = 3,
):
    if n_gpus <= 0:
        raise ValueError(f"`n_gpus` must be positive, got {n_gpus}.")
    if gradient_step_per_epoch <= 0:
        raise ValueError(
            f"`gradient_step_per_epoch` must be positive, got {gradient_step_per_epoch}."
        )
    if num_candidates <= 0:
        raise ValueError(f"`num_candidates` must be positive, got {num_candidates}.")
    if num_groups <= 0:
        raise ValueError(f"`num_groups` must be positive, got {num_groups}.")
    if max_train_batch_size <= 0:
        raise ValueError(
            f"`max_train_batch_size` must be positive, got {max_train_batch_size}."
        )

    total_sources = num_groups
    bsz = max_train_batch_size
    while True:
        if bsz < 1:
            raise ValueError(
                "Cannot find a speech batch layout that aligns "
                "`num_groups`, `n_gpus`, and `gradient_step_per_epoch`."
            )

        total_sources_per_rollout_batch = n_gpus * bsz
        if total_sources % total_sources_per_rollout_batch == 0:
            rollout_batches_per_epoch = total_sources // total_sources_per_rollout_batch
            if rollout_batches_per_epoch % gradient_step_per_epoch == 0:
                gradient_accumulation_steps = (
                    rollout_batches_per_epoch // gradient_step_per_epoch
                )
                return bsz, rollout_batches_per_epoch, gradient_accumulation_steps
        bsz -= 1


def speech_wotext_pure_nft():
    config = base.get_config()
    config.run_name = "speech_nft_wotext_pure"
    config.logdir = "logs"
    config.save_dir = "logs/nft/speech/wotext_pure"
    config.num_epochs = 500
    config.save_freq = 1
    config.eval_freq = 1
    config.save_best_only = True
    config.best_metric = "val_reward_mean"
    config.best_ckpt_name = "checkpoint-best.pt"
    config.train_log_name = "train.log"
    config.distributed_timeout_minutes = 120
    config.wandb_enabled = True
    config.wandb_project = "flow-grpo"
    config.wandb_entity = ""
    config.wandb_mode = ""
    config.mixed_precision = "fp16"
    config.resume_from = ""
    config.modality = "speech"

    config.speech = speech = ml_collections.ConfigDict()

    speech.model = model = ml_collections.ConfigDict()
    model.init_checkpoint = _asset_path("INIT_CHECKPOINT", "flow_nft/speech_flowse/ckpts/best.pt.tar")
    model.lora = lora = ml_collections.ConfigDict()
    lora.enabled = True
    lora.strategy = "multi_model"
    lora.r = 32  # LoRA rank
    lora.alpha = 64  # LoRA alpha
    lora.dropout = 0.0  # LoRA dropout
    lora.bias = "none"
    lora.init_lora_weights = "gaussian"
    lora.target_modules = ["to_q", "to_k", "to_v", "to_out.0"]
    lora.lora_path = None
    lora.save_adapter_only = True
    model.nnet_conf = nnet_conf = ml_collections.ConfigDict()
    nnet_conf.tokenizer = "pinyin"
    nnet_conf.tokenizer_path = _asset_path("TOKENIZER_PATH", "flow_nft/speech_flowse/Emilia_ZH_EN_pinyin/vocab.txt")
    nnet_conf.audio_drop_prob = 0.0
    nnet_conf.cond_drop_prob = 0.0

    nnet_conf.arch = arch = ml_collections.ConfigDict()
    arch.dim = 1024
    arch.depth = 22
    arch.heads = 16
    arch.ff_mult = 2
    arch.text_dim = 512
    arch.conv_layers = 4
    arch.dropout = 0.0
    arch.checkpoint_activations = False

    nnet_conf.mel_spec = mel_spec = ml_collections.ConfigDict()
    mel_spec.target_sample_rate = 24000
    mel_spec.n_mel_channels = 100
    mel_spec.hop_length = 256  # Mel hop length
    mel_spec.win_length = 1024
    mel_spec.n_fft = 1024
    mel_spec.mel_spec_type = "vocos"

    nnet_conf.vocoder = vocoder = ml_collections.ConfigDict()
    vocoder.is_local = True
    vocoder.local_path = _asset_path("VOCODER_PATH", "flow_nft/speech_flowse/vocos-mel-24khz")

    speech.data = data = ml_collections.ConfigDict()
    data.data_root = _asset_path("SPEECH_DATA_ROOT", "../DNS_noreverb")
    data.train_manifest = _asset_path("TRAIN_MANIFEST", "dataset/speech/train_manifest.jsonl")
    data.val_manifest = _asset_path("VAL_MANIFEST", "dataset/speech/val_manifest.jsonl")
    data.sample_rate = 16000
    data.train_batch_size = 1
    data.eval_batch_size = 2
    data.num_workers = 0
    data.drop_last = True

    speech.rollout = rollout = ml_collections.ConfigDict()
    rollout.num_candidates = 24
    rollout.steps = 32
    rollout.cfg_strength = 0.0
    rollout.solver = "flow"
    rollout.deterministic = False
    rollout.noise_level = 0.4
    rollout.sigma_min = None
    rollout.sigma_max = 1.0
    rollout.time_grid = []
    rollout.deterministic_mask = []
    rollout.collect_trajectory_similarity = False
    rollout.cond_type = "wotext"
    rollout.base_seed = 1234
    rollout.mixed_sampling_enabled = False
    rollout.mixed_group_size = 6
    rollout.mixed_strategy = "progressive"
    rollout.mixed_overlap = True
    rollout.mixed_overlap_step = 1
    rollout.mixed_update_interval = 6
    rollout.mixed_roll_back = True

    # ===== 3.1) Low-STD group filtering =====
    speech.group_filter = group_filter = ml_collections.ConfigDict()
    group_filter.enabled = False
    group_filter.mean_threshold = 3.80
    group_filter.std_threshold = 0.04

    speech.loss = loss = ml_collections.ConfigDict()
    loss.beta_mix = 1.0
    loss.adv_clip_max = 5.0
    loss.adv_weight_mode = "shifted_clip"
    loss.kl_coef = 0.001

    speech.snapshot = snapshot = ml_collections.ConfigDict()
    snapshot.decay_type = 1
    snapshot.track_buffers = True

    speech.train = train = ml_collections.ConfigDict()
    train.learning_rate = 2e-4
    train.adam_beta1 = 0.9  # Adam beta1
    train.adam_beta2 = 0.999  # Adam beta2
    train.adam_weight_decay = 1e-4
    train.adam_epsilon = 1e-8  # Adam epsilon
    train.lr_decay_steps = 800
    train.max_grad_norm = 1.0
    train.gradient_accumulation_steps = 4
    train.num_inner_epochs = 1
    train.rollout_batches_per_epoch = 0
    train.timestep_fraction = 0.99
    train.timestep_window = [0.0, 1.0]
    train.timesteps_per_batch = 0
    train.timestep_grid_steps = 0
    train.nft_target_time = None
    train.reward_branches = []
    train.multi_reward_update_mode = "branch"
    train.reward_conflict_filter = reward_conflict_filter = ml_collections.ConfigDict()
    reward_conflict_filter.enabled = False
    reward_conflict_filter.tau = 0.2
    reward_conflict_filter.snr_eps = 1e-8
    reward_conflict_filter.apply_group_keep_ratio = True
    reward_conflict_filter.post_normalize = True
    reward_conflict_filter.post_normalize_mode = "masked_whiten"
    reward_conflict_filter.post_normalize_eps = 1e-4
    train.global_advantage_std = False
    train.gather_advantages = False
    train.ema = False
    train.ema_decay = 0.9
    train.ema_update_interval = 1
    train.ema_resume_from_checkpoint = True

    speech.layout = layout = ml_collections.ConfigDict()
    layout.n_gpus = int(os.environ.get("WORLD_SIZE", 8))
    layout.gradient_step_per_epoch = 1
    layout.num_groups = 48
    layout.max_train_batch_size = int(data.train_batch_size)

    (
        data.train_batch_size,
        train.rollout_batches_per_epoch,
        train.gradient_accumulation_steps,
    ) = _resolve_speech_batch_layout(
        n_gpus=layout.n_gpus,
        gradient_step_per_epoch=layout.gradient_step_per_epoch,
        num_candidates=int(rollout.num_candidates),
        num_groups=layout.num_groups,
        max_train_batch_size=layout.max_train_batch_size,
    )

    speech.reward = reward = ml_collections.ConfigDict()
    reward.registry = ["dnsmos"]
    reward.eval_registry = []
    reward.weights = ml_collections.ConfigDict(
        {
            "dnsmos": 1.0,
        }
    )
    reward.normalization = "raw_linear"
    reward.batch_zscore_eps = 1e-6
    reward.primary_keys = ml_collections.ConfigDict(
        {
            "dnsmos": "dnsmos_avg",
        }
    )
    reward.raw_scales = ml_collections.ConfigDict(
        {
            "dnsmos": 1.0,
        }
    )
    reward.voice_eval_root = _asset_path("VOICE_EVALUATION_ROOT", "../voice_evaluation")
    reward.device = "train"
    reward.hf_cache_dir = None
    reward.speechbert_model_path = _speechbert_path(reward)
    reward.speaker_model_type = "wavlm"
    reward.speaker_model_path = _asset_path("SPEAKER_MODEL_PATH", Path(reward.voice_eval_root) / "wavlm-base-plus-sv")
    reward.speaker_code_path = None
    reward.tmp_root = None
    reward.allow_remote_hf = False

    speech.eval = eval_conf = ml_collections.ConfigDict()
    eval_conf.limit_batches = 0
    eval_conf.num_candidates = 1
    eval_conf.use_rollout_schedule = True
    eval_conf.use_rollout_deterministic_mask = False
    eval_conf.steps = 32

    return config


def _speech_wotext_pure_timestep_window(
    suffix: str,
    timestep_window: list[float],
    timesteps_per_batch: int = 16,
):
    config = speech_wotext_pure_nft()
    config.run_name = f"speech_nft_wotext_pure_{suffix}"
    config.save_dir = f"logs/nft/speech/wotext_pure_{suffix}"
    config.speech.train.timestep_window = list(timestep_window)
    config.speech.train.timesteps_per_batch = int(timesteps_per_batch)
    config.speech.train.timestep_fraction = 1.0
    return config


def speech_wotext_pure_nft_early32_train64():
    config = _speech_wotext_pure_timestep_window(
        suffix="early32_train64",
        timestep_window=[0.0, 0.5],
        timesteps_per_batch=32,
    )
    config.speech.rollout.steps = 32
    config.speech.train.timestep_grid_steps = 64
    config.save_dir = "logs/nft/speech/wotext_pure_early32_train64"
    return config


def speech_wotext_dnsmos_early16_train32_rollout32_baseline():
    """DNSMOS baseline: original 32-step uniform rollout and early-half NFT training."""
    config = _speech_wotext_pure_timestep_window(
        suffix="early16_train32",
        timestep_window=[0.0, 0.5],
        timesteps_per_batch=16,
    )
    config.speech.rollout.steps = 32
    config.speech.train.timestep_grid_steps = 32
    config.run_name = "speech_nft_dnsmos_early16_train32_rollout32_baseline"
    config.save_dir = "logs/nft/speech/dnsmos_early16_train32_rollout32_baseline"
    return config


def _set_uniform_speech_reward_branches(config):
    branch_specs = [
        ("dnsmos", "dnsmos_ovrl", 1.0),
        ("speaker_similarity", "speaker_similarity", 1.0),
        ("speechbertscore", "speechbertscore", 1.0),
    ]
    timestep_window = [0.0, 0.25]
    timesteps_per_batch = 32
    timestep_grid_steps = 128
    config.speech.train.reward_branches = [
        ml_collections.ConfigDict(
            {
                "name": name,
                "score_section": "raw",
                "metric_key": metric_key,
                "score_scale": 1.0,
                "loss_weight": loss_weight,
                "timestep_window": list(timestep_window),
                "timestep_fraction": 1.0,
                "timesteps_per_batch": timesteps_per_batch,
                "timestep_grid_steps": timestep_grid_steps,
            }
        )
        for name, metric_key, loss_weight in branch_specs
    ]
    config.speech.train.timestep_window = list(timestep_window)
    config.speech.train.timesteps_per_batch = timesteps_per_batch
    config.speech.train.timestep_grid_steps = timestep_grid_steps
    config.speech.train.timestep_fraction = 1.0
    config.speech.train.multi_reward_update_mode = "gd2po"
    config.speech.train.reward_conflict_filter.enabled = True
    config.speech.train.reward_conflict_filter.tau = 0.0
    return config


def speech_wotext_multi_reward():
    config = speech_wotext_pure_nft()
    config.run_name = (
        "speech_nft_wotext_multi_reward_gd2po_tau00_early025_train32_"
        "groups48_candidates24_clip3_equal111"
    )
    config.save_dir = (
        "logs/nft/speech/wotext_multi_reward_gd2po_tau00_early025_train32_"
        "groups48_candidates24_clip3_equal111"
    )
    config.best_metric = "stable_multi_reward"
    config.speech.loss.adv_clip_max = 3.0

    # Keep 48 source groups with 24 candidates each per optimizer update.
    rollout = config.speech.rollout
    train = config.speech.train
    data = config.speech.data
    layout = config.speech.layout
    rollout.num_candidates = 24
    layout.num_groups = 48
    (
        data.train_batch_size,
        train.rollout_batches_per_epoch,
        train.gradient_accumulation_steps,
    ) = _resolve_speech_batch_layout(
        n_gpus=int(layout.n_gpus),
        gradient_step_per_epoch=int(layout.gradient_step_per_epoch),
        num_candidates=int(rollout.num_candidates),
        num_groups=int(layout.num_groups),
        max_train_batch_size=int(layout.max_train_batch_size),
    )

    reward = config.speech.reward
    reward.registry = [
        "dnsmos",
        "speaker_similarity",
        "speechbertscore",
    ]
    reward.eval_registry = list(reward.registry)
    reward.weights = ml_collections.ConfigDict(
        {
            "dnsmos": 1.0,
            "speaker_similarity": 1.0,
            "speechbertscore": 1.0,
        }
    )
    reward.normalization = "batch_std"
    reward.primary_keys = ml_collections.ConfigDict(
        {
            "dnsmos": "dnsmos_ovrl",
            "speaker_similarity": "speaker_similarity",
            "speechbertscore": "speechbertscore",
        }
    )
    reward.raw_scales = ml_collections.ConfigDict(
        {
            "dnsmos": 1.0,
            "speaker_similarity": 1.0,
            "speechbertscore": 1.0,
        }
    )
    reward.speechbert_model_path = _speechbert_path(reward)
    reward.speaker_model_type = os.environ.get("SPEAKER_MODEL_TYPE", "eres2net")
    reward.speaker_model_path = os.environ.get(
        "SPEAKER_MODEL_PATH",
        str(Path(reward.voice_eval_root) / "eres2net"),
    )
    reward.speaker_code_path = os.environ.get(
        "SPEAKER_CODE_PATH",
        str(Path(reward.voice_eval_root) / "3D-Speaker"),
    )

    return _set_uniform_speech_reward_branches(config)


def speech_wotext_multi_reward_no_post_norm():
    """Use the filtered aggregate advantage without a second normalization."""
    config = speech_wotext_multi_reward()
    config.run_name = (
        "speech_nft_wotext_multi_reward_gd2po_tau00_early025_train32_"
        "groups48_candidates24_clip5_weight211_post_none"
    )
    config.save_dir = (
        "logs/nft/speech/wotext_multi_reward_gd2po_tau00_early025_train32_"
        "groups48_candidates24_clip5_weight211_post_none"
    )
    config.speech.loss.adv_clip_max = 5.0
    config.speech.reward.weights.dnsmos = 2.0
    config.speech.reward.weights.speaker_similarity = 1.0
    config.speech.reward.weights.speechbertscore = 1.0
    branch_loss_weights = {
        "dnsmos": 2.0,
        "speaker_similarity": 1.0,
        "speechbertscore": 1.0,
    }
    for branch in config.speech.train.reward_branches:
        branch.loss_weight = branch_loss_weights[str(branch.name)]
    config.speech.train.reward_conflict_filter.post_normalize = False
    config.speech.train.reward_conflict_filter.post_normalize_mode = "none"
    return config


def speech_wotext_multi_reward_rms_scale_tau03_weight211():
    """RMS-scaled GD2PO with tau=0.3, clip=5, and DNSMOS-heavy 2:1:1 weights."""
    config = speech_wotext_multi_reward_no_post_norm()
    config.run_name = (
        "speech_nft_wotext_multi_reward_gd2po_tau03_early025_train32_"
        "groups48_candidates24_clip5_weight211_post_rms_scale"
    )
    config.save_dir = (
        "logs/nft/speech/wotext_multi_reward_gd2po_tau03_early025_train32_"
        "groups48_candidates24_clip5_weight211_post_rms_scale"
    )
    config.speech.train.reward_conflict_filter.tau = 0.3
    config.speech.train.reward_conflict_filter.post_normalize = True
    config.speech.train.reward_conflict_filter.post_normalize_mode = "rms_scale"
    return config


def speech_wotext_multi_reward_early32_train64():
    config = speech_wotext_multi_reward()
    config.run_name = "speech_nft_wotext_multi_reward_gd2po_tau00_early32_train64_clip3_equal111"
    config.save_dir = "logs/nft/speech/wotext_multi_reward_gd2po_tau00_early32_train64_clip3_equal111"

    config.speech.rollout.steps = 32
    config.speech.train.timestep_window = [0.0, 0.5]
    config.speech.train.timesteps_per_batch = 32
    config.speech.train.timestep_grid_steps = 64
    for branch in config.speech.train.reward_branches:
        branch.timestep_window = [0.0, 0.5]
        branch.timesteps_per_batch = 32
        branch.timestep_grid_steps = 64

    return config
