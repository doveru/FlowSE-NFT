"""Configuration entry-point checks without optional training dependencies."""

import ast
from pathlib import Path
from types import SimpleNamespace


CONFIG = Path(__file__).resolve().parents[1] / "config" / "nft.py"
BASELINE = "speech_wotext_dnsmos_early16_train32_rollout32_baseline"


def test_retained_config_entrypoints_and_dependencies():
    tree = ast.parse(CONFIG.read_text())
    names = {node.name for node in tree.body if isinstance(node, ast.FunctionDef)}
    assert {name for name in names if name.startswith("speech_wotext_dnsmos_")} == {BASELINE}
    assert len([name for name in names if name.startswith("speech_")]) == 7
    assert "speech_wotext_timestep_diagnostic" not in names
    assert "speech_wotext_multi_reward_rms_scale" not in names
    assert "speech_wotext_multi_reward_masked_whiten" not in names
    assert "speech_wotext_multi_reward_rms_scale_tau03_weight211" in names
    for suffix in ("all16", "early16", "late16", "early16_train32"):
        assert "speech_wotext_pure_nft_" + suffix not in names
    for node in ast.walk(tree):
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Name):
            if node.func.id.startswith(("speech_wotext_", "_speech_wotext_")):
                assert node.func.id in names


def test_baseline_keeps_original_parent_overrides():
    tree = ast.parse(CONFIG.read_text())
    selected = [node for node in tree.body if isinstance(node, ast.FunctionDef)
                and node.name in {BASELINE, "_speech_wotext_pure_timestep_window"}]
    base = SimpleNamespace(speech=SimpleNamespace(
        train=SimpleNamespace(), rollout=SimpleNamespace()))
    namespace = {"speech_wotext_pure_nft": lambda: base}
    exec(compile(ast.Module(body=selected, type_ignores=[]), str(CONFIG), "exec"), namespace)
    config = namespace[BASELINE]()
    assert config is base
    assert config.speech.train.timestep_window == [0.0, 0.5]
    assert config.speech.train.timesteps_per_batch == 16
    assert config.speech.train.timestep_grid_steps == 32
    assert config.speech.train.timestep_fraction == 1.0
    assert config.speech.rollout.steps == 32
    assert config.run_name == "speech_nft_dnsmos_early16_train32_rollout32_baseline"
    assert config.save_dir == "logs/nft/speech/dnsmos_early16_train32_rollout32_baseline"
