"""Load PPO from a run directory by training_config.json (e.g. full A→B→C pipeline)."""

from __future__ import annotations

import json
import os

from stable_baselines3 import PPO


def resolve_training_run_dir(model_path: str) -> str:
    """Run root (contains training_config.json), even when model_path is under checkpoints/."""
    d = os.path.dirname(os.path.abspath(model_path))
    if os.path.basename(d) == "checkpoints":
        return os.path.dirname(d)
    return d


def load_policy_from_training_run(model_path: str):
    """
    Load PPO from ``model_path`` using ``algorithm`` in that run's training_config.json.
    Defaults to PPO when config is missing (older runs).
    """
    run_dir = resolve_training_run_dir(model_path)
    config_path = os.path.join(run_dir, "training_config.json")
    algorithm = "PPO"
    if os.path.isfile(config_path):
        with open(config_path, "r", encoding="utf-8") as handle:
            raw_algo = json.load(handle).get("algorithm", "PPO")
            algorithm = "PPO" if raw_algo is None else str(raw_algo).upper()

    if algorithm == "PPO":
        return PPO.load(model_path)
    raise ValueError(f"Unsupported algorithm in {config_path!r}: {algorithm!r}")
