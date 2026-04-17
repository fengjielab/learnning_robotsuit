"""Rebuild stage-grouped training run views without moving canonical run folders."""

from __future__ import annotations

import shutil
from pathlib import Path


SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_DIR = SCRIPT_DIR.parent
TRAINING_RUNS_DIR = PROJECT_DIR / "training_runs"
BY_STAGE_DIR = TRAINING_RUNS_DIR / "by_stage"
STATUSES = ("active", "completed", "interrupted", "failed")


def classify_run(run_name: str) -> str:
    if run_name.startswith("stage_a_cage_"):
        return "stage_a"
    if run_name.startswith("stage_b_grasp_"):
        return "stage_b"
    if run_name.startswith("stage_c_lift_"):
        return "stage_c"
    if run_name.startswith("stage2_local_grasp_"):
        return "legacy_stage2"
    if run_name.startswith("staged_pipeline_"):
        return "pipeline"
    return "misc"


def ensure_stage_dirs() -> None:
    for stage_name in ("stage_a", "stage_b", "stage_c", "pipeline", "legacy_stage2", "misc"):
        for status in STATUSES:
            (BY_STAGE_DIR / stage_name / status).mkdir(parents=True, exist_ok=True)


def clear_existing_views() -> None:
    if BY_STAGE_DIR.exists():
        shutil.rmtree(BY_STAGE_DIR)


def rebuild_views() -> list[str]:
    clear_existing_views()
    ensure_stage_dirs()

    created = []
    for status in STATUSES:
        status_dir = TRAINING_RUNS_DIR / status
        if not status_dir.exists():
            continue
        for entry in sorted(status_dir.iterdir()):
            if not entry.is_dir():
                continue
            stage_name = classify_run(entry.name)
            link_path = BY_STAGE_DIR / stage_name / status / entry.name
            link_path.symlink_to(entry)
            created.append(str(link_path))
    return created


def main() -> None:
    created = rebuild_views()
    print(f"重建完成，共创建 {len(created)} 个阶段视图链接。")
    print(f"输出目录：{BY_STAGE_DIR}")


if __name__ == "__main__":
    main()
