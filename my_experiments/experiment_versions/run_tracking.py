import json
import os
import shutil
import time


class RunTracker:
    """Tracks run directories, status transitions, and lightweight metadata."""

    STATUSES = ("active", "completed", "interrupted", "failed")

    def __init__(self, experiment_dir, run_name, script_name, purpose, runs_group=None):
        self.experiment_dir = os.path.abspath(experiment_dir)
        self.run_name = run_name
        self.script_name = script_name
        self.purpose = purpose
        self.runs_group = None
        if runs_group is not None:
            g = str(runs_group).strip().strip("/\\")
            if g:
                self.runs_group = g

        self.runs_root = os.path.join(self.experiment_dir, "training_runs")
        if self.runs_group:
            self.runs_root = os.path.join(self.runs_root, self.runs_group)
        self.status_dirs = {
            status: os.path.join(self.runs_root, status) for status in self.STATUSES
        }
        for directory in self.status_dirs.values():
            os.makedirs(directory, exist_ok=True)

        self.run_dir = os.path.join(self.status_dirs["active"], self.run_name)
        os.makedirs(self.run_dir, exist_ok=True)

        self.checkpoint_dir = os.path.join(self.run_dir, "checkpoints")
        self.log_dir = os.path.join(self.run_dir, "logs")
        self.tensorboard_dir = os.path.join(self.run_dir, "tensorboard")
        os.makedirs(self.checkpoint_dir, exist_ok=True)
        os.makedirs(self.log_dir, exist_ok=True)
        os.makedirs(self.tensorboard_dir, exist_ok=True)

        self.run_info_path = os.path.join(self.run_dir, "run_info.json")
        self.index_path = os.path.join(self.runs_root, "runs_index.json")

        self.info = {
            "run_name": self.run_name,
            "status": "active",
            "purpose": self.purpose,
            "script_name": self.script_name,
            "runs_group": self.runs_group,
            "experiment_dir": self.experiment_dir,
            "started_at": self._now(),
            "ended_at": None,
            "base_checkpoint": None,
            "final_steps": None,
            "artifacts": {},
            "notes": None,
            "error_message": None,
            "directories": self._directory_snapshot(),
        }
        self._write_run_info()
        self._update_index()

    def path_for(self, *parts):
        return os.path.join(self.run_dir, *parts)

    def record_base_checkpoint(self, checkpoint_path):
        self.info["base_checkpoint"] = os.path.abspath(checkpoint_path)
        self._write_run_info()
        self._update_index()

    def finalize(
        self,
        status,
        final_steps=None,
        artifacts=None,
        notes=None,
        error_message=None,
    ):
        if status not in self.STATUSES:
            raise ValueError(f"Unsupported run status: {status}")

        self.info["status"] = status
        self.info["ended_at"] = self._now()
        self.info["final_steps"] = final_steps
        self.info["notes"] = notes
        self.info["error_message"] = error_message
        if artifacts:
            self.info["artifacts"] = dict(artifacts)

        target_dir = os.path.join(self.status_dirs[status], self.run_name)
        if os.path.abspath(target_dir) != os.path.abspath(self.run_dir):
            if os.path.exists(target_dir):
                target_dir = f"{target_dir}_{int(time.time())}"
            shutil.move(self.run_dir, target_dir)
            self.run_dir = target_dir
            self.run_info_path = os.path.join(self.run_dir, "run_info.json")
            self.checkpoint_dir = os.path.join(self.run_dir, "checkpoints")
            self.log_dir = os.path.join(self.run_dir, "logs")
            self.tensorboard_dir = os.path.join(self.run_dir, "tensorboard")

        self.info["directories"] = self._directory_snapshot()
        self._write_run_info()
        self._update_index()
        return self.run_dir

    def _directory_snapshot(self):
        return {
            "run_dir": self.run_dir,
            "checkpoints": self.checkpoint_dir,
            "logs": self.log_dir,
            "tensorboard": self.tensorboard_dir,
        }

    def _write_run_info(self):
        with open(self.run_info_path, "w", encoding="utf-8") as handle:
            json.dump(self.info, handle, indent=2, ensure_ascii=False)

    def _update_index(self):
        if os.path.exists(self.index_path):
            with open(self.index_path, "r", encoding="utf-8") as handle:
                index = json.load(handle)
        else:
            index = {}
        index[self.run_name] = {
            "status": self.info["status"],
            "purpose": self.info["purpose"],
            "script_name": self.info["script_name"],
            "runs_group": self.info.get("runs_group"),
            "started_at": self.info["started_at"],
            "ended_at": self.info["ended_at"],
            "final_steps": self.info["final_steps"],
            "base_checkpoint": self.info["base_checkpoint"],
            "run_dir": self.run_dir,
        }
        with open(self.index_path, "w", encoding="utf-8") as handle:
            json.dump(index, handle, indent=2, ensure_ascii=False)

    @staticmethod
    def _now():
        return time.strftime("%Y-%m-%d %H:%M:%S")
