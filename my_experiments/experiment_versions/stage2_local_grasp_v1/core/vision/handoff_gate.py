"""Automatic handoff gating for entering the Stage A close-range policy."""

from __future__ import annotations


class StageAHandoffGate:
    """Tracks whether the robot is stably inside the Stage A pregrasp handoff region."""

    def __init__(
        self,
        xy_threshold: float = 0.025,
        vertical_threshold: float = 0.028,
        open_threshold: float = 0.03,
        stable_frames: int = 8,
    ):
        self.xy_threshold = float(xy_threshold)
        self.vertical_threshold = float(vertical_threshold)
        self.open_threshold = float(open_threshold)
        self.stable_frames = int(stable_frames)
        self.consecutive_hits = 0

    def reset(self):
        self.consecutive_hits = 0

    def evaluate(
        self,
        grasp_xy_dist: float,
        vertical_error: float,
        gripper_width: float,
        any_table_contact: bool,
        any_contact: bool,
        grasped: bool,
        update_streak: bool = False,
    ) -> dict[str, object]:
        reasons: list[str] = []

        if grasp_xy_dist >= self.xy_threshold:
            reasons.append("xy_too_far")
        if abs(vertical_error) >= self.vertical_threshold:
            reasons.append("z_too_far")
        if gripper_width <= self.open_threshold:
            reasons.append("gripper_not_open")
        if any_table_contact:
            reasons.append("table_contact")
        if any_contact or grasped:
            reasons.append("already_touching_object")

        ready_now = not reasons
        if update_streak:
            self.consecutive_hits = self.consecutive_hits + 1 if ready_now else 0

        can_handoff = ready_now and self.consecutive_hits >= self.stable_frames
        return {
            "ready_now": bool(ready_now),
            "can_handoff_to_stage_a": bool(can_handoff),
            "consecutive_hits": int(self.consecutive_hits),
            "stable_frames_required": int(self.stable_frames),
            "failure_reasons": reasons,
        }
