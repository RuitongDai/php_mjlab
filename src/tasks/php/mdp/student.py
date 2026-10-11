"""Five-skill sampling and privileged supervision for the PHP student."""

from dataclasses import dataclass
from pathlib import Path

import torch
from mjlab.tasks.tracking import mdp as tracking
from mjlab.managers.scene_entity_config import SceneEntityCfg
from mjlab.utils.lab_api.math import quat_apply_inverse

from ..dataset import BUNDLES, DATA_ROOT, motion_paths
from .commands import PhpMotionCommand, PhpMotionCommandCfg

EE_NAMES = (
    "left_ankle_roll_link",
    "right_ankle_roll_link",
    "left_wrist_yaw_link",
    "right_wrist_yaw_link",
)


class PhpStudentCommand(PhpMotionCommand):
    def __init__(self, cfg, env):
        super().__init__(cfg, env)
        groups = []
        for i, folder in enumerate(BUNDLES.values()):
            groups.extend([i] * len(motion_paths(str(Path(cfg.dataset_dir) / folder))))
        self.clip_experts = torch.tensor(groups, device=self.device)
        if cfg.fixed_skill is not None and cfg.fixed_motion_id is not None:
            if int(self.clip_experts[cfg.fixed_motion_id]) != list(BUNDLES).index(
                cfg.fixed_skill
            ):
                raise ValueError(
                    "fixed_motion_id does not belong to fixed_skill (Student IDs are global across all five bundles)"
                )
        # Equal probability per expert, uniform probability per frame within it.
        lengths = self.motion.motion_lengths.float()
        totals = torch.zeros(5, device=self.device).scatter_add_(
            0, self.clip_experts, lengths
        )
        self.sample_probabilities = lengths / totals[self.clip_experts] / 5

    def _sample(self, env_ids):
        if self.cfg.sampling_mode in ("start", "sequential"):
            if self.cfg.fixed_skill is not None and self.cfg.fixed_motion_id is None:
                selected = torch.where(
                    self.clip_experts == list(BUNDLES).index(self.cfg.fixed_skill)
                )[0]
                self._next_motion_ids[env_ids] = selected[
                    self._next_motion_ids[env_ids] % len(selected)
                ]
                # Track local cursors so a nonzero global offset never changes the sequence.
                ids = self._next_motion_ids[env_ids].clone()
                super()._sample(env_ids)
                self._next_motion_ids[env_ids] = torch.searchsorted(selected, ids) + 1
            else:
                super()._sample(env_ids)
            return
        probs = self.sample_probabilities.clone()
        if self.cfg.fixed_skill is not None:
            probs *= self.clip_experts == list(BUNDLES).index(self.cfg.fixed_skill)
        if self.cfg.fixed_motion_id is not None:
            probs *= (
                torch.arange(len(probs), device=self.device) == self.cfg.fixed_motion_id
            )
        if probs.sum() <= 0:
            raise ValueError("fixed_motion_id does not belong to fixed_skill")
        ids = torch.multinomial(probs, len(env_ids), replacement=True)
        self.motion_ids[env_ids] = ids
        self.time_steps[env_ids] = (
            self.motion.motion_start_idx[ids]
            + (
                torch.rand(len(ids), device=self.device)
                * self.motion.motion_lengths[ids]
            ).long()
        )
        self.variant_ids[env_ids] = (
            torch.rand(len(ids), device=self.device) * self.motion.variant_counts[ids]
        ).long()
        if self.cfg.fixed_variant_id is not None:
            self.variant_ids[env_ids] = self.cfg.fixed_variant_id
        self._episode_active[env_ids] = True


@dataclass(kw_only=True)
class PhpStudentCommandCfg(PhpMotionCommandCfg):
    dataset_dir: str = str(DATA_ROOT)
    require_velocity: bool = True
    fixed_skill: str | None = None
    sampling_mode: str = "balanced"

    def get_motion_paths(self):
        if self.fixed_skill is not None and self.fixed_skill not in BUNDLES:
            raise ValueError(f"fixed_skill must be one of {tuple(BUNDLES)}")
        return tuple(
            p
            for folder in BUNDLES.values()
            for p in motion_paths(str(Path(self.dataset_dir) / folder))
        )

    def build(self, env):
        return PhpStudentCommand(self, env)


def velocity_command(env):
    command = env.command_manager.get_term("motion")
    return command.motion.vel_cmd[command.time_steps]


def anchor_projected_gravity(env):
    command = env.command_manager.get_term("motion")
    return quat_apply_inverse(
        command.robot_anchor_quat_w, command.robot.data.gravity_vec_w
    )


def expert_id(env):
    command = env.command_manager.get_term("motion")
    return command.clip_experts[command.motion_ids, None].float()


def phase_bin(env):
    command = env.command_manager.get_term("motion")
    return (
        torch.searchsorted(command.motion.bin_starts, command.time_steps, right=True)
        - 1
    )[:, None].float()


def tracking_failure(env, kind: str, threshold: float, relax: bool = False):
    if relax:
        threshold *= 1 + min(max((env.common_step_counter - 24_000) / 216_000, 0), 1)
    if kind == "anchor_pos":
        return tracking.bad_anchor_pos(env, "motion", threshold)
    if kind == "anchor_ori":
        return tracking.bad_anchor_ori(
            env, SceneEntityCfg("robot"), "motion", threshold
        )
    if kind == "ee_body_pos":
        return tracking.bad_motion_body_pos(env, "motion", threshold, EE_NAMES)
    raise ValueError(kind)


def teacher_valid(env):
    # Supervision uses the original expert envelope even as student termination relaxes.
    invalid = tracking_failure(env, "anchor_pos", 0.5)
    invalid |= tracking_failure(env, "anchor_ori", 0.8)
    invalid |= tracking_failure(env, "ee_body_pos", 0.5)
    return (~invalid)[:, None].float()
