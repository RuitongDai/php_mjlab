"""Distill the five E1 teachers into one depth-conditioned 25-DoF student."""

from copy import deepcopy
from dataclasses import fields

from mjlab.managers.event_manager import EventTermCfg
from mjlab.managers.observation_manager import ObservationGroupCfg, ObservationTermCfg
from mjlab.sensor import CameraSensorCfg
from mjlab.utils.noise import UniformNoiseCfg

from ...mdp.commands import PhpMotionCommandCfg
from ...mdp.student import (
    PhpStudentCommandCfg,
    velocity_command,
    anchor_projected_gravity,
    expert_id,
    phase_bin,
    teacher_valid,
    tracking_failure,
)
from ...mdp.depth import PhpDepthImage, randomize_camera
from .env_cfg import (
    E1_BUNDLES,
    E1_DATA_ROOT,
    E1_END_EFFECTOR_BODY_NAMES,
    e1_php_teacher_env_cfg,
)


# Pose and FOV of depth_cam in the original E1 AMP XML. This quaternion is
# MuJoCo's conversion of xyaxes="0 -1 0  0.940 0 0.342" (about 70 deg down).
# Keep the existing PHP image pipeline's 106 x 60 input resolution.
E1_CAMERA_POS = (0.09, 0.0, 0.18)
E1_CAMERA_QUAT = (
    0.696371853264505,
    0.1227446209858446,
    -0.1227446209858446,
    -0.696371853264505,
)
E1_CAMERA_FOVY = 58.0


def e1_php_student_env_cfg(play=False):
    cfg = e1_php_teacher_env_cfg("High-Climb", play)
    source = cfg.commands["motion"]
    kwargs = {
        f.name: deepcopy(getattr(source, f.name)) for f in fields(PhpMotionCommandCfg)
    }
    kwargs.update(
        dataset_dir=str(E1_DATA_ROOT),
        motion_bundles=tuple(E1_BUNDLES.values()),
        end_effector_body_names=E1_END_EFFECTOR_BODY_NAMES,
        require_velocity=True,
        sampling_mode="sequential" if play else "balanced",
    )
    cfg.commands["motion"] = PhpStudentCommandCfg(**kwargs)
    cfg.scene.num_envs = 1 if play else 256
    teacher = cfg.observations.pop("actor")
    teacher.enable_corruption = False
    cfg.observations["teacher"] = teacher
    terms = {
        "projected_gravity": ObservationTermCfg(
            func=anchor_projected_gravity,
            noise=UniformNoiseCfg(n_min=-0.05, n_max=0.05),
        )
    }
    terms.update(
        {
            n: deepcopy(teacher.terms[n])
            for n in ("base_ang_vel", "joint_pos", "joint_vel", "actions")
        }
    )
    terms["velocity_command"] = ObservationTermCfg(func=velocity_command)
    cfg.observations["student_proprio"] = ObservationGroupCfg(
        terms=terms, concatenate_terms=True, enable_corruption=not play
    )
    cfg.observations["student_depth"] = ObservationGroupCfg(
        terms={
            "depth": ObservationTermCfg(
                func=PhpDepthImage,
                params={"corrupt": not play, "gaussian_blur": (3, 1.0)},
            )
        },
        concatenate_terms=True,
        enable_corruption=False,
    )
    for name, func in (
        ("expert_id", expert_id),
        ("phase_bin", phase_bin),
        ("dagger_valid", teacher_valid),
    ):
        cfg.observations[name] = ObservationGroupCfg(
            terms={name: ObservationTermCfg(func=func)},
            concatenate_terms=True,
            enable_corruption=False,
        )
    camera = CameraSensorCfg(
        name="depth_camera",
        parent_body="robot/torso_link",
        pos=E1_CAMERA_POS,
        quat=E1_CAMERA_QUAT,
        fovy=E1_CAMERA_FOVY,
        width=106,
        height=60,
        data_types=("depth",),
        use_textures=False,
        use_shadows=False,
        enabled_geom_groups=(0, 2),
    )
    cfg.scene.sensors = (*cfg.scene.sensors, camera)
    if not play:
        cfg.events["php_camera"] = EventTermCfg(
            func=randomize_camera,
            mode="startup",
            params={"pos": E1_CAMERA_POS, "quat": E1_CAMERA_QUAT},
        )
    for name, threshold in (
        ("anchor_pos", 0.5),
        ("anchor_ori", 0.8),
        ("ee_body_pos", 0.5),
    ):
        cfg.terminations[name].func = tracking_failure
        cfg.terminations[name].params = {
            "kind": name,
            "threshold": threshold,
            "relax": not play,
        }
    return cfg
