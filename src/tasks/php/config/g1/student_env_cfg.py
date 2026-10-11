"""All five PHP experts distilled into a depth-conditioned student."""

from copy import deepcopy
from dataclasses import fields

import torch
from mjlab.managers.event_manager import EventTermCfg
from mjlab.managers.observation_manager import ObservationGroupCfg, ObservationTermCfg
from mjlab.sensor import CameraSensorCfg
from mjlab.utils.noise import UniformNoiseCfg

from src.assets.robots.unitree_g1.g1_sphere_hand_constants import (
    get_g1_sphere_hand_spec,
)
from ...dataset import BUNDLES, DATA_ROOT
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
from ...mdp.depth import PhpDepthImage, FOVY, camera_quaternion, randomize_camera
from .env_cfg import php_teacher_env_cfg


def student_robot_spec():
    spec = get_g1_sphere_hand_spec()
    # The original sensor omits torso/head meshes, which enclose the camera.
    for name in ("torso_link", "head_link", "logo_link"):
        body = spec.body(name)
        if body is not None:
            for geom in body.geoms:
                if geom.group == 2:
                    geom.group = 4
    return spec


def php_student_env_cfg(play=False):
    cfg = php_teacher_env_cfg("High-Climb", play)
    source = cfg.commands["motion"]
    kwargs = {
        f.name: deepcopy(getattr(source, f.name)) for f in fields(PhpMotionCommandCfg)
    }
    kwargs.update(
        dataset_dir=str(DATA_ROOT),
        require_velocity=True,
        sampling_mode="sequential" if play else "balanced",
    )
    cfg.commands["motion"] = PhpStudentCommandCfg(**kwargs)
    if all((DATA_ROOT / b).is_dir() for b in BUNDLES.values()):
        cfg.commands["motion"].bind_scene_entities(cfg.scene)
    cfg.scene.num_envs = 1 if play else 256
    cfg.scene.entities["robot"].spec_fn = student_robot_spec
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
        pos=(0.01, 0.01, 0.44),
        quat=tuple(camera_quaternion(torch.tensor([1.0, 27.0, 1.0])).tolist()),
        fovy=FOVY,
        width=106,
        height=60,
        data_types=("depth",),
        use_textures=False,
        use_shadows=False,
        enabled_geom_groups=(0, 2),
    )
    cfg.scene.sensors = (*cfg.scene.sensors, camera)
    if not play:
        cfg.events["php_camera"] = EventTermCfg(func=randomize_camera, mode="startup")
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
