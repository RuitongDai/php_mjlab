import json
from pathlib import Path

import torch
import yaml
from mjlab.rl import MjlabOnPolicyRunner

from ..dataset import BUNDLES, VELOCITY_COMMAND_DIM, VELOCITY_COMMAND_NAMES


class PhpStudentRunner(MjlabOnPolicyRunner):
    def __init__(self, env, train_cfg, log_dir=None, device="cpu"):
        super().__init__(env, train_cfg, log_dir, device)
        checkpoints = train_cfg.get("teacher_checkpoint_files", ())
        if checkpoints:
            self.load_teachers(checkpoints)

    def load_teachers(self, checkpoints):
        if len(checkpoints) != 5:
            raise ValueError(
                "Supply exactly five teacher checkpoints: " + ", ".join(BUNDLES)
            )
        paths = [Path(p).expanduser().resolve() for p in checkpoints]
        if len(set(paths)) != 5:
            raise ValueError("The five teachers must have distinct checkpoint paths")
        for i, (skill, path) in enumerate(zip(BUNDLES, paths)):
            env_file = path.parent / "params" / "env.yaml"
            if env_file.is_file():
                # BaseLoader treats Python YAML tags as plain data; no objects execute.
                metadata = yaml.load(env_file.read_text(), Loader=yaml.BaseLoader)
                folder = Path(metadata["commands"]["motion"]["dataset_dir"]).name
                scale = metadata["actions"]["joint_pos"]["scale"]
                if (
                    folder != BUNDLES[skill]
                    or not isinstance(scale, str)
                    or float(scale) != 1.0
                ):
                    raise ValueError(
                        f"Teacher {skill} metadata does not match its skill/action scale: {env_file}"
                    )
            else:
                print(
                    f"[PHP] No params/env.yaml for {skill}; checkpoint architecture is checked, skill identity relies on supplied order."
                )
            checkpoint = torch.load(path, map_location=self.device, weights_only=False)
            if "php_version" in checkpoint:
                raise ValueError(
                    f"Expected a teacher checkpoint, received a student: {path}"
                )
            state = dict(checkpoint["actor_state_dict"])
            # Same compatibility migration as MJLab's base checkpoint loader.
            for old, new in (
                ("std", "distribution.std_param"),
                ("log_std", "distribution.log_std_param"),
            ):
                if old in state:
                    state[new] = state.pop(old)
            self.alg.teacher.experts[i].load_state_dict(state, strict=True)
            self.alg.teacher.loaded[i] = True
            print(f"[PHP] Frozen teacher {i} ({skill}): {path}")

    def learn(self, num_learning_iterations, init_at_random_ep_len=False):
        if not self.alg.teacher.loaded.all():
            raise ValueError(
                'PHP training requires --agent.teacher-checkpoint-files=\'("path0", "path1", "path2", "path3", "path4")\', or a full student resume checkpoint'
            )
        return super().learn(num_learning_iterations, init_at_random_ep_len)

    def load(self, path, load_cfg=None, strict=True, map_location=None):
        infos = super().load(path, load_cfg, strict, map_location or self.device)
        if load_cfg is None or load_cfg.get("iteration"):
            # Base checkpoints store the last completed index. Resume at the next
            # update; do not repeat its log/checkpoint index or restart the ramp.
            self.current_learning_iteration = self.alg.num_updates
        return infos

    def export_policy_to_onnx(self, path, filename="policy.onnx", verbose=False):
        super().export_policy_to_onnx(path, filename, verbose)
        robot = self.env.unwrapped.scene["robot"]
        depth_params = (
            self.env.unwrapped.cfg.observations["student_depth"].terms["depth"].params
        )
        blur = depth_params.get("gaussian_blur", (3, 1.0))
        metadata = {
            "format": "php-depth-student-v2",
            "expert_order": list(BUNDLES),
            "inputs": {
                "obs": [1, 93 + VELOCITY_COMMAND_DIM],
                "student_depth": [1, 1, 58, 87],
            },
            "proprio_order": [
                "projected_gravity:3",
                "base_ang_vel:3",
                "joint_pos_rel:29",
                "joint_vel_rel:29",
                "last_action:29",
                f"vel_cmd:{VELOCITY_COMMAND_DIM}",
            ],
            "velocity_command_encoding": "one_hot",
            "velocity_command_names": list(VELOCITY_COMMAND_NAMES),
            "depth": {
                "raw_hw": [60, 106],
                "crop_tblr": [2, 0, 4, 4],
                "gaussian_blur": None
                if blur is None
                else {
                    "kernel_size": blur[0],
                    "sigma": blur[1],
                    "padding": "reflect",
                    "stage": "after_crop_before_resize",
                },
                "resize": "bicubic_antialias_58x87",
                "input_measurement": "optical_axis_metres",
                "range_m": [0.3, 3.0],
                "normalize": "(depth - 0.3) / 2.7 - 0.5",
                "training_lag_frames": [3, 4],
            },
            "joint_names": list(robot.joint_names),
            "projected_gravity_frame": "torso_link",
            "action_scale": 1.0,
            "default_joint_pos": robot.data.default_joint_pos[0].cpu().tolist(),
            "note": "Export consumes preprocessed depth and proprioception; camera preprocessing and command generation run outside ONNX.",
        }
        (Path(path) / (Path(filename).stem + ".json")).write_text(
            json.dumps(metadata, indent=2) + "\n"
        )
