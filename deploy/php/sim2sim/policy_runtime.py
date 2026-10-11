"""PHP-specific observation, depth and deterministic ONNX inference contract."""

from collections import deque
import json
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

COMMAND_NAMES = (
    "stand",
    "forward",
    "left_45",
    "left_90",
    "right_45",
    "right_90",
    "run_forward",
    "run_left_45",
    "run_left_90",
    "run_right_45",
    "run_right_90",
)
INPUTS = {"obs": [1, 104], "student_depth": [1, 1, 58, 87]}
PROPRIO_ORDER = [
    "projected_gravity:3",
    "base_ang_vel:3",
    "joint_pos_rel:29",
    "joint_vel_rel:29",
    "last_action:29",
    "vel_cmd:11",
]


def one_hot(command):
    value = np.zeros(11, dtype=np.float32)
    value[COMMAND_NAMES.index(command)] = 1
    return value


def build_proprioception(
    torso_rotation,
    angular_velocity,
    joint_pos,
    joint_vel,
    previous_action,
    default_joint_pos,
    command,
):
    """Gravity uses torso; gyro uses the pelvis IMU, exactly as in training."""
    gravity = np.asarray(torso_rotation).reshape(3, 3).T @ [0.0, 0.0, -1.0]
    observation = np.concatenate(
        (
            gravity,
            angular_velocity,
            np.asarray(joint_pos) - default_joint_pos,
            joint_vel,
            previous_action,
            command,
        )
    ).astype(np.float32)
    if observation.shape != (104,) or not np.isfinite(observation).all():
        raise ValueError("Expected finite 104-D PHP observation")
    command = observation[-11:]
    if not np.all((command == 0) | (command == 1)) or command.sum() != 1:
        raise ValueError("PHP command must be one-hot, not continuous velocity")
    return observation


@torch.inference_mode()
def preprocess_depth_z(depth_z, gaussian_blur=(3, 1.0)):
    """Native optical Z -> exact corruption-free training depth preprocessing."""
    raw = np.asarray(depth_z, dtype=np.float32)
    if raw.shape != (60, 106):
        raise ValueError("Raw depth must have shape (60, 106)")
    depth = torch.from_numpy(raw.copy())
    depth = torch.where(torch.isfinite(depth) & (depth > 0), depth, 3.0).clamp(0.3, 3.0)
    depth = depth[None, None, 2:, 4:-4]
    if gaussian_blur is not None:
        size, sigma = gaussian_blur
        if size <= 0 or size % 2 == 0 or not np.isfinite(sigma) or sigma <= 0:
            raise ValueError("Invalid Gaussian blur")
        padding = size // 2
        if padding >= min(depth.shape[-2:]):
            raise ValueError("Gaussian kernel too large")
        coords = torch.arange(size, dtype=depth.dtype) - padding
        kernel = torch.exp(-0.5 * (coords / sigma).square())
        kernel /= kernel.sum()
        kernel = (kernel[:, None] * kernel[None, :])[None, None]
        depth = F.conv2d(F.pad(depth, (padding,) * 4, mode="reflect"), kernel)
    depth = F.interpolate(
        depth, (58, 87), mode="bicubic", align_corners=False, antialias=True
    ).clamp(max=3.0)
    depth = torch.where(depth < 0.15, 3.0, depth)
    return ((depth - 0.3) / 2.7 - 0.5).numpy().copy()


class DepthHistory:
    def __init__(self, lag="random", seed=0):
        if lag not in ("random", "0", "3", "4"):
            raise ValueError("Depth lag must be random, 0, 3 or 4")
        self.lag = lag
        self.rng = np.random.default_rng(seed)
        self.frames = deque(maxlen=6)

    def reset(self):
        self.frames.clear()

    def push(self, frame):
        if not self.frames:
            self.frames.extend(frame.copy() for _ in range(6))
        else:
            self.frames.append(frame.copy())
        lag = int(self.rng.integers(3, 5)) if self.lag == "random" else int(self.lag)
        return self.frames[-1 - lag].copy()


class PhpOnnxPolicy:
    def __init__(self, path, provider="cpu"):
        import onnxruntime as ort

        path = Path(path).expanduser().resolve()
        providers = ["CPUExecutionProvider"]
        if provider == "cuda":
            if "CUDAExecutionProvider" not in ort.get_available_providers():
                raise RuntimeError("CUDA ONNX Runtime is unavailable")
            providers.insert(0, "CUDAExecutionProvider")
        elif provider != "cpu":
            raise ValueError("provider must be cpu or cuda")
        options = ort.SessionOptions()
        options.intra_op_num_threads = 2
        self.session = ort.InferenceSession(
            str(path), sess_options=options, providers=providers
        )
        embedded = self.session.get_modelmeta().custom_metadata_map.get("php_sim2sim")
        if embedded is None:
            raise ValueError("Missing PHP deployment metadata; use export_student_onnx")
        self.metadata = json.loads(embedded)
        m = self.metadata
        if m.get("format") != "php-sim2sim-v1" or m.get("php_version") != 2:
            raise ValueError("Unsupported PHP deployment version")
        if m.get("velocity_command_names") != list(COMMAND_NAMES):
            raise ValueError("PHP velocity command order differs")
        if m.get("proprio_order") != PROPRIO_ORDER:
            raise ValueError("PHP proprioception order differs")
        if {v.name: v.shape for v in self.session.get_inputs()} != INPUTS:
            raise ValueError(
                "PHP ONNX inputs must be obs(1,104) and student_depth(1,1,58,87)"
            )
        if {v.name: v.shape for v in self.session.get_outputs()} != {
            "actions": [1, 29]
        }:
            raise ValueError("PHP ONNX output must be actions(1,29)")
        self.default_joint_pos = np.asarray(m["default_joint_pos"], dtype=np.float32)
        if (
            self.default_joint_pos.shape != (29,)
            or not np.isfinite(self.default_joint_pos).all()
        ):
            raise ValueError("Invalid default joint positions")
        if m["action_scale"] != 1.0:
            raise ValueError("PHP requires action scale 1.0")

    def __call__(self, obs, depth):
        inputs = {
            "obs": np.asarray(obs, dtype=np.float32).reshape(1, 104),
            "student_depth": np.asarray(depth, dtype=np.float32),
        }
        if inputs["student_depth"].shape != (1, 1, 58, 87):
            raise ValueError("Incorrect processed depth shape")
        if not all(np.isfinite(v).all() for v in inputs.values()):
            raise FloatingPointError("Policy inputs contain NaN/Inf")
        result = self.session.run(["actions"], inputs)[0]
        if result.shape != (1, 29) or not np.isfinite(result).all():
            raise FloatingPointError("Invalid policy output")
        return result[0].copy()

    def joint_target(self, action):
        return self.default_joint_pos + action
