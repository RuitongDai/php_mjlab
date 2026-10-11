"""Export PHP v2 Student on CPU, without teachers, motion data or a GPU simulator."""

import argparse
from dataclasses import asdict
import hashlib
import json
from pathlib import Path

import numpy as np
import torch
import yaml
from tensordict import TensorDict

from .policy_runtime import PhpOnnxPolicy
from .scene import build_scene


def read_params(path):
    return (
        yaml.load(path.read_text(), Loader=yaml.BaseLoader) if path.is_file() else None
    )


def export(checkpoint_file, output_file):
    from src.tasks.php.config.g1.student_rl_cfg import php_student_runner_cfg
    from src.tasks.php.rl.models import PhpStudentModel

    checkpoint_file = Path(checkpoint_file).expanduser().resolve()
    output_file = Path(output_file).expanduser().resolve()
    if output_file.suffix != ".onnx":
        raise ValueError("--output-file must end in .onnx")
    checkpoint = torch.load(checkpoint_file, map_location="cpu", weights_only=False)
    if checkpoint.get("php_version") != 2:
        raise ValueError(
            "Requires PHP Student v2 checkpoint (11 command dimensions), not a teacher"
        )
    state = checkpoint["actor_state_dict"]
    cfg = asdict(php_student_runner_cfg())
    actor_cfg = cfg["actor"]
    actor_cfg.pop("class_name")
    # Weight shapes allow smaller test/CLI-overridden networks without guessing.
    weights = sorted(
        (int(k.split(".")[1]), v)
        for k, v in state.items()
        if k.startswith("mlp.") and k.endswith(".weight")
    )
    if not weights or weights[0][1].shape[1] != 136 or weights[-1][1].shape[0] != 29:
        raise ValueError("Checkpoint is not a PHP 104-proprio + 32-depth actor")
    actor_cfg["hidden_dims"] = tuple(w.shape[0] for _, w in weights[:-1])
    agent_params = read_params(checkpoint_file.parent / "params/agent.yaml")
    if agent_params is not None:
        saved = agent_params["actor"]
        actor_cfg["activation"] = saved["activation"]
        actor_cfg["obs_normalization"] = saved["obs_normalization"].lower() == "true"
        if (
            saved.get("distribution_cfg", {}).get("class_name")
            != "GaussianDistribution"
        ):
            raise ValueError("Only the PHP Gaussian action distribution is supported")
        actor_cfg["distribution_cfg"]["std_type"] = saved["distribution_cfg"][
            "std_type"
        ]
    obs = TensorDict(
        {
            "student_proprio": torch.zeros(1, 104),
            "student_depth": torch.zeros(1, 1, 58, 87),
        },
        batch_size=[1],
    )
    actor = PhpStudentModel(obs, cfg["obs_groups"], "actor", 29, **actor_cfg)
    actor.load_state_dict(state, strict=True)
    actor.eval()
    _, _, metadata = build_scene()
    env_params = read_params(checkpoint_file.parent / "params/env.yaml")
    if env_params is not None:
        if float(env_params["actions"]["joint_pos"]["scale"]) != 1.0:
            raise ValueError("Checkpoint action scale differs from PHP deployment")
        if (
            float(env_params["sim"]["mujoco"]["timestep"]) != metadata["physics_dt"]
            or int(env_params["decimation"]) != 4
        ):
            raise ValueError("Checkpoint control timing differs from PHP deployment")
        camera = next(
            s for s in env_params["scene"]["sensors"] if s["name"] == "depth_camera"
        )
        for key, name in (
            ("pos", "camera_pos"),
            ("quat", "camera_quat_wxyz"),
            ("fovy", "camera_fovy"),
        ):
            if not np.allclose(
                np.asarray(camera[key], dtype=float), metadata[name], atol=1e-6
            ):
                raise ValueError(f"Checkpoint camera {key} differs from PHP deployment")
        depth = env_params["observations"]["student_depth"]["terms"]["depth"]["params"]
        saved_blur = depth.get("gaussian_blur")
        blur = (
            None
            if saved_blur in (None, "null", "None")
            else [int(saved_blur[0]), float(saved_blur[1])]
        )
        if blur != metadata["gaussian_blur"]:
            raise ValueError(
                "Checkpoint depth blur differs from current PHP deployment"
            )
    else:
        print(
            "[PHP] No params/env.yaml: using current PHP camera/robot/depth defaults."
        )
    clip = None if agent_params is None else agent_params.get("clip_actions")
    metadata["clip_actions"] = None if clip in (None, "null", "None") else float(clip)
    if metadata["clip_actions"] is not None and metadata["clip_actions"] <= 0:
        raise ValueError("clip_actions must be positive")
    metadata["checkpoint"] = checkpoint_file.name
    metadata["checkpoint_sha256"] = hashlib.sha256(
        checkpoint_file.read_bytes()
    ).hexdigest()
    output_file.parent.mkdir(parents=True, exist_ok=True)
    module = actor.as_onnx().cpu().eval()
    torch.onnx.export(
        module,
        module.get_dummy_inputs(),
        str(output_file),
        input_names=module.input_names,
        output_names=module.output_names,
        opset_version=18,
        dynamo=False,
    )
    import onnx

    model = onnx.load(output_file)
    entry = model.metadata_props.add()
    entry.key, entry.value = "php_sim2sim", json.dumps(metadata)
    onnx.checker.check_model(model)
    onnx.save(model, output_file)
    output_file.with_suffix(".json").write_text(json.dumps(metadata, indent=2) + "\n")
    policy = PhpOnnxPolicy(output_file)
    generator = torch.Generator().manual_seed(123)
    with torch.inference_mode():
        for _ in range(3):
            proprio = torch.randn(1, 104, generator=generator)
            depth = torch.rand(1, 1, 58, 87, generator=generator) - 0.5
            expected = module(proprio, depth).numpy()[0]
            actual = policy(proprio.numpy(), depth.numpy())
            np.testing.assert_allclose(actual, expected, atol=2e-5, rtol=2e-4)
    print(f"[OK] Exported {output_file}; CPU PyTorch/ONNX parity passed.")
    return output_file


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint-file", type=Path, required=True)
    parser.add_argument("--output-file", type=Path)
    args = parser.parse_args()
    torch.set_num_threads(2)
    export(
        args.checkpoint_file,
        args.output_file or args.checkpoint_file.with_name("php_student.onnx"),
    )


if __name__ == "__main__":
    main()
