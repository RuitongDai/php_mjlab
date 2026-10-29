"""Play PHP teachers or the depth student in MJLab."""

# ruff: noqa: E402 -- prefer this checkout over other editable src packages.

import os
import sys
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Literal

_REPO_ROOT = Path(__file__).resolve().parents[1]
if str(_REPO_ROOT) in sys.path:
    sys.path.remove(str(_REPO_ROOT))
sys.path.insert(0, str(_REPO_ROOT))

import torch
import tyro
from mjlab.envs import ManagerBasedRlEnv
from mjlab.rl import MjlabOnPolicyRunner, RslRlVecEnvWrapper
from mjlab.tasks.registry import list_tasks, load_env_cfg, load_rl_cfg, load_runner_cls
from mjlab.utils.torch import configure_torch_backends
from mjlab.utils.wrappers import VideoRecorder
from mjlab.viewer import NativeMujocoViewer


@dataclass(frozen=True)
class PlayConfig:
    checkpoint_file: str | None = None
    agent: Literal["trained", "zero", "random"] = "trained"
    dataset_dir: str | None = None
    num_envs: int = 1
    device: str | None = None
    viewer: Literal["auto", "native", "viser"] = "auto"
    motion_id: int | None = None
    terrain_variant: int = 0
    php_skill: Literal["Locomotion", "Low-Step", "High-Step", "Low-Climb", "High-Climb"] | None = None
    no_terminations: bool = False
    video: bool = False
    video_length: int = 200


def configure_environment(task_id: str, cfg: PlayConfig):
    env_cfg = load_env_cfg(task_id, play=True)
    if cfg.num_envs < 1 or cfg.terrain_variant < 0:
        raise ValueError("num-envs must be positive and terrain-variant nonnegative")
    if cfg.motion_id is not None and cfg.motion_id < 0:
        raise ValueError("motion-id must be nonnegative")
    env_cfg.scene.num_envs = cfg.num_envs
    motion = env_cfg.commands["motion"]
    if cfg.dataset_dir is not None:
        motion.dataset_dir = str(Path(cfg.dataset_dir).expanduser().resolve())
    motion.fixed_motion_id = cfg.motion_id
    motion.fixed_variant_id = cfg.terrain_variant
    if cfg.php_skill is not None:
        if not hasattr(motion, "fixed_skill"):
            raise ValueError("--php-skill requires Unitree-G1-PHP-Student")
        motion.fixed_skill = cfg.php_skill
    motion.bind_scene_entities(env_cfg.scene)
    if cfg.no_terminations:
        env_cfg.terminations = {}
    return env_cfg


def run_play(task_id: str, cfg: PlayConfig):
    configure_torch_backends()
    checkpoint = None
    if cfg.agent == "trained":
        if cfg.checkpoint_file is None:
            raise ValueError("--checkpoint-file is required for a trained policy")
        checkpoint = Path(cfg.checkpoint_file).expanduser().resolve()
        if not checkpoint.is_file():
            raise FileNotFoundError(checkpoint)
    if cfg.video and checkpoint is None:
        raise ValueError("Video recording requires a trained checkpoint")
    device = cfg.device or ("cuda:0" if torch.cuda.is_available() else "cpu")
    env_cfg = configure_environment(task_id, cfg)
    agent_cfg = load_rl_cfg(task_id)
    env = ManagerBasedRlEnv(
        cfg=env_cfg, device=device, render_mode="rgb_array" if cfg.video else None
    )
    try:
        if cfg.video:
            env = VideoRecorder(
                env,
                video_folder=checkpoint.parent / "videos" / "play",
                step_trigger=lambda step: step == 0,
                video_length=cfg.video_length,
                disable_logger=True,
            )
        wrapped = RslRlVecEnvWrapper(env, clip_actions=agent_cfg.clip_actions)
        if cfg.agent == "trained":
            runner_cls = load_runner_cls(task_id) or MjlabOnPolicyRunner
            runner = runner_cls(wrapped, asdict(agent_cfg), device=device)
            runner.load(
                str(checkpoint), load_cfg={"actor": True}, strict=True, map_location=device
            )
            policy = runner.get_inference_policy(device=device)
        else:
            shape = env.unwrapped.action_space.shape

            def policy(obs):
                del obs
                if cfg.agent == "zero":
                    return torch.zeros(shape, device=device)
                return 2 * torch.rand(shape, device=device) - 1

        viewer = cfg.viewer
        if viewer == "auto":
            viewer = "native" if os.environ.get("DISPLAY") or os.environ.get("WAYLAND_DISPLAY") else "viser"
        print(f"[PHP] task={task_id}, checkpoint={checkpoint}, device={device}, viewer={viewer}")
        if viewer == "native":
            NativeMujocoViewer(wrapped, policy).run()
        else:
            from src.tasks.php.viewer import PhpViserPlayViewer

            PhpViserPlayViewer(wrapped, policy).run()
    finally:
        env.close()


def main():
    os.environ["PHP_ONLY"] = "1"
    import mjlab.tasks  # noqa: F401
    import src.tasks  # noqa: F401

    tasks = [
        name for name in list_tasks()
        if name.startswith(("Unitree-G1-PHP-", "E1-PHP-"))
    ]
    chosen, remaining = tyro.cli(
        tyro.extras.literal_type_from_choices(tasks),
        add_help=False, return_unknown_args=True, config=mjlab.TYRO_FLAGS,
    )
    cfg = tyro.cli(
        PlayConfig, args=remaining, config=mjlab.TYRO_FLAGS,
        prog=f"{sys.argv[0]} {chosen}",
    )
    run_play(chosen, cfg)


if __name__ == "__main__":
    main()
