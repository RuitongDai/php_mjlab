"""Run the PHP depth student with viewer, terminal, motion or fixed commands."""

import argparse
from contextlib import nullcontext
import json
import os
from pathlib import Path
import threading
import time
import sys

if "--headless" in sys.argv:
    os.environ.setdefault("MUJOCO_GL", "egl")

import mujoco
import numpy as np
import torch

from .policy_runtime import (
    COMMAND_NAMES,
    DepthHistory,
    PhpOnnxPolicy,
    build_proprioception,
    one_hot,
    preprocess_depth_z,
)
from .scene import DepthCamera, build_scene, check_contract, initialize
from .depth_overlay import DepthOverlay


class CommandController:
    """Latched commands; callbacks never mutate simulation state."""

    KEYS = {
        ord("W"): "forward",
        ord("Q"): "left_45",
        ord("A"): "left_90",
        ord("E"): "right_45",
        ord("D"): "right_90",
        ord("S"): "stand",
        ord(" "): "stand",
    }

    def __init__(self, initial):
        self.initial = initial
        self.command = initial
        self.running = initial.startswith("run_")
        self.reset_requested = False
        self.lock = threading.Lock()

    def key_callback(self, key):
        with self.lock:
            if key in (ord("R"), 259):  # R or native viewer Backspace reset
                self.reset_requested = True
            elif key == ord("T"):
                self.running = not self.running
                if self.command != "stand":
                    base = self.command.removeprefix("run_")
                    self.command = "run_" + base if self.running else base
            elif key in self.KEYS:
                base = self.KEYS[key]
                self.command = (
                    "run_" + base if self.running and base != "stand" else base
                )

    def poll(self):
        with self.lock:
            reset = self.reset_requested
            self.reset_requested = False
            if reset:
                self.command = self.initial
                self.running = self.initial.startswith("run_")
            return self.command, reset


def load_commands(motion_file):
    from src.tasks.php.dataset import compact_velocity_commands

    with np.load(motion_file, allow_pickle=False) as m:
        raw = m["vel_cmd"]
        if raw.shape != (len(m["joint_pos"]), 15) or not np.isfinite(raw).all():
            raise ValueError("Invalid motion vel_cmd")
        result = compact_velocity_commands(raw, motion_file)
    if not np.all((result == 0) | (result == 1)) or not np.all(result.sum(-1) == 1):
        raise ValueError("Motion commands must be one-hot")
    return result.astype(np.float32)


def run(args):
    if (
        args.command_source == "terminal"
        and not args.validate_only
        and not sys.stdin.isatty()
    ):
        raise ValueError("--command-source=terminal requires an interactive terminal")
    torch.set_num_threads(2)
    if args.headless and args.depth_backend == "renderer":
        os.environ.setdefault("MUJOCO_GL", "egl")
    policy = PhpOnnxPolicy(args.policy, args.provider)
    model, b, contract = build_scene(
        args.motion_file, args.terrain_variant, args.terrain_file, args.ccd_iterations
    )
    check_contract(policy.metadata, contract)
    data = initialize(model, b, args.motion_file, args.start_frame)
    commands = (
        load_commands(args.motion_file) if args.command_source == "motion" else None
    )
    if args.validate_only:
        print(f"[OK] PHP model/ONNX contract; {model.nu} actions, {model.ngeom} geoms.")
        return {}
    camera = DepthCamera(model, b, args.depth_backend)
    history = DepthHistory(args.depth_lag, args.seed)
    controller = CommandController(args.command)
    terminal = None
    if args.command_source == "terminal":
        from .terminal_keyboard import TerminalKeyboard

        terminal = TerminalKeyboard(controller)
    saved = (data.qpos.copy(), data.qvel.copy(), data.ctrl.copy())
    previous = np.zeros(29, dtype=np.float32)
    control_dt = contract["control_dt"]
    decimation = round(control_dt / model.opt.timestep)
    steps = 0
    frame_offset = 0
    max_action = 0.0
    resets = 0
    last_time = data.time
    started = time.monotonic()
    if args.headless:
        context = nullcontext(None)
    else:
        from mujoco import viewer as native_viewer

        context = native_viewer.launch_passive(
            model,
            data,
            key_callback=None if terminal is not None else controller.key_callback,
        )
    print("[PHP] W/Q/A/E/D: directions; S/Space: stand; T: walk/run; R: reset.")
    print(
        f"[PHP] command_source={args.command_source}; depth={args.depth_backend}, lag={args.depth_lag}"
    )
    if terminal is not None:
        print(
            "[PHP] Focus this terminal and press keys (no Enter needed); Ctrl+C exits.",
            flush=True,
        )
    displayed_command = None
    input_context = terminal if terminal is not None else nullcontext()
    try:
        with input_context, context as viewer:
            overlay = (
                DepthOverlay(viewer) if viewer is not None and args.show_depth else None
            )
            if viewer is not None:
                viewer.cam.distance = 3.0
                viewer.cam.elevation = -18.0
                # External view shows torso/head; DepthCamera keeps its own
                # group mask (0, 2), excluding these camera-enclosing meshes.
                viewer.opt.geomgroup[4] = 1
                if overlay is not None:
                    print(
                        "[PHP] Upper-right: policy depth 87x58, including input delay; "
                        "near=black, far=white."
                    )
            while args.duration == 0 or steps * control_dt < args.duration - 1e-9:
                if viewer is not None and not viewer.is_running():
                    break
                tick = time.monotonic()
                if terminal is not None:
                    terminal.poll()
                name, reset = controller.poll()
                if terminal is not None and (name != displayed_command or reset):
                    print(
                        f"[PHP] command={name}" + (" (reset)" if reset else ""),
                        flush=True,
                    )
                    displayed_command = name
                if reset or data.time < last_time:
                    mujoco.mj_resetData(model, data)
                    data.qpos[:], data.qvel[:], data.ctrl[:] = saved
                    mujoco.mj_forward(model, data)
                    previous.fill(0)
                    history.reset()
                    frame_offset = 0
                    resets += 1
                    last_time = data.time
                if commands is not None:
                    index = args.start_frame + frame_offset
                    if index >= len(commands):
                        print("[PHP] Motion command sequence finished.")
                        break
                    command = commands[index]
                else:
                    command = one_hot(
                        args.command if args.command_source == "fixed" else name
                    )
                # mj_step leaves some derived fields at the previous integration state.
                mujoco.mj_forward(model, data)
                raw = camera.capture(data)
                current = preprocess_depth_z(raw, contract["gaussian_blur"])
                depth = history.push(current)
                obs = build_proprioception(
                    data.xmat[b.torso],
                    data.sensordata[b.gyro : b.gyro + 3],
                    data.qpos[b.qpos],
                    data.qvel[b.dof],
                    previous,
                    policy.default_joint_pos,
                    command,
                )
                action = policy(obs, depth)
                clip = policy.metadata.get("clip_actions")
                if clip is not None:
                    action = action.clip(-clip, clip)
                data.ctrl[b.actuators] = policy.joint_target(action)
                previous = action.copy()
                max_action = max(max_action, float(np.max(np.abs(action))))
                for _ in range(decimation):
                    mujoco.mj_step(model, data)
                if not np.isfinite(data.qpos).all() or not np.isfinite(data.qvel).all():
                    raise FloatingPointError("Simulation state contains NaN/Inf")
                if data.time <= last_time and not reset and steps > 0:
                    raise FloatingPointError(
                        "MuJoCo reset time unexpectedly; check numerical warnings"
                    )
                last_time = data.time
                steps += 1
                frame_offset += 1
                if viewer is not None:
                    with viewer.lock():
                        viewer.cam.lookat[:] = data.xpos[b.torso]
                    if overlay is not None:
                        overlay.update(depth)
                    viewer.sync()
                if args.realtime:
                    time.sleep(max(0.0, control_dt - (time.monotonic() - tick)))
        report = {
            "control_steps": steps,
            "simulated_seconds": steps * control_dt,
            "wall_seconds": time.monotonic() - started,
            "resets": resets,
            "max_abs_action": max_action,
            "final_root_xyz": data.qpos[:3].tolist(),
            "command_source": args.command_source,
            "depth_backend": args.depth_backend,
        }
        if args.report:
            args.report.parent.mkdir(parents=True, exist_ok=True)
            args.report.write_text(json.dumps(report, indent=2) + "\n")
        print(json.dumps(report, ensure_ascii=False))
        return report
    finally:
        camera.close()


def parse_args(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--policy", type=Path, required=True)
    p.add_argument(
        "--motion-file",
        type=Path,
        help="Initial pose and paired terrain; also commands in motion mode",
    )
    p.add_argument(
        "--terrain-file",
        type=Path,
        help="Static obstacle XML, or PHP *_terrain.npy; overrides paired terrain",
    )
    p.add_argument("--terrain-variant", type=int, default=0)
    p.add_argument("--start-frame", type=int, default=0)
    p.add_argument(
        "--command-source",
        choices=("keyboard", "terminal", "motion", "fixed"),
        default="keyboard",
        help="keyboard: MuJoCo window keys; terminal: immediate keys in the launching terminal",
    )
    p.add_argument("--command", choices=COMMAND_NAMES, default="stand")
    p.add_argument("--depth-backend", choices=("ray", "renderer"), default="ray")
    p.add_argument("--depth-lag", choices=("random", "0", "3", "4"), default="random")
    p.add_argument("--provider", choices=("cpu", "cuda"), default="cpu")
    p.add_argument("--ccd-iterations", type=int, default=50)
    p.add_argument(
        "--duration",
        type=float,
        default=30.0,
        help="Seconds; 0 runs until closed (motion mode stops at clip end)",
    )
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--headless", action="store_true")
    p.add_argument(
        "--show-depth", action=argparse.BooleanOptionalAction, default=True,
        help="Show the exact delayed policy depth at upper-right (ignored headless)",
    )
    p.add_argument("--realtime", action=argparse.BooleanOptionalAction, default=True)
    p.add_argument("--validate-only", action="store_true")
    p.add_argument("--report", type=Path)
    args = p.parse_args(argv)
    if args.command_source == "motion" and args.motion_file is None:
        p.error("--command-source=motion requires --motion-file")
    if args.duration < 0 or not np.isfinite(args.duration) or args.ccd_iterations < 1:
        p.error("Invalid duration or CCD iterations")
    if args.terrain_variant < 0 or args.start_frame < 0:
        p.error("Variant/frame must be nonnegative")
    for name in ("policy", "motion_file", "terrain_file"):
        value = getattr(args, name)
        if value is not None:
            setattr(args, name, value.expanduser().resolve())
    return args


if __name__ == "__main__":
    try:
        run(parse_args())
    except KeyboardInterrupt:
        print("\n[PHP] Stopped.")
