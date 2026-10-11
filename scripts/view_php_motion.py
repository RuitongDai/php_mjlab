"""Preview PHP motion/terrain pairs without a policy or physics simulation.

Run: python -m scripts.view_php_motion
Validate all pairs without opening a server: python -m scripts.view_php_motion --check
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path
import time

import mujoco
import numpy as np


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_DATA = ROOT / "src/assets/motions/php_motion"
ROBOT_XML = ROOT / "src/assets/robots/unitree_g1/xmls/g1_29dof_spherehand.xml"


@dataclass
class Clip:
    qpos: np.ndarray
    terrain: np.ndarray
    fps: float


def load_clip(path: Path, model: mujoco.MjModel) -> Clip:
    """Map named joints to MJCF order; terrain rows are xyz, wxyz, full size."""
    with np.load(path, allow_pickle=False) as data:
        fps = float(np.asarray(data["fps"]).item())
        names = data["joint_names"].tolist()
        body_names = data["body_names"].tolist()
        joints = data["joint_pos"]
        offset = joints.shape[1] - len(names)
        if offset not in (0, 7) or len(joints) < 2:
            raise ValueError(f"Unsupported joint layout: {path}: {joints.shape}")
        root = body_names.index("pelvis")
        qpos = np.tile(model.qpos0, (len(joints), 1))
        qpos[:, :3] = data["body_pos_w"][:, root]
        qpos[:, 3:7] = data["body_quat_w"][:, root]
        for joint_id in range(model.njnt):
            if model.jnt_type[joint_id] == mujoco.mjtJoint.mjJNT_FREE:
                continue
            name = model.joint(joint_id).name
            qpos[:, model.jnt_qposadr[joint_id]] = joints[:, names.index(name) + offset]
    if not np.isfinite(fps) or fps <= 0 or not np.isfinite(qpos).all():
        raise ValueError(f"Invalid motion values: {path}")
    norms = np.linalg.norm(qpos[:, 3:7], axis=1, keepdims=True)
    if np.any(norms < 1e-8):
        raise ValueError(f"Zero root quaternion: {path}")
    qpos[:, 3:7] /= norms
    terrain_path = path.with_name(path.name.removesuffix("_motion.npz") + "_terrain.npy")
    terrain = np.load(terrain_path, allow_pickle=False)
    if terrain.ndim != 3 or terrain.shape[2] != 10:
        raise ValueError(f"Expected (boxes, variants, 10): {terrain_path}")
    if not np.isfinite(terrain).all() or np.any(terrain[..., 7:] <= 0):
        raise ValueError(f"Invalid terrain values: {terrain_path}")
    if terrain.shape[0] and (
        terrain.shape[1] == 0
        or not np.allclose(np.linalg.norm(terrain[..., 3:7], axis=-1), 1, atol=1e-5)
    ):
        raise ValueError(f"Invalid terrain variants/quaternions: {terrain_path}")
    return Clip(qpos, terrain, fps)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--motion-file", type=Path, default=DEFAULT_DATA,
                        help="One *_motion.npz file or a directory (searched recursively).")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8081)
    parser.add_argument("--check", action="store_true", help="Validate all pairs and exit.")
    args = parser.parse_args()
    source = args.motion_file.expanduser().resolve()
    files = sorted(source.rglob("*_motion.npz")) if source.is_dir() else [source]
    if not files:
        parser.error(f"No *_motion.npz files found in {source}")
    model = mujoco.MjModel.from_xml_path(str(ROBOT_XML))
    data = mujoco.MjData(model)
    if args.check:
        frames = 0
        for path in files:
            clip = load_clip(path, model)
            for frame in (0, len(clip.qpos) // 2, len(clip.qpos) - 1):
                data.qpos[:] = clip.qpos[frame]
                mujoco.mj_forward(model, data)
                if not np.isfinite(data.xpos).all():
                    raise ValueError(f"Invalid robot pose: {path}, frame {frame}")
            frames += len(clip.qpos)
        print(f"Validated {len(files)} motion/terrain pairs, {frames} frames; named joints and FK OK.")
        return

    import viser
    from mjlab.viewer.viser.conversions import merge_geoms

    base = source if source.is_dir() else source.parent
    labels = {str(path.relative_to(base)): path for path in files}
    initial = next((label for label in labels if label.endswith("run_forward_0.6s_step0s_motion.npz")), next(iter(labels)))
    clip = load_clip(labels[initial], model)
    server = viser.ViserServer(host=args.host, port=args.port)
    server.scene.set_up_direction("+z")
    server.scene.add_grid("/ground", width=40, height=40, plane="xy")
    robot_handles = []
    for body_id in range(1, model.nbody):
        geoms = [i for i in range(model.ngeom)
                 if model.geom_bodyid[i] == body_id and model.geom_group[i] == 2]
        if geoms:
            handle = server.scene.add_mesh_trimesh(
                f"/robot/{model.body(body_id).name}", merge_geoms(model, geoms))
            robot_handles.append((body_id, handle))

    server.gui.add_markdown("**PHP 动作预览**\n\n显示参考动作及配套地形。")
    selection = server.gui.add_dropdown("动作", options=tuple(labels), initial_value=initial)
    playing = server.gui.add_checkbox("播放", initial_value=True)
    speed = server.gui.add_slider("速度", min=0.1, max=2.0, step=0.1, initial_value=1.0)
    frame_slider = server.gui.add_slider("帧", min=0, max=len(clip.qpos) - 1, step=1, initial_value=0)
    variant = server.gui.add_slider("地形变体", min=0, max=max(1, clip.terrain.shape[1] - 1), step=1, initial_value=0)
    reset_camera = server.gui.add_button("重置视角")
    info = server.gui.add_markdown("")
    terrain_handles = []

    def aim_camera(client):
        center = (clip.qpos[:, :3].min(axis=0) + clip.qpos[:, :3].max(axis=0)) / 2
        span = max(4.0, float(np.ptp(clip.qpos[:, :2], axis=0).max()))
        client.camera.look_at = tuple(center)
        client.camera.position = tuple(center + np.array([span * 0.35, -span * 0.8, span * 0.5]))

    @server.on_client_connect
    def on_connect(client):
        aim_camera(client)

    @reset_camera.on_click
    def on_reset(_):
        for client in server.get_clients().values():
            aim_camera(client)

    active = initial
    terrain_variant = -1
    shown_frame = -1
    cursor = 0.0
    last_time = time.monotonic()
    print(f"Loaded {len(files)} clips. Open http://{args.host}:{server.get_port()}", flush=True)
    try:
        while True:
            now = time.monotonic()
            elapsed = min(now - last_time, 0.1)
            last_time = now
            if selection.value != active:
                active = selection.value
                clip = load_clip(labels[active], model)
                frame_slider.value = 0
                frame_slider.max = len(clip.qpos) - 1
                variant.value = 0
                variant.max = max(1, clip.terrain.shape[1] - 1)
                variant.disabled = clip.terrain.shape[0] == 0 or clip.terrain.shape[1] <= 1
                cursor, shown_frame, terrain_variant = 0.0, -1, -1
                for client in server.get_clients().values():
                    aim_camera(client)
            if variant.value != terrain_variant:
                terrain_variant = int(variant.value)
                with server.atomic():
                    for handle in terrain_handles:
                        handle.remove()
                    terrain_handles.clear()
                    if clip.terrain.shape[0]:
                        for i, box in enumerate(clip.terrain[:, terrain_variant]):
                            terrain_handles.append(server.scene.add_box(
                                f"/terrain/box_{i}", position=box[:3], wxyz=box[3:7],
                                dimensions=box[7:], color=(90, 150, 200)))
            if int(frame_slider.value) != shown_frame:
                cursor = float(frame_slider.value)
            elif playing.value:
                cursor = (cursor + elapsed * clip.fps * speed.value) % len(clip.qpos)
            frame = int(cursor)
            if frame != shown_frame:
                data.qpos[:] = clip.qpos[frame]
                mujoco.mj_forward(model, data)
                with server.atomic():
                    for body_id, handle in robot_handles:
                        handle.position = data.xpos[body_id].copy()
                        handle.wxyz = data.xquat[body_id].copy()
                    frame_slider.value = frame
                    info.content = (f"{frame / clip.fps:.2f} / {(len(clip.qpos) - 1) / clip.fps:.2f} 秒 · "
                                    f"{clip.fps:g} FPS · {clip.terrain.shape[0]} 个障碍")
                shown_frame = frame
            time.sleep(1 / 60)
    except KeyboardInterrupt:
        pass
    finally:
        server.stop()


if __name__ == "__main__":
    main()
