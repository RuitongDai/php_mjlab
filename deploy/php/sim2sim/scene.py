"""Native MuJoCo robot, physical terrain and camera bindings for PHP."""

from dataclasses import dataclass
from pathlib import Path

import mujoco
import numpy as np

from .policy_runtime import COMMAND_NAMES, INPUTS, PROPRIO_ORDER


@dataclass
class Bindings:
    joint_names: list[str]
    qpos: np.ndarray
    dof: np.ndarray
    actuators: np.ndarray
    torso: int
    gyro: int
    camera: int
    root_qpos: int
    root_dof: int


def name_id(model, kind, name):
    index = mujoco.mj_name2id(model, kind, name)
    if index < 0:
        raise ValueError(f"Missing {kind}: {name}")
    return index


def build_scene(motion_file=None, variant=0, terrain_file=None, ccd_iterations=50):
    from mjlab.entity import Entity
    from src.tasks.php.config.g1.student_env_cfg import php_student_env_cfg

    cfg = php_student_env_cfg(play=True)
    camera = next(s for s in cfg.scene.sensors if s.name == "depth_camera")
    spec = mujoco.MjSpec.from_string("""<mujoco>
      <visual>
        <headlight diffuse="0.6 0.6 0.6" ambient="0.3 0.3 0.3" specular="0 0 0"/>
        <rgba haze="0.15 0.25 0.35 1"/>
        <global offwidth="640" offheight="480"/>
      </visual>
      <asset>
        <!-- Match the Parkour sim2sim ground and sky. -->
        <texture type="skybox" builtin="gradient" rgb1="0.3 0.5 0.7"
                 rgb2="0 0 0" width="512" height="3072"/>
        <texture type="2d" name="groundplane_texture" builtin="checker" mark="edge"
                 rgb1="0.20 0.30 0.40" rgb2="0.10 0.20 0.30" markrgb="0.8 0.8 0.8"
                 width="300" height="300"/>
        <material name="groundplane" texture="groundplane_texture" texuniform="true"
                  texrepeat="5 5" reflectance="0.2"/>
      </asset>
      <worldbody>
        <light pos="0 0 5" dir="0 0 -1" directional="true"/>
        <geom name="floor" type="plane" size="0 0 .1" group="0"
              friction="1 .005 .0001" material="groundplane" contype="1" conaffinity="1"/>
      </worldbody></mujoco>""")
    spec.attach(
        Entity(cfg.scene.entities["robot"]).spec,
        prefix="",
        suffix="",
        frame=spec.worldbody.add_frame(),
    )
    if terrain_file is None and motion_file is not None:
        path = Path(motion_file)
        terrain_file = path.with_name(
            path.name.removesuffix("_motion.npz") + "_terrain.npy"
        )
    if terrain_file is not None and Path(terrain_file).suffix.lower() == ".xml":
        terrain = mujoco.MjSpec.from_file(str(terrain_file))
        if any(
            (list(terrain.joints), list(terrain.actuators),
             list(terrain.sensors), list(terrain.keys))
        ):
            raise ValueError("Terrain XML must contain static obstacles only")
        for geom in terrain.geoms:
            # Match the PHP box collision bit and the depth camera's visible group.
            geom.group = 0
            geom.contype = 2
            geom.conaffinity = 0
        spec.attach(
            terrain,
            prefix="terrain_",
            suffix="",
            frame=spec.worldbody.add_frame(),
        )
    elif terrain_file is not None:
        boxes = np.load(terrain_file, allow_pickle=False)
        if boxes.ndim != 3 or boxes.shape[2] != 10 or not 0 <= variant < boxes.shape[1]:
            raise ValueError(
                "Terrain must be (boxes,variants,10); check --terrain-variant"
            )
        if not np.isfinite(boxes).all() or np.any(boxes[..., 7:] <= 0):
            raise ValueError("Terrain has invalid box dimensions/values")
        for i, box in enumerate(boxes[:, variant]):
            if not np.isclose(np.linalg.norm(box[3:7]), 1.0, atol=1e-5):
                raise ValueError("Terrain quaternion is not normalized")
            spec.worldbody.add_geom(
                name=f"php_box_{i}",
                type=mujoco.mjtGeom.mjGEOM_BOX,
                pos=box[:3],
                quat=box[3:7],
                size=box[7:] * 0.5,
                group=0,
                contype=2,
                conaffinity=0,
                condim=3,
                friction=(0.9, 0.005, 0.0001),
                rgba=(0.35, 0.59, 0.78, 1),
            )
    spec.body("torso_link").add_camera(
        name="php_depth", pos=camera.pos, quat=camera.quat, fovy=camera.fovy
    )
    sim = cfg.sim.mujoco
    spec.option.timestep = sim.timestep
    spec.option.integrator = mujoco.mjtIntegrator.mjINT_IMPLICITFAST
    spec.option.iterations = sim.iterations
    spec.option.ls_iterations = sim.ls_iterations
    spec.option.ccd_iterations = ccd_iterations
    spec.option.impratio = sim.impratio
    spec.option.tolerance = sim.tolerance
    spec.option.ls_tolerance = sim.ls_tolerance
    model = spec.compile()
    if (model.nq, model.nv, model.nu) != (36, 35, 29):
        raise ValueError("Expected floating-base 29-DoF G1")
    joints = [
        i for i in range(model.njnt) if model.jnt_type[i] == mujoco.mjtJoint.mjJNT_HINGE
    ]
    names = [model.joint(i).name for i in joints]
    actuators = np.array(
        [name_id(model, mujoco.mjtObj.mjOBJ_ACTUATOR, n) for n in names]
    )
    if not np.array_equal(model.actuator_trnid[actuators, 0], joints):
        raise ValueError("Actuator-to-joint mapping differs")
    root = name_id(model, mujoco.mjtObj.mjOBJ_JOINT, "floating_base_joint")
    gyro = name_id(model, mujoco.mjtObj.mjOBJ_SENSOR, "imu_ang_vel")
    b = Bindings(
        names,
        model.jnt_qposadr[joints],
        model.jnt_dofadr[joints],
        actuators,
        name_id(model, mujoco.mjtObj.mjOBJ_BODY, "torso_link"),
        int(model.sensor_adr[gyro]),
        name_id(model, mujoco.mjtObj.mjOBJ_CAMERA, "php_depth"),
        int(model.jnt_qposadr[root]),
        int(model.jnt_dofadr[root]),
    )
    model.vis.map.znear = 0.001 / model.stat.extent
    model.vis.map.zfar = max(10.0, float(model.stat.extent) * 5) / model.stat.extent
    if model.nkey != 1:
        raise ValueError("Expected one robot home keyframe")
    blur = cfg.observations["student_depth"].terms["depth"].params["gaussian_blur"]
    contract = {
        "format": "php-sim2sim-v1",
        "php_version": 2,
        "inputs": INPUTS,
        "proprio_order": PROPRIO_ORDER,
        "velocity_command_names": list(COMMAND_NAMES),
        "joint_names": names,
        "default_joint_pos": model.key_qpos[0, b.qpos].tolist(),
        "action_scale": 1.0,
        "physics_dt": sim.timestep,
        "control_dt": sim.timestep * cfg.decimation,
        "camera_pos": list(camera.pos),
        "camera_quat_wxyz": list(camera.quat),
        "camera_fovy": camera.fovy,
        "gaussian_blur": list(blur) if blur else None,
        "gravity_frame": "torso_link",
        "gyro_sensor": "imu_ang_vel",
        "actuator_gainprm": model.actuator_gainprm[b.actuators].tolist(),
        "actuator_biasprm": model.actuator_biasprm[b.actuators].tolist(),
        "actuator_forcerange": model.actuator_forcerange[b.actuators].tolist(),
        "joint_armature": model.dof_armature[b.dof].tolist(),
        "joint_damping": model.dof_damping[b.dof].tolist(),
        "depth": {
            "raw_hw": [60, 106],
            "crop_tblr": [2, 0, 4, 4],
            "range_m": [0.3, 3.0],
            "resize": "bicubic_antialias_58x87",
            "normalize": "(depth - 0.3) / 2.7 - 0.5",
            "lag_frames": [3, 4],
        },
    }
    return model, b, contract


def check_contract(actual, expected):
    for key, value in expected.items():
        if actual.get(key) != value:
            raise ValueError(
                f"Export/runtime mismatch for {key}; re-export with matching PHP configuration"
            )


def initialize(model, b, motion_file=None, frame=0):
    data = mujoco.MjData(model)
    mujoco.mj_resetDataKeyframe(model, data, 0)
    if motion_file is not None:
        with np.load(motion_file, allow_pickle=False) as m:
            if not np.isclose(np.asarray(m["fps"]).item(), 50.0):
                raise ValueError("Motion must be 50 FPS")
            if not 0 <= frame < len(m["joint_pos"]):
                raise ValueError("Invalid --start-frame")
            names = m["joint_names"].tolist()
            order = np.array([names.index(n) for n in b.joint_names])
            root = m["body_names"].tolist().index("pelvis")
            q, v = b.root_qpos, b.root_dof
            data.qpos[q : q + 3] = m["body_pos_w"][frame, root]
            quat = m["body_quat_w"][frame, root].astype(float)
            quat /= np.linalg.norm(quat)
            data.qpos[q + 3 : q + 7] = quat
            data.qvel[v : v + 3] = m["body_lin_vel_w"][frame, root]
            rotation = np.zeros(9)
            mujoco.mju_quat2Mat(rotation, quat)
            data.qvel[v + 3 : v + 6] = (
                rotation.reshape(3, 3).T @ m["body_ang_vel_w"][frame, root]
            )
            for key, address, allowed in (
                ("joint_pos", b.qpos, (0, 7)),
                ("joint_vel", b.dof, (0, 6)),
            ):
                offset = m[key].shape[1] - len(names)
                if offset not in allowed:
                    raise ValueError(f"Invalid floating-base prefix: {key}")
                target = data.qpos if key == "joint_pos" else data.qvel
                target[address] = m[key][frame, order + offset]
    elif frame != 0:
        raise ValueError("--start-frame requires --motion-file")
    data.ctrl[b.actuators] = data.qpos[b.qpos]
    mujoco.mj_forward(model, data)
    if not np.isfinite(data.qpos).all() or not np.isfinite(data.qvel).all():
        raise ValueError("Invalid initial robot state")
    return data


class DepthCamera:
    """CPU rays avoid a GL context; renderer is available for raster comparison."""

    def __init__(self, model, b, backend="ray"):
        self.model, self.b = model, b
        self.renderer = None
        self.groups = np.array([1, 0, 1, 0, 0, 0], dtype=np.uint8)
        focal = 30 / np.tan(np.deg2rad(model.cam_fovy[b.camera]) / 2)
        x, y = np.meshgrid(
            (np.arange(106) + 0.5 - 53) / focal, (np.arange(60) + 0.5 - 30) / focal
        )
        directions = np.stack([x, -y, -np.ones_like(x)], axis=-1).reshape(-1, 3)
        self.directions = directions / np.linalg.norm(directions, axis=1, keepdims=True)
        self.distance = np.empty(60 * 106)
        self.geom_ids = np.empty(60 * 106, dtype=np.int32)
        if backend == "renderer":
            self.renderer = mujoco.Renderer(model, height=60, width=106)
            self.renderer.enable_depth_rendering()
            self.option = mujoco.MjvOption()
            self.option.geomgroup[:] = self.groups
        elif backend != "ray":
            raise ValueError("Unknown depth backend")

    def capture(self, data):
        if self.renderer is not None:
            self.renderer.update_scene(
                data, camera=self.b.camera, scene_option=self.option
            )
            return np.asarray(self.renderer.render(), dtype=np.float32).copy()
        rotation = data.cam_xmat[self.b.camera].reshape(3, 3)
        vectors = np.ascontiguousarray(self.directions @ rotation.T)
        mujoco.mj_multiRay(
            self.model,
            data,
            data.cam_xpos[self.b.camera],
            vectors.ravel(),
            self.groups,
            True,
            -1,
            self.geom_ids,
            self.distance,
            None,
            len(self.distance),
            # This cutoff culls geoms by their location, not the ray's hit
            # distance. A finite cutoff incorrectly drops our infinite floor
            # once the camera moves away from the plane's origin. Apply the
            # sensor's 3 m range in preprocess_depth_z, after intersection.
            mujoco.mjMAXVAL,
        )
        z = np.where(self.distance > 0, self.distance * -self.directions[:, 2], 3.0)
        return z.reshape(60, 106).astype(np.float32)

    def close(self):
        if self.renderer is not None:
            self.renderer.close()
