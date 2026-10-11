"""Compare complete Student inputs at identical states in MJLab and native MuJoCo.

Requires the five PHP motion bundles and a CUDA device. No trained policy needed.
"""

import json

import mujoco
import numpy as np
import torch

from .policy_runtime import COMMAND_NAMES, build_proprioception, preprocess_depth_z
from .scene import DepthCamera, build_scene, initialize


@torch.inference_mode()
def main():
    from mjlab.envs import ManagerBasedRlEnv
    from src.tasks.php.config.g1.student_env_cfg import php_student_env_cfg
    from src.tasks.php.dataset import BUNDLES, VELOCITY_COMMAND_NAMES
    from src.tasks.php.mdp.terrain import place_boxes

    torch.set_num_threads(2)
    assert tuple(VELOCITY_COMMAND_NAMES) == COMMAND_NAMES
    cfg = php_student_env_cfg(play=True)
    cfg.commands["motion"].debug_vis = False
    env = ManagerBasedRlEnv(cfg, device="cuda:0")
    results = []
    try:
        env.reset()
        command = env.command_manager.get_term("motion")
        robot = env.scene["robot"]
        ids = torch.tensor([0], device=env.device)
        np.testing.assert_allclose(env.scene.env_origins[0].cpu().numpy(), 0)
        for expert, skill in enumerate(BUNDLES):
            clip = int(torch.where(command.clip_experts == expert)[0][0])
            path = command.motion.paths[clip]
            model, b, contract = build_scene(path, variant=0)
            assert list(robot.joint_names) == b.joint_names
            np.testing.assert_allclose(
                robot.data.default_joint_pos[0].cpu().numpy(),
                contract["default_joint_pos"],
                atol=1e-7,
            )
            np.testing.assert_allclose(robot.data.default_joint_vel[0].cpu().numpy(), 0)
            for frame in (50, 150, 300):
                if frame >= int(command.motion.motion_lengths[clip]):
                    continue
                data = initialize(model, b, path, frame)
                command.motion_ids[0] = clip
                command.variant_ids[0] = 0
                command.time_steps[0] = command.motion.motion_start_idx[clip] + frame
                place_boxes(command, ids)
                for field in ("qpos", "qvel"):
                    target = getattr(env.sim.data, field)
                    target[0] = torch.as_tensor(
                        getattr(data, field), dtype=target.dtype, device=env.device
                    )
                # Include nonzero action history to catch raw-action/target confusion.
                previous = np.linspace(-0.15, 0.15, 29, dtype=np.float32)
                env.action_manager.process_action(
                    torch.from_numpy(previous[None]).to(env.device)
                )
                env.sim.forward()
                env.scene.update(env.step_dt)
                env.sim.sense()
                env.observation_manager.reset()
                obs = env.observation_manager.compute()
                vel = command.motion.vel_cmd[command.time_steps[0]].cpu().numpy()
                mujoco.mj_forward(model, data)
                proprio = build_proprioception(
                    data.xmat[b.torso],
                    data.sensordata[b.gyro : b.gyro + 3],
                    data.qpos[b.qpos],
                    data.qvel[b.dof],
                    previous,
                    np.asarray(contract["default_joint_pos"]),
                    vel,
                )
                expected_proprio = obs["student_proprio"][0].cpu().numpy()
                np.testing.assert_allclose(
                    proprio, expected_proprio, atol=3e-5, rtol=3e-5
                )
                camera = DepthCamera(model, b)
                try:
                    depth = preprocess_depth_z(
                        camera.capture(data), contract["gaussian_blur"]
                    )
                finally:
                    camera.close()
                expected_depth = obs["student_depth"].cpu().numpy()
                errors = np.abs(depth - expected_depth)
                # Allow rare silhouette-edge differences between CPU mesh and Warp rays.
                assert np.quantile(errors, 0.999) < 0.002, (skill, frame, errors.max())
                assert errors.max() < 0.03, (skill, frame, errors.max())
                result = {
                    "skill": skill,
                    "frame": frame,
                    "proprio_max_abs": float(np.abs(proprio - expected_proprio).max()),
                    "depth_max_abs": float(errors.max()),
                    "depth_p999_abs": float(np.quantile(errors, 0.999)),
                }
                results.append(result)
                print(json.dumps(result), flush=True)
        print(
            f"PASS: complete 104-D proprio and 58x87 depth alignment at {len(results)} states"
        )
    finally:
        env.close()


if __name__ == "__main__":
    main()
