import math

import torch
import torch.nn.functional as F
from mjlab.managers.event_manager import requires_model_fields
from mjlab.utils.lab_api.math import quat_from_euler_xyz, quat_mul

FOVY = math.degrees(2 * math.atan(math.tan(math.radians(89.5) / 2) * 60 / 106))


def camera_quaternion(angles):
    local = quat_from_euler_xyz(*torch.deg2rad(angles).unbind(-1))
    optical = quat_from_euler_xyz(
        *torch.deg2rad(angles.new_tensor([-90, 0, -90])).unbind(-1)
    )
    # Original camera looks along +Z with +Y down; MuJoCo uses -Z, +Y up.
    return quat_mul(
        quat_mul(local, optical.expand_as(local)),
        angles.new_tensor([0, 1, 0, 0]).expand_as(local),
    )


@requires_model_fields("cam_pos", "cam_quat")
def randomize_camera(env, env_ids, sensor_name="depth_camera"):
    if env_ids is None:
        env_ids = torch.arange(env.num_envs, device=env.device)
    index = env.scene[sensor_name].camera_idx
    n = len(env_ids)
    env.sim.model.cam_pos[env_ids, index] = torch.tensor(
        [0.01, 0.01, 0.44], device=env.device
    ) + torch.empty(n, 3, device=env.device).uniform_(-0.025, 0.025)
    angles = torch.tensor([1.0, 27.0, 1.0], device=env.device) + (
        torch.rand(n, 3, device=env.device) * 2 - 1
    ) * torch.tensor([2.5, 3.0, 2.5], device=env.device)
    env.sim.model.cam_quat[env_ids, index] = camera_quaternion(angles)


class PhpDepthImage:
    def __init__(self, cfg, env):
        self.tick = -1
        self.pending = torch.ones(env.num_envs, dtype=torch.bool, device=env.device)
        self.history = torch.zeros(env.num_envs, 6, 58, 87, device=env.device)
        self.output = torch.zeros(env.num_envs, 1, 58, 87, device=env.device)
        self.bias = torch.randn(env.num_envs, 1, 1, device=env.device) * 0.03
        self._gaussian_kernel = None
        self._gaussian_key = None
        focal = 30 / math.tan(math.radians(FOVY) / 2)
        y = (torch.arange(60, device=env.device) + 0.5 - 30) / focal
        x = (torch.arange(106, device=env.device) + 0.5 - 53) / focal
        self.ray_to_z = torch.rsqrt(1 + y[:, None] ** 2 + x[None, :] ** 2)

    def reset(self, env_ids=None):
        self.pending[slice(None) if env_ids is None else env_ids] = True

    def _blur(self, depth, gaussian_blur):
        if gaussian_blur is None:
            return depth
        size, sigma = gaussian_blur
        if size <= 0 or size % 2 == 0 or not math.isfinite(sigma) or sigma <= 0:
            raise ValueError(
                "Gaussian blur requires an odd positive kernel and finite positive sigma"
            )
        padding = size // 2
        if padding >= min(depth.shape[-2:]):
            raise ValueError("Gaussian kernel is too large for reflection padding")
        key = (size, sigma, depth.device, depth.dtype)
        if key != self._gaussian_key:
            coords = (
                torch.arange(size, device=depth.device, dtype=depth.dtype) - padding
            )
            kernel = torch.exp(-0.5 * (coords / sigma).square())
            kernel /= kernel.sum()
            self._gaussian_kernel = (kernel[:, None] * kernel[None, :])[None, None]
            self._gaussian_key = key
        return F.conv2d(
            F.pad(depth[:, None], (padding,) * 4, mode="reflect"),
            self._gaussian_kernel,
        )[:, 0]

    def process(self, ranges, corrupt, gaussian_blur=(3, 1.0)):
        depth = torch.where(ranges > 0, ranges * self.ray_to_z, 3.0).clamp(0.3, 3.0)
        depth = self._blur(depth[:, 2:, 4:-4], gaussian_blur)
        depth = F.interpolate(
            depth[:, None],
            size=(58, 87),
            mode="bicubic",
            align_corners=False,
            antialias=True,
        )[:, 0].clamp(max=3.0)
        depth = torch.where(depth < 0.15, 3.0, depth)
        if corrupt:
            sx = depth.new_tensor([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]])[None, None]
            gx = F.conv2d(depth[:, None], sx, padding=1)
            gy = F.conv2d(depth[:, None], sx.transpose(-1, -2), padding=1)
            grad = (gx.square() + gy.square()).sqrt()[:, 0]
            grad[:, :3] = grad[:, -3:] = 0
            grad[:, :, :3] = grad[:, :, -3:] = 0
            r = torch.rand_like(depth)
            shuffle = F.max_pool2d(((grad > 1) & (r < 0.9)).float()[:, None], 3, 1, 1)[
                :, 0
            ].bool()
            patches = F.unfold(
                F.pad(depth[:, None], (1, 1, 1, 1), mode="circular"), 3
            ).view(-1, 9, 58, 87)
            neighbors = patches.gather(
                1, torch.randint(9, (len(depth), 1, 58, 87), device=depth.device)
            )[:, 0]
            depth = torch.where(shuffle, neighbors, depth)
            empty = (
                (grad > 0.6)
                & (F.max_pool2d(depth[:, None], 3, 1, 1)[:, 0] > 2.5)
                & (r < 0.7)
            )
            depth[empty] = 3.0
            depth = depth + torch.randn_like(depth) * 0.03 + self.bias
        return (depth - 0.3) / 2.7 - 0.5

    def __call__(
        self,
        env,
        sensor_name="depth_camera",
        corrupt=True,
        lag_min=3,
        lag_max=4,
        gaussian_blur: tuple[int, float] | None = (3, 1.0),
    ):
        if not 0 <= lag_min <= lag_max < 6:
            raise ValueError("Depth latency must satisfy 0 <= min <= max < 6")
        new_tick = self.tick != env.common_step_counter
        if new_tick or self.pending.any():
            current = self.process(
                env.scene[sensor_name].data.depth[..., 0], corrupt, gaussian_blur
            )
            if new_tick:
                self.history[:, :-1] = self.history[:, 1:].clone()
                self.history[:, -1] = current
            ids = self.pending.nonzero().flatten()
            self.history[ids] = current[ids, None].expand(-1, 6, -1, -1)
            lag = torch.randint(
                lag_min, lag_max + 1, (env.num_envs,), device=env.device
            )
            value = self.history[
                torch.arange(env.num_envs, device=env.device), 5 - lag, None
            ]
            if new_tick:
                self.output = value.clone()
            else:
                self.output[ids] = value[ids]
            self.pending.zero_()
            self.tick = env.common_step_counter
        return self.output
