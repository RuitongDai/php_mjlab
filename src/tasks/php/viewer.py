"""Keep Viser box dimensions synchronized with the per-environment physics model."""

from mjlab.viewer import ViserPlayViewer
import numpy as np
import time
import torch


class PhpViserPlayViewer(ViserPlayViewer):
    def setup(self):
        super().setup()
        env = self.env.unwrapped
        command = env.command_manager.get_term("motion")
        if "student_depth" in env.obs_buf:
            # Student camera omits the torso/head visuals (physics group 4).
            # In the external viewer these are ordinary robot visuals: attach
            # their handles to the G2 visibility toggle without changing either
            # the MuJoCo model or the camera's enabled_geom_groups=(0, 2).
            robot_bodies = {int(i) for i in env.scene["robot"].indexing.body_ids}
            handles = self._scene.mesh_handles_by_group
            for key in tuple(handles):
                body, group, _ = key
                if body not in robot_bodies or group != 4:
                    continue
                sub = 1 + max((k[2] for k in handles if k[:2] == (body, 2)), default=-1)
                handle = handles.pop(key)
                handles[(body, 2, sub)] = handle
                handle.visible = self._scene.geom_groups_visible[2]
        body_ids = {
            int(i)
            for entity in command.terrain_entities
            for i in entity.indexing.body_ids
        }
        # MJLab's normal Viser meshes are built from compile-time geometry.
        # Replace only the resized boxes; Native already syncs model geom_size.
        for key in tuple(self._scene.mesh_handles_by_group):
            if key[0] in body_ids:
                self._scene.mesh_handles_by_group.pop(key).remove()
        # The shared world frame follows Viser's camera-tracking translation.
        # Parenting here also keeps boxes aligned when tracking is toggled
        # while paused, without racing the asynchronous scene update.
        world_frame = self._scene.fixed_bodies_frame.name
        self._php_boxes = [
            self._server.scene.add_box(
                f"{world_frame}/php/box_{i}", color=(90, 150, 200)
            )
            for i in range(command.motion.max_boxes)
        ]
        self._php_geom_ids = torch.tensor(
            [int(e.indexing.geom_ids[0]) for e in command.terrain_entities],
            device=env.device,
            dtype=torch.long,
        )
        self._php_body_ids = torch.tensor(
            [int(e.indexing.body_ids[0]) for e in command.terrain_entities],
            device=env.device,
            dtype=torch.long,
        )
        self._php_box_counts = command.motion.box_counts.cpu().tolist()
        self._php_layout_key = None
        self._php_next_depth_update = 0.0
        self._php_next_camera_update = 0.0
        self._php_depth_env = self._php_camera_env = None
        self._php_depth = None
        if "student_depth" in self.env.unwrapped.obs_buf:
            with self._server.gui.add_folder("PHP Student depth"):
                self._server.gui.add_markdown(
                    "Policy input after preprocessing and latency; display maps [-0.5, 0.5] to grayscale."
                )
                self._php_depth = self._server.gui.add_image(
                    np.zeros((174, 261, 3), dtype=np.uint8), label="58 × 87 depth"
                )

    def _update_camera_feeds(self, sim, has_pending_updates):
        # Raw camera preview is separate from policy sensing. Encoding and
        # transmitting both depth previews at the 60 Hz viewer tick is wasteful.
        now = time.monotonic()
        index = self._scene.env_idx
        if now >= self._php_next_camera_update or index != self._php_camera_env:
            super()._update_camera_feeds(sim, has_pending_updates)
            self._php_next_camera_update = now + 0.1
            self._php_camera_env = index
        else:
            self._camera_update_last_ms = 0.0

    def sync_env_to_viewer(self):
        super().sync_env_to_viewer()
        # Match the base viewer's 30 Hz scene updates instead of doing an extra
        # GPU synchronization for custom geometry on every 60 Hz tick.
        if self._counter % 2:
            return
        self._sync_php_boxes()
        self._sync_php_depth()

    def _sync_php_boxes(self):
        env = self.env.unwrapped
        command = env.command_manager.get_term("motion")
        index = self._scene.env_idx
        motion, variant = (
            torch.stack((command.motion_ids[index], command.variant_ids[index]))
            .cpu()
            .tolist()
        )
        key = (index, motion, variant)
        count = self._php_box_counts[motion]
        # PHP boxes are fixed for the episode. Transfer the selected layout in
        # one batch only when motion/variant/environment changes. Their parent
        # world frame already handles camera tracking, including while paused.
        changed = key != self._php_layout_key
        if changed:
            poses = (
                torch.cat(
                    (
                        2 * env.sim.model.geom_size[index, self._php_geom_ids],
                        env.sim.data.xpos[index, self._php_body_ids],
                        env.sim.data.xquat[index, self._php_body_ids],
                    ),
                    dim=-1,
                )
                .cpu()
                .numpy()
            )
        with self._server.atomic():
            for i, handle in enumerate(self._php_boxes):
                handle.visible = i < count and self._scene.geom_groups_visible[0]
                if changed and i < count:
                    handle.dimensions = tuple(poses[i, :3])
                    handle.position = poses[i, 3:6]
                    handle.wxyz = poses[i, 6:10]
        self._php_layout_key = key

    def _sync_php_depth(self):
        env = self.env.unwrapped
        index = self._scene.env_idx
        now = time.monotonic()
        if now < self._php_next_depth_update and index == self._php_depth_env:
            return
        if self._php_depth is not None:
            depth = env.obs_buf["student_depth"][index, 0].detach()
            grey = ((depth + 0.5).clamp(0, 1).cpu().numpy() * 255).astype(np.uint8)
            self._php_depth.image = np.repeat(
                np.repeat(np.repeat(grey[..., None], 3, axis=2), 3, axis=0), 3, axis=1
            )
        self._php_next_depth_update = now + 0.1
        self._php_depth_env = index

    def reset_environment(self):
        super().reset_environment()
        self._php_layout_key = None
        self._php_next_depth_update = self._php_next_camera_update = 0.0
