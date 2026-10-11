"""Display the exact delayed PHP policy input in the native MuJoCo viewer."""

import mujoco
import numpy as np


class DepthOverlay:
    def __init__(self, viewer):
        self.viewer = viewer

    def update(self, depth):
        depth = np.asarray(depth)
        if depth.shape != (1, 1, 58, 87) or not np.isfinite(depth).all():
            raise ValueError("Expected finite PHP student depth (1,1,58,87)")
        viewport = self.viewer.viewport
        if viewport is None or viewport.width <= 0 or viewport.height <= 0:
            return
        # PHP inputs are (metres - 0.3) / 2.7 - 0.5. Use a fixed grayscale
        # range: near = black, far = white. Never normalize per frame.
        gray = np.rint(np.clip(depth[0, 0] + 0.5, 0, 1) * 255).astype(np.uint8)
        margin = min(12, (viewport.width - 1) // 2, (viewport.height - 1) // 2)
        scale = min(
            4, (viewport.width - 2 * margin) / 87,
            (viewport.height - 2 * margin) / 58,
        )
        width, height = max(1, int(87 * scale)), max(1, int(58 * scale))
        # Nearest-neighbour enlargement preserves the actual input pixels.
        x = np.minimum(np.arange(width) * 87 // width, 86)
        y = np.minimum(np.arange(height) * 58 // height, 57)
        image = np.repeat(gray[y[:, None], x][..., None], 3, axis=-1)
        target = mujoco.MjrRect(
            viewport.left + viewport.width - width - margin,
            viewport.bottom + viewport.height - height - margin,
            width, height,
        )
        self.viewer.set_images((target, image))
