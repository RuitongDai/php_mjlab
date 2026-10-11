# Low-speed step example training bundle

This bundle contains 56 prepared Unitree G1 motion/terrain pairs at 50 FPS,
selected from the PHP five-teacher training datasets.

Each `*_motion.npz` is already converted to the Holosoma training format.
Its `*_terrain.npy` contains paired terrain boxes and variants; an empty box
array represents ground without obstacles. Keep each pair together.
No motion-matching generation, conversion, or W&B dataset access is required.

From the PHP repository root, after training environment setup:

```bash
REGISTRY=file:///absolute/path/to/this/extracted/bundle \
LOGGER=disabled NGPUS=1 \
bash wbt_training/training_runs/run_terrain_teacher.sh \
    --training.headless True --training.num-envs 4096
```

These are example inputs for teacher training, not pretrained teacher checkpoints.
Train the corresponding teacher before using its checkpoint for student training.

Project: https://github.com/amazon-far/php_parkour
Dataset variant: `low_speed_step_final_v0`. Motion and terrain files are unchanged from the source bundle.
