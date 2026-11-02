# e1-motions-low-step-v1

56 E1 V3 motion/terrain pairs at 50 FPS.

Each *_motion.npz contains named E1 joints/bodies and a 15-column one-hot vel_cmd. joint_pos/joint_vel retain the 7/6 floating-base prefixes; names describe the 25 hinges. Each *_terrain.npy has shape (boxes, 10, 10), with full box sizes and wxyz quaternions. An empty box axis means flat ground. OBJ files are visualization companions.

Use an E1 training task and its E1 body-name selection (wrists end in wrist_pitch_link); the G1 task's wrist_yaw_link names do not apply. Angles are constrained to the model limits; adjustments are recorded in ../manifest.json.
