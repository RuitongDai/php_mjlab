# Parkour_mjlab

`parkour_mjlab` is a reinforcement learning project built upon
[MJLab](https://github.com/mujocolab/mjlab.git), using MuJoCo as its physics
simulation backend for **G1 Parkour** on the Unitree G1 robot.

MJLab combines [Isaac Lab](https://github.com/isaac-sim/IsaacLab)'s proven API
with best-in-class [MuJoCo](https://github.com/google-deepmind/mujoco_warp)
physics to provide lightweight, modular abstractions for reinforcement learning
robotics research and sim-to-real deployment.

## Installation
**Conda environment**

```bash
conda create -n mjlab python=3.11
conda activate mjlab
```

**Install dependencies**
```bash
sudo apt install -y libyaml-cpp-dev libboost-all-dev libeigen3-dev libspdlog-dev libfmt-dev
```

**Install parkour_mjlab**
```bash
git clone https://github.com/yhx1203/parkour_mjlab.git
```

```bash
cd parkour_mjlab
python -m pip install -e . 
```


## unitree_mujoco SDK2 sim2sim

**Preparation**

[unitree_sdk2_python](https://github.com/unitreerobotics/unitree_sdk2_python)

```bash
conda activate mjlab

cd ~
sudo apt install python3-pip
git clone https://github.com/unitreerobotics/unitree_sdk2_python.git
cd unitree_sdk2_python
pip3 install -e .
```


**Start the Unitree MuJoCo server in the first terminal**

```bash
cd parkour_mjlab
conda activate mjlab

python deploy/parkour/sim2sim/unitree_mujoco_parkour_server.py \
  --show-depth \
  --domain-id 6 \
  --interface lo \
  --duration 0
```

**Start the ONNX controller in a second terminal**

```bash
cd parkour_mjlab
conda activate mjlab

python deploy/parkour/sim2sim/g1_parkour_unitree_mujoco.py \
  --checkpoint-file logs/rsl_rl/g1_parkour/parkour_test/policy.onnx \
  --domain-id 6 \
  --interface lo \
  --cmd-x 0.6 \
  --duration 0
```

## Roadmap

- [ ] Release the training code
- [x] Release the sim2sim deployment code (MuJoCo)
- [ ] Release the sim2real deployment code



## Acknowledgements

This project builds upon and benefits from the following open-source repositories:

- [mjlab](https://github.com/mujocolab/mjlab)
- [unitree_rl_mjlab](https://github.com/unitreerobotics/unitree_rl_mjlab)
- [unitree_mujoco](https://github.com/unitreerobotics/unitree_mujoco)
