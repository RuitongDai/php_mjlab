"""E1 V3 25-DoF robot configuration, organized like mjlab's G1 constants.

The original 21 joints use the E1 AMP configuration's PD and home-pose
parameters; effort and velocity limits follow E1_V2/E1.urdf. The four restored
wrist joints extend that configuration to match ``xmls/e1.xml``; their PD gains
are initial tuning values, not measured specs.
"""

from copy import deepcopy
from pathlib import Path

import mujoco

from mjlab.actuator import BuiltinPositionActuatorCfg
from mjlab.entity import EntityArticulationInfoCfg, EntityCfg
from mjlab.utils.os import update_assets
from mjlab.utils.spec_config import CollisionCfg

##
# MJCF and assets.
##

E1_XML: Path = Path(__file__).resolve().parent / "xmls" / "e1.xml"
assert E1_XML.exists()


def get_assets(meshdir: str) -> dict[str, bytes]:
  assets: dict[str, bytes] = {}
  update_assets(assets, E1_XML.parent.parent / "meshes", meshdir)
  return assets


def get_spec() -> mujoco.MjSpec:
  """Load the robot asset with sensor bindings defined in the XML."""
  spec = mujoco.MjSpec.from_file(str(E1_XML))
  # Use a virtual asset directory so Scene.write exports portable mesh paths.
  spec.meshdir = "assets"
  spec.assets = get_assets(spec.meshdir)
  return spec


##
# Joint ordering.
##

# XML order. The 21-joint AMP order is preserved when the wrists are omitted.
E1_JOINT_NAMES = (
  "left_hip_pitch_joint",
  "left_hip_roll_joint",
  "left_hip_yaw_joint",
  "left_knee_joint",
  "left_ankle_pitch_joint",
  "left_ankle_roll_joint",
  "right_hip_pitch_joint",
  "right_hip_roll_joint",
  "right_hip_yaw_joint",
  "right_knee_joint",
  "right_ankle_pitch_joint",
  "right_ankle_roll_joint",
  "waist_yaw_joint",
  "left_shoulder_pitch_joint",
  "left_shoulder_roll_joint",
  "left_shoulder_yaw_joint",
  "left_elbow_joint",
  "left_wrist_roll_joint",
  "left_wrist_pitch_joint",
  "right_shoulder_pitch_joint",
  "right_shoulder_roll_joint",
  "right_shoulder_yaw_joint",
  "right_elbow_joint",
  "right_wrist_roll_joint",
  "right_wrist_pitch_joint",
)

##
# Actuator config.
##

# PD gains and armature from E1_V3/e1_21dof.py in E1_AMP_Vision_mjlab.
# Effort/velocity limits follow the supplied X3/E1_V2/E1.urdf. In particular,
# hip roll is limited to 60 Nm rather than the AMP configuration's 120 Nm.
E1_ARMATURE = 0.01

# Nominal motor speed limits (rad/s), retained as metadata. As in G1's
# BuiltinPositionActuatorCfg, these are not hard velocity clamps.
E1_VELOCITY_LIMITS: dict[str, float] = {
  ".*_hip_pitch_joint": 12.04,
  ".*_hip_roll_joint": 13.09,
  ".*_hip_yaw_joint": 13.61,
  ".*_knee_joint": 12.04,
  ".*_ankle_pitch_joint": 15.71,
  ".*_ankle_roll_joint": 15.71,
  "waist_yaw_joint": 13.09,
  ".*_shoulder_pitch_joint": 13.09,
  ".*_shoulder_roll_joint": 13.61,
  ".*_shoulder_yaw_joint": 15.71,
  ".*_elbow_joint": 13.09,
  ".*_wrist_roll_joint": 15.71,
  ".*_wrist_pitch_joint": 12.57,
}

E1_ACTUATOR_HIP_PITCH = BuiltinPositionActuatorCfg(
  target_names_expr=(".*_hip_pitch_joint",),
  stiffness=200.0,
  damping=5.0,
  effort_limit=120.0,
  armature=E1_ARMATURE,
)
E1_ACTUATOR_HIP_ROLL = BuiltinPositionActuatorCfg(
  target_names_expr=(".*_hip_roll_joint",),
  stiffness=200.0,
  damping=5.0,
  effort_limit=60.0,
  armature=E1_ARMATURE,
)
E1_ACTUATOR_HIP_YAW = BuiltinPositionActuatorCfg(
  target_names_expr=(".*_hip_yaw_joint",),
  stiffness=80.0,
  damping=3.0,
  effort_limit=36.0,
  armature=E1_ARMATURE,
)
E1_ACTUATOR_KNEE = BuiltinPositionActuatorCfg(
  target_names_expr=(".*_knee_joint",),
  stiffness=200.0,
  damping=5.0,
  effort_limit=120.0,
  armature=E1_ARMATURE,
)
E1_ACTUATOR_ANKLE_PITCH = BuiltinPositionActuatorCfg(
  target_names_expr=(".*_ankle_pitch_joint",),
  stiffness=80.0,
  damping=3.0,
  effort_limit=36.0,
  armature=E1_ARMATURE,
)
E1_ACTUATOR_ANKLE_ROLL = BuiltinPositionActuatorCfg(
  target_names_expr=(".*_ankle_roll_joint",),
  stiffness=60.0,
  damping=2.0,
  effort_limit=30.0,
  armature=E1_ARMATURE,
)
E1_ACTUATOR_WAIST = BuiltinPositionActuatorCfg(
  target_names_expr=("waist_yaw_joint",),
  stiffness=150.0,
  damping=4.0,
  effort_limit=60.0,
  armature=E1_ARMATURE,
)
E1_ACTUATOR_SHOULDER_PITCH_AND_ELBOW = BuiltinPositionActuatorCfg(
  target_names_expr=(".*_shoulder_pitch_joint", ".*_elbow_joint"),
  stiffness=30.0,
  damping=2.0,
  effort_limit=60.0,
  armature=E1_ARMATURE,
)
E1_ACTUATOR_SHOULDER_ROLL = BuiltinPositionActuatorCfg(
  target_names_expr=(".*_shoulder_roll_joint",),
  stiffness=30.0,
  damping=2.0,
  effort_limit=36.0,
  armature=E1_ARMATURE,
)
E1_ACTUATOR_SHOULDER_YAW = BuiltinPositionActuatorCfg(
  target_names_expr=(".*_shoulder_yaw_joint",),
  stiffness=30.0,
  damping=2.0,
  effort_limit=15.0,
  armature=E1_ARMATURE,
)

# Initial wrist gains: the 21-DoF AMP reference does not configure wrists.
E1_ACTUATOR_WRIST_ROLL = BuiltinPositionActuatorCfg(
  target_names_expr=(".*_wrist_roll_joint",),
  stiffness=30.0,
  damping=2.0,
  effort_limit=15.0,
  armature=E1_ARMATURE,
)
E1_ACTUATOR_WRIST_PITCH = BuiltinPositionActuatorCfg(
  target_names_expr=(".*_wrist_pitch_joint",),
  stiffness=10.0,
  damping=1.0,
  effort_limit=6.0,
  armature=E1_ARMATURE,
)

##
# Keyframe config.
##

HOME_KEYFRAME = EntityCfg.InitialStateCfg(
  pos=(0.0, 0.0, 0.75),
  joint_pos={
    ".*_hip_pitch_joint": -0.10,
    ".*_knee_joint": 0.23,
    ".*_ankle_pitch_joint": -0.13,
    "left_shoulder_roll_joint": 0.25,
    "right_shoulder_roll_joint": -0.25,
    ".*_elbow_joint": 1.0,
    ".*_wrist_(roll|pitch)_joint": 0.0,
  },
  joint_vel={".*": 0.0},
)

##
# Collision config.
##

# Same presets as G1, with the E1 reference's foot friction of 0.8.
FULL_COLLISION = CollisionCfg(
  geom_names_expr=(".*_collision",),
  condim={r"^(left|right)_foot[1-7]_collision$": 3, ".*_collision": 1},
  priority={r"^(left|right)_foot[1-7]_collision$": 1},
  friction={r"^(left|right)_foot[1-7]_collision$": (0.8,)},
)

FULL_COLLISION_WITHOUT_SELF = CollisionCfg(
  geom_names_expr=(".*_collision",),
  contype=0,
  conaffinity=1,
  condim={r"^(left|right)_foot[1-7]_collision$": 3, ".*_collision": 1},
  priority={r"^(left|right)_foot[1-7]_collision$": 1},
  friction={r"^(left|right)_foot[1-7]_collision$": (0.8,)},
)

FEET_ONLY_COLLISION = CollisionCfg(
  geom_names_expr=(r"^(left|right)_foot[1-7]_collision$",),
  contype=0,
  conaffinity=1,
  condim=3,
  priority=1,
  friction=(0.8,),
)

##
# Final config.
##

E1_ARTICULATION = EntityArticulationInfoCfg(
  actuators=(
    E1_ACTUATOR_HIP_PITCH,
    E1_ACTUATOR_HIP_ROLL,
    E1_ACTUATOR_HIP_YAW,
    E1_ACTUATOR_KNEE,
    E1_ACTUATOR_ANKLE_PITCH,
    E1_ACTUATOR_ANKLE_ROLL,
    E1_ACTUATOR_WAIST,
    E1_ACTUATOR_SHOULDER_PITCH_AND_ELBOW,
    E1_ACTUATOR_SHOULDER_ROLL,
    E1_ACTUATOR_SHOULDER_YAW,
    E1_ACTUATOR_WRIST_ROLL,
    E1_ACTUATOR_WRIST_PITCH,
  ),
  soft_joint_pos_limit_factor=0.9,
)


def get_e1_robot_cfg() -> EntityCfg:
  """Return an independent E1 config with all 25 joints controlled."""
  return EntityCfg(
    init_state=deepcopy(HOME_KEYFRAME),
    collisions=(deepcopy(FULL_COLLISION),),
    spec_fn=get_spec,
    articulation=deepcopy(E1_ARTICULATION),
    sort_actuators=True,
  )


E1_ACTION_SCALE: dict[str, float] = {}
for actuator in E1_ARTICULATION.actuators:
  assert isinstance(actuator, BuiltinPositionActuatorCfg)
  assert actuator.effort_limit is not None
  for name in actuator.target_names_expr:
    E1_ACTION_SCALE[name] = 0.25 * actuator.effort_limit / actuator.stiffness


if __name__ == "__main__":
  import mujoco.viewer as viewer

  from mjlab.entity import Entity

  robot = Entity(get_e1_robot_cfg())
  viewer.launch(robot.spec.compile())
