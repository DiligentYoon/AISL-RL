import os

import isaaclab.sim as sim_utils
from isaaclab.assets import RigidObjectCfg


# Jig asset paths
current_dir = os.path.dirname(__file__)

JIG_ASSET = {
    "urdf_path": os.path.join(current_dir, "urdf/GOAT_stand_jig.urdf"),
    "usd_path": os.path.join(current_dir, "usd/GOAT_stand_jig.usd"),
    "usd_place": os.path.join(current_dir, "usd/"),
    "usd_filename": "GOAT_stand_jig.usd",
}

# URDF -> USD conversion for jig
jig_urdf_cfg = sim_utils.UrdfConverterCfg(
    root_link_name="jig_link",
    asset_path=JIG_ASSET["urdf_path"],
    usd_dir=JIG_ASSET["usd_place"],
    usd_file_name=JIG_ASSET["usd_filename"],
    fix_base=False,   # movable base import
    joint_drive=sim_utils.UrdfConverterCfg.JointDriveCfg(
        drive_type="force",
        gains=sim_utils.UrdfConverterCfg.JointDriveCfg.PDGainsCfg(
            stiffness=0.0,
            damping=0.0,
        ),
    ),
)

jig_urdf_converter = sim_utils.UrdfConverter(cfg=jig_urdf_cfg)

if os.path.normpath(jig_urdf_converter.usd_path) == os.path.normpath(JIG_ASSET["usd_path"]):
    print("[JIG] URDF conversion success!")
else:
    print("[JIG] URDF conversion failed!")
    print("converted:", jig_urdf_converter.usd_path)
    print("expected :", JIG_ASSET["usd_path"])


## ===================== Jig Object ======================= ##
JIGCFG: RigidObjectCfg = RigidObjectCfg(
    prim_path="{ENV_REGEX_NS}/Jig",
    spawn=sim_utils.UsdFileCfg(
        usd_path=jig_urdf_converter.usd_path,
        scale=(1.0, 0.2, 1.2),
        rigid_props=sim_utils.RigidBodyPropertiesCfg(
            rigid_body_enabled=True,
            kinematic_enabled=True,
            disable_gravity=True,
        ),
        collision_props=sim_utils.CollisionPropertiesCfg(),
    ),
    init_state=RigidObjectCfg.InitialStateCfg(
        pos=(0.0, 0.0, 0.0),
        rot=(1.0, 0.0, 0.0, 0.0),
        lin_vel=(0.0, 0.0, 0.0),
        ang_vel=(0.0, 0.0, 0.0),
    )
)