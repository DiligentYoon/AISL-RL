import os
import isaaclab.sim as sim_utils
from isaaclab.assets import ArticulationCfg
from isaaclab.actuators import DelayedPDActuatorCfg
from lib.assets.actuators.actuator_cfg import GearDelayedPDActuatorCfg


# Robot asset paths
current_dir = os.path.dirname(__file__)
WF_GOAT_ASSET = {
    "urdf_path": os.path.join(current_dir, "urdf/WF_GOAT.urdf"),
    "usd_path": os.path.join(current_dir, "usd/WF_GOAT.usd"),
    "usd_place": os.path.join(current_dir, "usd/"),
    "usd_filename": "WF_GOAT.usd"
}

# URDF to USD conversion
urdf_cfg: sim_utils.UrdfConverterCfg = sim_utils.UrdfConverterCfg(
    root_link_name = "base_Link",
    asset_path = WF_GOAT_ASSET["urdf_path"],
    usd_dir = WF_GOAT_ASSET["usd_place"],
    usd_file_name = WF_GOAT_ASSET["usd_filename"],
    fix_base=False,
    joint_drive=sim_utils.UrdfConverterCfg.JointDriveCfg(
        drive_type="force",
        gains=sim_utils.UrdfConverterCfg.JointDriveCfg.PDGainsCfg(stiffness=0.0, damping=0.0)
    ),
)
urdf_converter = sim_utils.UrdfConverter(cfg = urdf_cfg)

# URDF conversion check
if urdf_converter.usd_path == WF_GOAT_ASSET["usd_path"]:
    print("urdf conversion success!")
else:
    print("urdf conversion failed!")

# GOAT Cfg
GOAT_CFG: ArticulationCfg = ArticulationCfg(           
    prim_path="{ENV_REGEX_NS}/Robot",         
    soft_joint_pos_limit_factor=0.9,
    spawn=sim_utils.UsdFileCfg(
        usd_path=urdf_converter.usd_path,
        scale=(1.0, 1.0, 1.0),
        activate_contact_sensors=True,
        rigid_props=sim_utils.RigidBodyPropertiesCfg(
            disable_gravity=False,
            retain_accelerations=False,
            linear_damping=0.0,
            angular_damping=0.0,
            max_linear_velocity=1000.0,
            max_angular_velocity=1000.0,
            max_depenetration_velocity=1.0,
        ),
        articulation_props=sim_utils.ArticulationRootPropertiesCfg(
            enabled_self_collisions=False,
            solver_position_iteration_count=4,
            solver_velocity_iteration_count=0,
            fix_root_link=False
        ),
    ),

    # Link, Joint list in Isaac sim
    # Link = ['base_Link', 'hip_L_Link', 'hip_R_Link', 'thigh_L_Link', 'thigh_R_Link', 'calf_L_Link', 'calf_R_Link', 'wheel_L_Link', 'wheel_R_Link']
    # Joint = ['hip_L_Joint', 'hip_R_Joint', 'thigh_L_Joint', 'thigh_R_Joint', 'knee_L_Joint', 'knee_R_Joint', 'wheel_L_Joint', 'wheel_R_Joint']
    
    # Initial Joint pos and vel
    init_state=ArticulationCfg.InitialStateCfg(
        pos=(0.0, 0.0, 0.594),
        joint_pos={
            "hip_L_Joint": 0.0,
            "hip_R_Joint": 0.0,
            "thigh_L_Joint": 0.738,
            "thigh_R_Joint": -0.738,
            "knee_L_Joint": 1.462,
            "knee_R_Joint": -1.462,
            "wheel_L_Joint": 0.0,
            "wheel_R_Joint": 0.0,
            },
        ),

    # Actuators cfg
    actuators={
        "hip": DelayedPDActuatorCfg(
            joint_names_expr=["hip_.*",],
            effort_limit=4.5,
            velocity_limit=15.0,
            min_delay=0,
            max_delay=4,
            stiffness={
                "hip_L_Joint": 5.0,
                "hip_R_Joint": 5.0,
            },                                      
            damping={
                "hip_L_Joint": 0.1,
                "hip_R_Joint": 0.1,
            },
            friction={
                "hip_L_Joint": 0.12,
                "hip_R_Joint": 0.12,
            },
            dynamic_friction={
                "hip_L_Joint": 5.646268e-02,
                "hip_R_Joint": 5.646268e-02,
            },
            viscous_friction={
                "hip_L_Joi0.00nt": 3.190248e-01,
                "hip_R_Joint": 3.190248e-01,
            },             
            armature={
                "hip_L_Joint": 0.01,
                "hip_R_Joint": 0.01,
            }
            
        #         friction=0.12,                                      # Static friction coefficient
        #         dynamic_friction=5.646268e-02,                      # Dynamic friction coefficient 
        #         viscous_friction=3.190248e-01,                      # Viscous friction coefficient          
        ),

        "thigh": DelayedPDActuatorCfg(
            joint_names_expr=["thigh_.*",],
            effort_limit=4.5,
            velocity_limit=15.0,
            min_delay=0,
            max_delay=4,
            stiffness={
                "thigh_L_Joint": 5.0,
                "thigh_R_Joint": 5.0,
            },                                      
            damping={
                "thigh_L_Joint": 0.1,
                "thigh_R_Joint": 0.1,
            },
            friction={
                "thigh_L_Joint": 0.12,
                "thigh_R_Joint": 0.12,
            },              
            dynamic_friction={
                "thigh_L_Joint": 5.646268e-02,
                "thigh_R_Joint": 5.646268e-02,
            },
            viscous_friction={
                "thigh_L_Joint": 3.190248e-01,
                "thigh_R_Joint": 3.190248e-01,
            },             
            armature={
                "thigh_L_Joint": 0.01,
                "thigh_R_Joint": 0.01,
            }   
            #         friction=0.12,
            #         dynamic_friction=5.646268e-02,
            #         viscous_friction=3.190248e-01,          
        ),

        "knee": GearDelayedPDActuatorCfg(
            joint_names_expr=["knee_.*",],
            effort_limit=9.0,
            velocity_limit=7.5,
            gamma=1.0,
            gear_ratio=2.0,
            min_delay=0,
            max_delay=4,
            stiffness={
                "knee_L_Joint": 5.0,
                "knee_R_Joint": 5.0,
            },                                      
            damping={
                "knee_L_Joint": 0.1,
                "knee_R_Joint": 0.1,
            },
            friction={
                "knee_L_Joint": 0.12,
                "knee_R_Joint": 0.12,
            },        
            dynamic_friction={
                "knee_L_Joint": 5.373143e-02,
                "knee_R_Joint": 5.373143e-02,
            },
            viscous_friction={
                "knee_L_Joint": 8.441387e-02,
                "knee_R_Joint": 8.441387e-02,
            },                   
            armature={
                "knee_L_Joint": 0.01,
                "knee_R_Joint": 0.01,
            }    
            #         friction=0.12,
            #         dynamic_friction=5.373143e-02,
            #         viscous_friction=8.441387e-02,         
        ),
        
        "wheel": DelayedPDActuatorCfg(
            joint_names_expr=["wheel_.*",],
            effort_limit=2.5,
            velocity_limit=15.0,
            min_delay=0,
            max_delay=4,
            stiffness={
                "wheel_L_Joint": 0.0,
                "wheel_R_Joint": 0.0,
            },                                      
            damping={
                "wheel_L_Joint": 0.02,
                "wheel_R_Joint": 0.02,
            },
            friction={
                "wheel_L_Joint": 0.07,
                "wheel_R_Joint": 0.07,
            },      
            dynamic_friction={
                "wheel_L_Joint": 3.218126e-02,
                "wheel_R_Joint": 3.218126e-02,
            },
            viscous_friction={
                "wheel_L_Joint": 1.715931e-02,
                "wheel_R_Joint": 1.715931e-02,
            },       
            armature={
                "wheel_L_Joint": 0.01,
                "wheel_R_Joint": 0.01,
            }             
        ),
        #         friction=0.07,
        #         dynamic_friction=3.218126e-02,
        #         viscous_friction=1.715931e-02,
    }
)