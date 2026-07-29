import isaaclab.sim as sim_utils
import gymnasium
import torch

from isaaclab.sim import SimulationCfg
from isaaclab.assets import ArticulationCfg, AssetBaseCfg
from isaaclab.actuators import DCMotorCfg
from isaaclab.sensors import ContactSensorCfg
from isaaclab.scene import InteractiveSceneCfg
from isaaclab.utils import configclass
from isaaclab.terrains import TerrainImporterCfg, TerrainGeneratorCfg, SubTerrainBaseCfg
from lib.utils.terrain_utils import *
from lib.env.WF_GOAT.base.WF_GOAT_base_env_cfg import GOATBaseEnvCfg
from isaaclab.terrains.height_field.hf_terrains_cfg import *


@configclass
class GOATStandTerrainEnvCfg(GOATBaseEnvCfg):
    ## ==================== Environment parameters ==================== ##
    episode_length_s = 10.0
    sim_dt = 0.005                              # 200Hz torque controller
    decimation = 2                              # 100Hz policy
    action_space = 8                            # [L + R, joint pos + wheel velocity]
    observation_space = 29                      # Observation space
    state_space = 41                            # State space including privilege information
    max_episode_length = episode_length_s/sim_dt 

    ## ==================== Controller gain ==================== ##
    joint_kp = torch.tensor([[3.3, 2.7, 14.0]])
    joint_kd = torch.tensor([[0.1, 0.1, 0.01]])
    wheel_kp = torch.tensor([[1.0]])
    wheel_ki = torch.tensor([[1.0]])
    PD_LPF_gain = 0.059
    PI_LPF_gain = 0.059
    joint_action_weight = 0.3
    wheel_action_weight = 1.0
    
    ## ==================== Robot configuration ==================== ##
    leg_dof = 3                                 # Hip, Thigh, Knee
    num_leg = 2                                 # Bipedal
    n_leg_j = leg_dof * num_leg
    num_total_joints = n_leg_j + num_leg        # Whee per legs
    torque_limits = torch.tensor([4.5, 4.5, 4.5, 4.5, 4.5, 4.5, 2.5, 2.5])
    joint_input_limits = torch.tensor([[0, 0], [0, 0], [0, 0], [0, 0], [0, 0], [0, 0]])         # Currently not used
    

    ## ==================== Curriculum parameters ==================== ##
    total_DR_curriculum_level = 5               # Domain Randomization curriculum level
    total_task_curriculum_level = ["balancing", "recovery"]
    success_rate_buffer_len = 500

    max_base_acceleration_noise_per = 10        # Noise percentage (%)
    max_base_angular_vel_noise_per = 20
    max_gravity_vector_noise_per = 5
    max_base_quaternion_noise_per = 5
    max_joint_pos_noise_per = 3
    max_joint_vel_noise_per = 150

    max_terrain_friction_random_per = 50        # Friction randomization (%)
    max_terrain_restitution_random_per = 50     # Restitution randomization (%)

    default_terrain_static_friction = 0.7       # Default terrain configuration
    default_terrain_dynamic_friction = 0.5
    default_terrain_restitution = 0.4

    ## ==================== Terminal condition ==================== ##
    height_reset_condition = 0.35               # meter (m)
    base_tilt_reset_condition = 28              # degree

    ## ==================== Reward Shaping ==================== ##
    target_height = 0.45                        # meter (m)
    upright_threshold = 5                       # degree
    height_threshold = 0.1                      # meter (m)
    curriculum_level_up_threshold = 0.8         # success rate
    curriculum_level_down_threshold = 0.2

    r_upright_weight = 0.5
    r_height_weight = 0.0
    r_vel_lin_weight = 0.01
    r_vel_ang_weight = 0.01
    r_vel_joint_weight = 0.0
    r_effort_weight = 0.0
    r_terminated_weight = 0.0
    r_alive_weight = 1.0

    # Simulation
    sim: SimulationCfg = SimulationCfg(
        dt=sim_dt,
        render_interval=decimation,
        physics_material=sim_utils.RigidBodyMaterialCfg(
            friction_combine_mode="multiply",
            restitution_combine_mode="multiply",
            static_friction=1.0,
            dynamic_friction=1.0,
            restitution=0.0,
        ),
    )

    # Terrain
    flat = FlatTerrain()
    pyramid = PyramidStairsTerrain(border_width=0.1,
                                   step_height_range=(0.1, 0.3),
                                   step_width=0.2,
                                   platform_width=0.3,
                                   holes=False)
    inverted_pyramid = InvertedPyramidStairsTerrain(border_width=0.1,
                                                    step_height_range=(0.1, 0.3),
                                                    step_width=0.2,
                                                    platform_width=0.3,
                                                    holes=False)
    random_grid = RandomGridTerrain(grid_width=0.3,
                                    grid_height_range=(0.0, 0.3),
                                    platform_width=1.0,
                                    holes=True)
    rails = RailsTerrain()

    pit = PitTerrain(double_pit=True)

    box = BoxTerrain(double_box=True)

    gap = GapTerrain()

    floating_ring = FloatingRingTerrain(ring_width_range=(0.1, 0.8),
                                        ring_height_range=(0.0, 0.8),
                                        ring_thickness=0.5,
                                        platform_width=0.3)
    star = StarTerrain()

    repeated_object = RepeatedObjectsTerrain(object_type="box",
                                             object_prop_start=(5, 0.5),
                                             object_prop_end=(10, 1.0))

    random_uniform = random_uniform_terrain_init(noise_range=(0.0, 0.2),
                                                 noise_step=0.1)

    terrain_generator_cfg = TerrainGeneratorCfg(
        curriculum = True,
        size=[10.0, 10.0],
        num_rows = 1,                               # curriculum level for each terrain
        num_cols = 6,                               # variation of terrain
        color_scheme="random",
        sub_terrains = {
            "flat_terrain": SubTerrainBaseCfg(function=flat.generate,
                                              proportion=0.0),                              # TODO: 각 proportion은 따로 변수화해서 관리

            "pyramid_stairs_terrain": SubTerrainBaseCfg(function=pyramid.generate,
                                                        proportion=0.0),

            "inverted_pyramid_stairs_terrain": SubTerrainBaseCfg(function=inverted_pyramid.generate,
                                                                 proportion=0.0),
            "random_grid_terrain": SubTerrainBaseCfg(function=random_grid.generate,
                                                     proportion=0.0),

            "rails_terrain": SubTerrainBaseCfg(function=rails.generate,
                                               proportion=0.0),
            "pit_terrain": SubTerrainBaseCfg(function=pit.generate,
                                             proportion=0.0),
            "box_terrain": SubTerrainBaseCfg(function=box.generate,
                                             proportion=0.0),
            "gap_terrain": SubTerrainBaseCfg(function=gap.generate,
                                             proportion=0.0),
            "floating_ring_terrain": SubTerrainBaseCfg(function=floating_ring.generate,
                                                       proportion=0.0),
            "star_terrain": SubTerrainBaseCfg(function=star.generate,
                                              proportion=0.0),
            # "repeated_objects_terrain": SubTerrainBaseCfg(function=repeated_object.generate,
            #                                               proportion=0.5),
            "random_uniform_terrain": HfRandomUniformTerrainCfg(proportion=1.0,                     # TODO: 위의 trimesh랑 여기서 부터 있는 height terrain은 호출방식이 살짝 다름. 우선 통일 안해놓음
                                                                noise_range=(0.0, 0.2),
                                                                noise_step=0.1),
            "pyramid_sloped_terrain": HfPyramidSlopedTerrainCfg(proportion=1.0,
                                                                slope_range=(0.3, 1.0),
                                                                platform_width=1.0,
                                                                inverted=False),
            "pyramid_stairs_terrain": HfPyramidStairsTerrainCfg(proportion=1.0,
                                                                step_height_range=(0.3, 0.6),
                                                                step_width=0.5,
                                                                platform_width=1.0,
                                                                inverted=False),
            "discrete_obstacles_terrain": HfDiscreteObstaclesTerrainCfg(proportion=1.0,
                                                                        obstacle_height_mode="choice",
                                                                        obstacle_width_range=(0.5, 0.8),
                                                                        obstacle_height_range=(0.5, 0.8),
                                                                        num_obstacles=10,
                                                                        platform_width=1.0),
            "wave_terrain": HfWaveTerrainCfg(proportion=1.0,
                                             amplitude_range=(0.5, 1.0),
                                             num_waves=3),
            "stepping_stones_terrain": HfSteppingStonesTerrainCfg(proportion=1.0,
                                                                  stone_height_max=0.5,
                                                                  stone_width_range=(0.3, 0.6),
                                                                  stone_distance_range=(0.5, 1.0),
                                                                  holes_depth=-0.5,
                                                                  platform_width=1.0)
        }
    
    )

    terrain_importer_cfg = TerrainImporterCfg(
        prim_path="/World/ground",
        terrain_type="generator",
        terrain_generator=terrain_generator_cfg,
        env_spacing=5.0,
        physics_material=sim_utils.RigidBodyMaterialCfg(
            static_friction=default_terrain_static_friction,
            dynamic_friction=default_terrain_dynamic_friction,
            restitution=default_terrain_restitution                 # Collision
        ),
        debug_vis=False
    )

    contact_sensor = ContactSensorCfg(
        prim_path="/World/envs/env_.*/Robot/.*wheel.*",
        history_length=0,
        update_period=0.0                                           # Update every period
    )