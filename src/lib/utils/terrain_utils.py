from isaaclab.utils import configclass
from isaaclab.terrains.trimesh.mesh_terrains import *
from isaaclab.terrains.trimesh.mesh_terrains_cfg import *


@configclass
class ObjectCfg(MeshRepeatedObjectsTerrainCfg.ObjectCfg):
    """Configuration for repeated boxes."""

    size: tuple[float, float] = MISSING
    """The width (along x) and length (along y) of the box (in m)."""
    max_yx_angle: float = 0.0
    """The maximum angle along the y and x axis. Defaults to 0.0."""
    degrees: bool = True
    """Whether the angle is in degrees. Defaults to True."""

class FlatTerrain:
    def __init__(self):
        """
        Flat terrain initialization
        """
        self.cfg = MeshPlaneTerrainCfg()

    def generate(self, difficulty, cfg):
        return flat_terrain(difficulty=difficulty, cfg=self.cfg)

class PyramidStairsTerrain:
    def __init__(self,
                 border_width: float = 0.0,
                 step_height_range: tuple[float, float] = (0.05, 0.2),
                 step_width: float = 0.3,
                 platform_width: float = 1.0,
                 holes: bool = False):
        """
        Pyramid stairs terrain initialization
        
        :param border_width(float): The width of the border around the terrain (in m). Defaults to 0.0.\n
                                    The border is a flat terrain with the same height as the terrain.
        :param step_height_range(tuple): The minimum and maximum height of the steps (in m).
        :param step_width(float): The width of the steps (in m).
        :param platform_width(float): The width of the square platform at the center of the terrain. Defaults to 1.0.
        :param holes(bool): If True, the terrain will have holes in the steps. Defaults to False.
        """
        self.cfg = MeshPyramidStairsTerrainCfg()
        self.cfg.border_width = border_width
        self.cfg.step_height_range = step_height_range
        self.cfg.step_width = step_width
        self.cfg.platform_width = platform_width
        self.cfg.holes = holes
    
    def generate(self, difficulty, cfg):
        return pyramid_stairs_terrain(difficulty=difficulty, cfg=self.cfg)

class InvertedPyramidStairsTerrain:
    def __init__(self,
                 border_width: float = 0.0,
                 step_height_range: tuple[float, float] = (0.05, 0.2),
                 step_width: float = 0.3,
                 platform_width: float = 1.0,
                 holes: bool = False):
        """
        Inverted Pyramid stairs terrain initialization (Goes down)
        
        :param border_width(float): The width of the border around the terrain (in m). Defaults to 0.0.\n
                                    The border is a flat terrain with the same height as the terrain.
        :param step_height_range(tuple): The minimum and maximum height of the steps (in m).
        :param step_width(float): The width of the steps (in m).
        :param platform_width(float): The width of the square platform at the center of the terrain. Defaults to 1.0.
        :param holes(bool): If True, the terrain will have holes in the steps. Defaults to False.
        """
        self.cfg = MeshInvertedPyramidStairsTerrainCfg()
        self.cfg.border_width = border_width
        self.cfg.step_height_range = step_height_range
        self.cfg.step_width = step_width
        self.cfg.platform_width = platform_width
        self.cfg.holes = holes
    
    def generate(self, difficulty, cfg):
        return inverted_pyramid_stairs_terrain(difficulty=difficulty, cfg=self.cfg)

class RandomGridTerrain:
    def __init__(self,
                 grid_width: float = 0.3,
                 grid_height_range: tuple[float, float] = (-0.1, 0.1),
                 platform_width: float = 1.0,
                 holes: bool = False):
        """
        Random Grid terrain initialization (Uneven ground)
        
        :param grid_width(float): The width of the grid cells (in m).
        :param grid_height_range(tuple): The minimum and maximum height of the grid cells (in m).
        :param platform_width(float): The width of the square platform at the center of the terrain. Defaults to 1.0.
        :param holes(bool): If True, the terrain will have holes in the steps. Defaults to False.
        """
        self.cfg = MeshRandomGridTerrainCfg()
        self.cfg.grid_width = grid_width
        self.cfg.grid_height_range = grid_height_range
        self.cfg.platform_width = platform_width
        self.cfg.holes = holes

    def generate(self, difficulty, cfg):
        return random_grid_terrain(difficulty=difficulty, cfg=self.cfg)
    
class RailsTerrain:
    def __init__(self,
                 rail_thickness_range: tuple[float, float] = (0.0, 1.0),
                 rail_height_range: tuple[float, float] = (0.05, 0.2),
                 platform_width: float = 1.0):
        """
        Rails terrain initialization (Beam-like structures)
        
        :param rail_thickness_range(tuple): The thickness of the inner and outer rails (in m)..
        :param rail_height_range(tuple): The minimum and maximum height of the rails (in m).
        :param platform_width(float): The width of the square platform at the center of the terrain. Defaults to 1.0..
        """
        self.cfg = MeshRailsTerrainCfg()
        self.cfg.rail_thickness_range = rail_thickness_range
        self.cfg.rail_height_range = rail_height_range
        self.cfg.platform_width = platform_width

    def generate(self, difficulty, cfg):
        return rails_terrain(difficulty=difficulty, cfg=self.cfg)
    
class PitTerrain:
    def __init__(self,
                 pit_depth_range: tuple = (0.5, 2.0),
                 platform_width: float = 1.0,
                 double_pit: bool = False):
        """
        Pit terrain initialization (Hole/Gap in the ground)
        
        :param pit_depth_range(tuple): The minimum and maximum height of the pit (in m).
        :param platform_width(float): The width of the square platform at the center of the terrain. Defaults to 1.0.
        :param double_pit(bool): If True, the pit contains two levels of stairs. Defaults to False.
        """
        self.cfg = MeshPitTerrainCfg()
        self.cfg.pit_depth_range = pit_depth_range
        self.cfg.platform_width = platform_width
        self.cfg.double_pit = double_pit

    def generate(self, difficulty, cfg):
        return pit_terrain(difficulty=difficulty, cfg=self.cfg)
    
class BoxTerrain:
    def __init__(self,
                 box_height_range: tuple[float, float] = (0.0, 1.0),
                 platform_width: float = 1.0,
                 double_box: bool = False):
        """
        Box terrain initialization (Discrete obstacles / Stepping stones)
        
        :param box_height_range(tuple): The minimum and maximum height of the box (in m).
        :param platform_width(float): The width of the square platform at the center of the terrain. Defaults to 1.0.
        :param double_box(bool): If True, the pit contains two levels of stairs/boxes. Defaults to False.
        """
        # Usually mapped to DiscreteObstacles in Isaac Lab
        self.cfg = MeshBoxTerrainCfg()
        self.cfg.box_height_range = box_height_range
        self.cfg.platform_width = platform_width
        self.cfg.double_box = double_box

    def generate(self, difficulty, cfg):
        return box_terrain(difficulty=difficulty, cfg=self.cfg)
    
class GapTerrain:
    def __init__(self,
                 gap_width_range: tuple[float, float] = (0.1, 0.5),
                 platform_width: float = 1.0):
        """
        Gap terrain initialization (Discontinuous ground)
        
        :param gap_width_range(tuple): The minimum and maximum width of the gap (in m)..
        :param platform_width(float): The width of the square platform at the center of the terrain. Defaults to 1.0.
        """
        self.cfg = MeshGapTerrainCfg()
        self.cfg.gap_width_range = gap_width_range
        self.cfg.platform_width = platform_width

    def generate(self, difficulty, cfg):
        return gap_terrain(difficulty=difficulty, cfg=self.cfg)
    
class FloatingRingTerrain:
    def __init__(self,
                 ring_width_range: tuple[float, float] = (0.0, 1.0),
                 ring_height_range: tuple[float, float] = (0.0, 1.0),
                 ring_thickness: float = 0.5,
                 platform_width: float = 0.5):
        """
        Floating Ring terrain initialization
        
        :param ring_width_range(tuple): The minimum and maximum width of the ring (in m).
        :param ring_height_range(tuple): The minimum and maximum height of the ring (in m).
        :param ring_thickness(float): The thickness (along z) of the ring (in m).
        :param platform_width(float): The width of the square platform at the center of the terrain. Defaults to 1.0.
        """
        self.cfg = MeshFloatingRingTerrainCfg()
        self.cfg.ring_width_range = ring_width_range
        self.cfg.ring_height_range = ring_height_range
        self.cfg.ring_thickness = ring_thickness
        self.cfg.platform_width = platform_width

    def generate(self, difficulty, cfg):
        return floating_ring_terrain(difficulty=difficulty, cfg=self.cfg)
    
class StarTerrain:
    def __init__(self,
                 num_bars: int = 5,
                 bar_width_range: tuple[float, float] = (0.1, 0.3),
                 bar_height_range: tuple[float, float] = (0.1, 0.3),
                 platform_width: float = 1.0):
        """
        Star pattern terrain initialization
        
        :param num_bars(int): The number of bars per-side the star. Must be greater than 2.
        :param bar_width_range(tuple): The minimum and maximum width of the bars in the star (in m).
        :param bar_height_range(tuple): The minimum and maximum height of the bars in the star (in m).
        :param platform_width(float): The width of the cylindrical platform at the center of the terrain. Defaults to 1.0.
        """
        self.cfg = MeshStarTerrainCfg()
        self.cfg.num_bars = num_bars
        self.cfg.bar_width_range = bar_width_range
        self.cfg.bar_height_range = bar_height_range
        self.cfg.platform_width = platform_width

    def generate(self, difficulty, cfg):
        return star_terrain(difficulty=difficulty, cfg=self.cfg)
    
class RepeatedObjectsTerrain:
    def __init__(self,
                 object_type: Literal["cylinder", "box", "cone"],
                 object_prop_start: tuple[float, float] = None,
                 object_prop_end: tuple[float, float] =None,
                 size: float= 0.3,
                 max_yx_angle: float = 90.0,
                 degrees: bool = True,
                 abs_height_noise: tuple[float, float] = (0.0, 0.0),
                 rel_height_noise: tuple[float, float] = (1.0, 1.0),
                 platform_width: float = 1.0,
                 platform_height: float = -1.0):
        """
        Repeated Objects terrain initialization (Loading external meshes)
        
        :param object_type(str): The type of object to generate.
        :param object_prop_start(tuple): (num_objects, height) at the start of the curriculum.
        :param object_prop_end(tuple): (num_objects, height) at the end of the curriculum.
        :param size(float): The width (along x) and length (along y) for the box (in m).\n
                             The radius for the pyramids (in m).
        :param max_yx_angle(float): The maximum angle along the y and x axis. Defaults to 90.0.
        :param degrees(bool): Whether the angle is in degrees. Defaults to True.
        :param abs_height_noise(float): The minimum and maximum amount of additive noise for the height of the objects. Default is set to 0.0,
        which is no noise.
        :param rel_height_noise(float): The minimum and maximum amount of multiplicative noise for the height of the objects. Default is set to 1.0,
        which is no noise.
        :param platform_width(float): The width of the cylindrical platform at the center of the terrain. Defaults to 1.0.
        :param platform_height(float): The height of the platform. Defaults to -1.0.\n
                                        If the value is negative, the height is the same as the object height.
        """
        self.cfg = MeshRepeatedObjectsTerrainCfg()
        self.cfg.object_type = object_type
        self.object_params_start = ObjectCfg(object_prop_start[0], object_prop_start[1])
        self.object_params_end = ObjectCfg(object_prop_end[0], object_prop_end[1])
        self.object_params_start.size = size
        self.object_params_start.max_yx_angle = max_yx_angle
        self.object_params_start.degrees = degrees
        self.object_params_end.size = size
        self.object_params_end.max_yx_angle = max_yx_angle
        self.object_params_end.degrees = degrees
        self.cfg.object_params_start = self.object_params_start
        self.cfg.object_params_end = self.object_params_end
        self.cfg.max_height_noise = None
        self.cfg.abs_height_noise = abs_height_noise
        self.cfg.rel_height_noise = rel_height_noise
        self.cfg.platform_width = platform_width
        self.cfg.platform_height = platform_height

    def generate(self, difficulty, cfg):
        return repeated_objects_terrain(difficulty=difficulty, cfg=self.cfg)