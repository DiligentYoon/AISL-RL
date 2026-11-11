from __future__ import annotations

import numpy as np
import trimesh

from isaaclab.terrains import MeshPlaneTerrainCfg
from isaaclab.terrains.trimesh.mesh_terrains import flat_terrain
from isaaclab.utils import configclass

# SubTerrain Cfg
@configclass
class WallBoxTerrainCfg(MeshPlaneTerrainCfg):
    """Configure a floor tile with one static wall."""
    # SubTerrainBase function
    # function = wall_box_terrain
    
    # Set mesh size
    wall_size: tuple[float, float, float] = (0.1, 3.0, 2.0)
    box_size_range: tuple[float, float] = (0.5, 0.8)
    
    # Keep space between the spawn origin, table, and wall.
    wall_distance: float = 2.5
    box_position: tuple[float, float, float] = (0.6, 0.6, 0.6)

class WallBoxTerrain:
    def __init__(self,
                 wall_distance: float = 2.5,
                 box_size_range: tuple[float, float] = (0.5, 0.8),
                 box_position: tuple[float, float, float] = (0.6, 0.6, 0.6)):
        self.cfg = WallBoxTerrainCfg()
        self.cfg.wall_distance = wall_distance
        self.cfg.box_size_range = box_size_range
        self.cfg.box_position = box_position

    # Generate function
    def wall_generate(self, difficulty: float, cfg: WallBoxTerrainCfg):
        """Return floor and wall meshes. cfg defines geometry and difficulty is unused."""
        meshes, origin = flat_terrain(difficulty, cfg)              # Difficulty is neglected

        # Wall
        wall = trimesh.creation.box(extents=cfg.wall_size)
        wall.apply_translation(origin + np.array([cfg.wall_distance, 0.0, cfg.wall_size[2] / 2.0]))
        
        meshes.append(wall)
        return meshes, origin

    def wall_box_generate(self, difficulty: float, cfg: WallBoxTerrainCfg):
        """Return floor, wall, box meshes. cfg defines geometry and difficulty is unused."""
        meshes, origin = flat_terrain(difficulty, cfg)              # Difficulty is neglected

        # Wall
        wall = trimesh.creation.box(extents=cfg.wall_size)
        wall.apply_translation(origin + np.array([cfg.wall_distance, 0.0, cfg.wall_size[2] / 2.0]))

        # Box
        box = trimesh.creation.box(extents=cfg.box_size)
        box.apply_translation(origin + np.array([cfg.box_position]))
        
        meshes.append(wall)
        meshes.append(box)
        return meshes, origin

    # Default generate function
    def generate(self, difficulty:float, cfg: WallBoxTerrainCfg):
        return self.wall_box_generate(self, difficulty=difficulty, cfg=self.cfg)