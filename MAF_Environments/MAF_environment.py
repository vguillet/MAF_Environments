################################################################################################################
"""
Environment based on the popular Runscape MMORPG.
https://mapgenie.io/old-school-runescape/maps/runescape-surface
"""

# Built-in/Generic Imports
from copy import deepcopy
import random
import os
from abc import ABC, abstractmethod

# Libs
import numpy as np
from PIL import Image
import matplotlib.cm as cm
import matplotlib.pyplot as plt

from pathfinding.core.diagonal_movement import DiagonalMovement
from pathfinding.core.grid import Grid
from pathfinding.finder.a_star import AStarFinder

# Own modules
from MAF_Environments.Tools.Grid_tools import reduce_grid_scale
from MAF_Environments.Tools.tile_type import *
from MAF_Environments.Tools.action import Action
from MAF_Environments.Tools.Grid_tools import convert_coordinates

# from src.Visualiser.Visualiser import Visualiser
# from src.Visualiser.Visualiser_tools import Visualiser_tools


__version__ = '1.1.1'
__author__ = 'Victor Guillet'
__date__ = '31/01/2020'

################################################################################################################

SUPPORTED_ACTIONS = [
    Action.WAIT,
    Action.UP,
    Action.DOWN,
    Action.LEFT,
    Action.RIGHT,
]


class MAF_environment:
    def __init__(self,
                 namespace: str,         # Name of the environment instance
                 environment_type: str,  # Type of environment
                 map_reference: str = "Unknown",     # Reference to the map used
                 ):
        """
        RS environment class, used to generate RS environments
        """

        # ----- Setup reference properties
        self.id = id(self)
        self.name = namespace
        self.environment_type = environment_type
        self.map_reference = map_reference

    def __str__(self):
        return f"{self.id} - {self.environment_type} environment - {self.map_reference}"

    def __repr__(self):
        return self.__str__()

    # ===================================== Properties
    @property
    def maze_array(self):
        pass

    @property
    def state(self):
        return self._state

    @state.setter
    def state(self, state):
        # -> If goal reached, set goal to none
        if state == self.goal:
            self.goal = None

        self._state = state

    @property
    def goal(self):
        return self._goal

    @goal.setter
    def goal(self, goal):
        # -> Reset path to goal if new goal
        if goal != self._goal:
            self._path_to_goal = None
            self._goal = goal

    @property
    def path_to_goal(self):
        if self.goal is None:
            return []

        # -> Check if path must be recomputed
        recompute_path = False

        if self._path_to_goal is None:
            recompute_path = True
        elif self.state not in self._path_to_goal:
            recompute_path = True

        # -> Recompute path if necessary
        if recompute_path:
            # -> Compute new path to goal from state
            self._path_to_goal = self.compute_path_from_to(start=self.state, end=self.goal)

        else:
            # -> Drop all steps up to and including the current one
            index = self._path_to_goal.index(self.state)
            self._path_to_goal = self._path_to_goal[index:]

        return self._path_to_goal

    # -------------------- Abstract
    @property
    @abstractmethod
    def valid_start_positions(self) -> list:
        pass

    # ===================================== Prints
    @staticmethod
    def _render_state(image_array = None,
                      height_map = None,  
                      paths: list = [], 
                      positions: list = [],
                      POIs: list = [],
                      show: bool = True,
                      flat: bool = False
                      ):

        # -> Create matplotlib plot   
        # ----- 2D plot
        if height_map is None or flat is True:
            fig, ax = plt.subplots()
            
            # > Background
            if image_array is not None:
                ax.imshow(image_array)

            # > Colors
            positions_colors = cm.rainbow(np.linspace(0, 1, len(positions)))

            # > Paths
            if paths:
                for i, path in enumerate(paths):
                    ax.plot(*zip(*path), color=positions_colors[i])

            # > POIs
            if POIs:
                ax.scatter(*zip(*POIs), color="orange", s=8, marker="D")

            # > Positions
            if positions:
                ax.scatter(*zip(*positions), color=positions_colors, s=8)

            # -> Display plot
            if show:
                plt.show()

        # ----- 3D plot
        else:
            fig = plt.figure() 
            axes = fig.gca(projection ='3d') 

            # -> Create terrain surface
            terrain_colors = np.divide(image_array, 255)
            
            (x, y) = np.meshgrid(np.arange(terrain_colors.shape[1]), np.arange(terrain_colors.shape[0]))
            axes.plot_surface(x, y, height_map, rstride=4, cstride=4, facecolors=terrain_colors, linewidth=0)   

            axes.set_zlim3d(30, 400)                    # viewrange for z-axis should be [-4,4]

            # -> Display plot
            if show:
                plt.show()
        
        return fig, ax, image_array

    @staticmethod
    def gen_rainbow_color_list(length):
        colors = list(cm.rainbow(np.linspace(0, 1, length)))

        for color in range(len(colors)):
            colors[color] = list(colors[color])

            for value in range(len(colors[color])):
                colors[color][value] = int(255 * colors[color][value])

            colors[color] = colors[color][:-1]

        return colors

    @staticmethod
    def get_shade(rgb_color: tuple, shade_factor: int):
        rgb_color[0] *= shade_factor
        rgb_color[1] *= shade_factor
        rgb_color[2] *= shade_factor

        return rgb_color

    @staticmethod
    def hex_to_rgb(hex: str):
        print(hex)
        rgb = []
        for i in (0, 2, 4):
            decimal = int(hex[i:i + 2], 16)
            rgb.append(decimal)

        return tuple(rgb)

    @staticmethod
    def rgb_to_hex(r: int, g: int, bv):
        return '#{:02x}{:02x}{:02x}'.format(r, g, b)

    # -------------------- Abstract
    @abstractmethod
    def render_terrain(self, show: bool = True, paths: list = [], positions: list = []):
        pass

    @abstractmethod
    def render_comms(self, show: bool = True, paths: list = [], positions: list = []):
        pass

    @abstractmethod
    def animate(self,
                background: str = "terrain",    # -> "terrain" or "comms"
                paths: list = [],
                duration: int = 200,
                save_path: str = None
                ):
        pass

    @abstractmethod
    def render_path(self, path, positions: list = [], show: bool = True):
        pass

    # ===================================== Interfaces
    def reset(self):
        self.state = self.start_state

    @abstractmethod
    def step(self, action):
        pass

    @abstractmethod
    def compute_path_from_to(self, start, end):
        pass
