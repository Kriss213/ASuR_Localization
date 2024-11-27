__author__ = "Krišs Aleksandrs Vasermans"
"""
This file contains classes used for FastSLAM study project.

Classes: Environment, Robot, Particle, Landmark, FastSLAM
"""
from copy import deepcopy
from typing import List, Tuple, Union
from matplotlib.axes import Axes
from matplotlib.figure import Figure
from matplotlib.image import AxesImage
from matplotlib.collections import PatchCollection
from matplotlib.patches import FancyArrow
import numpy as np
import os
from termcolor import colored
import matplotlib.pyplot as plt
from matplotlib.text import Annotation
from scipy.stats.mstats import winsorize

os.system('color')

OBSTACLE = 1
FREE_SPACE = 0
PLT_WAIT = 1e-3

class Environment:
    """
    This class initializes and updates a new environment with given size.
    """
    cell_colors = {
        'X': 'red',
        '↓': 'green',
        '←': 'green',
        '↑': 'green',
        '→': 'green',
        '-': 'grey',
        '|': 'grey',
        ' ': 'white',
    }

    directions = {
        0: '→',
        90: '↑',
        180: '←',
        270: '↓'
    }

    def __init__(self, size:tuple=(10,10), map:str=None, p_obstacle:float=0.25):
        """
        Initialize a new environment.

        :param size: Grid size like (height, width)
        :param map: Load pre-generated map. If None, a new map is generated.
        :param p_obstacle: Chance of obstacle when generating new map.

        :return: An Environment instance.
        """

        if map:
            self._grid:np.ndarray = np.loadtxt(map).astype(np.int8)

            self._height:int = len(self._grid)
            self._width:int = len(self._grid[0])
            self._size:tuple = (self._height, self._width)
        else:
            self._size:tuple = size
            self._height:int = size[0]
            self._width:int = size[1]
            
            # 0 represents a free cell, 1 - an obstacle
            self._grid:np.ndarray = np.random.choice([FREE_SPACE,OBSTACLE], p=[1-p_obstacle, p_obstacle], size=(self._height,self._width)).astype(np.int8)

        # like grid but just keeps track of robot locations
        self.__robot_layer:list = [] 
        
        # parameters for graphical display
        self.__init_display()
        
    # getters and setters
    @property
    def size(self):
        return self._size
    @size.setter
    def size(self, nv):
        raise Exception(f"Variable 'size' is read-only!")
    @property
    def width(self):
        return self._width
    @width.setter
    def width(self, nv):
        raise Exception(f"Attribute 'width' is read-only!")
    @property
    def height(self):
        return self._height
    @height.setter
    def height(self, nv):
        raise Exception(f"Attribute 'height' is read-only!")
    @property
    def grid(self):
        return self._grid
    @grid.setter
    def grid(self, nv):
        raise Exception(f"Attribute 'grid' is read-only!")      

    def print(self, robots: List["Robot"]) -> None:
        """
        Print environment in its current state.

        :param robots: a list of Robots to display

        :raises Exception: if attempted to place robot on obstacle.
        """        
       
        # get robot locations
        # reset robot_layer
        self.__robot_layer = [[' ' for _ in range(self._width)] for i in range(self._height)]
        for robot in robots:

            if self._grid[robot.y, robot.x] == OBSTACLE:
                raise Exception(f"Space {(robot.x, robot.y)} is occupied.") 
            self.__robot_layer[robot.y][robot.x] = robot._get_direction_symbol()

        #grid_np = np.array(self._grid)
        robot_layer_np = np.array(self.__robot_layer)

        str_grid = np.where(robot_layer_np != ' ', robot_layer_np, self._grid)
        str_grid[str_grid == FREE_SPACE] = ' '
        str_grid[str_grid == OBSTACLE] = 'X'
        
        str_grid_list = str_grid.tolist()

        header = """
.-------------------------------------------------------------.
| _____            _                                      _   |
|| ____|_ ____   _(_)_ __ ___  _ __  _ __ ___   ___ _ __ | |_ |
||  _| | '_ \ \ / / | '__/ _ \| '_ \| '_ ` _ \ / _ \ '_ \| __||
|| |___| | | \ V /| | | | (_) | | | | | | | | |  __/ | | | |_ |
||_____|_| |_|\_/ |_|_|  \___/|_| |_|_| |_| |_|\___|_| |_|\__||
'-------------------------------------------------------------'"""
        print(colored(header, 'red'))
        for row in str_grid_list:
            row_len = 0
            row_str = colored("|    ", self.cell_colors['|'])
            row_len += len("|    ")
            for i, cell in enumerate(row):
                cell_str = 'X' if cell==str(OBSTACLE) else ' ' if cell==str(FREE_SPACE) else cell                        
                row_str += colored(f"{cell_str}", self.cell_colors[cell_str])
                row_len += len(f"{cell}")
                if i != len(row)-1:
                    row_str += colored("    |    ", self.cell_colors['|'])
                    row_len += len("    |    ")
            row_str += colored("    |", self.cell_colors['|'])
            row_len += len("    |")
        

            print(colored("-"*row_len, self.cell_colors['-']))
            print(row_str)
        print(colored("-"*row_len, self.cell_colors['-']))
        
        print(colored("="*row_len, 'red'), '\n')

    def __init_display(self):
        plt.ion()
        self.__fig, self.__ax = plt.subplots(1, 1, figsize=(5, 5))
        title = "Environment"
        self.__ax.set_title(title)
        
        self.__ax.imshow(self._grid, cmap='Greys')
        self.__ax.grid(True)
        self.__ax.set_xticklabels([])
        self.__ax.set_yticklabels([])
        self.__ax.set_xticks([-0.5+i for i in range(self._width)])
        self.__ax.set_yticks([-0.5+i for i in range(self._height)])
        self.__annotations:List[Annotation] = []
        
        plt.pause(PLT_WAIT)
        
    def display(self, robots: List["Robot"]=None) -> None:
        """
        Display environment in matplotlib window.
        
        :param robots: A list of Robot instances to place in environment
        """
        # Plot the environment matrix
        if robots:
            for ann in self.__annotations:
                ann.remove()
            self.__annotations.clear()
            
            for robot in robots:
                self.__annotations.append(
                    self.__ax.annotate(
                        self.directions[robot.theta],
                        (robot.x, robot.y),
                        fontsize=36,
                        color='red',
                        ha='center', va='center')
                )
        self.__fig.canvas.draw()  # Update the plot
        self.__fig.canvas.flush_events()
        plt.pause(PLT_WAIT)
       
    def save_map(self, file_name:str) -> None:
        """
        Save map to file.

        :param file_name: File path to save map to.
        """
        np.savetxt(file_name, self.grid, fmt='%d')

    def _get_perception_from_point(self, x:int, y:int, theta:int=None, as_coords:bool=False) -> dict:
        """
        Get distance to or coordinates of closest obstacle on all sides of point.
        
        :param x: X coordinate
        :param y: Y coordinate
        :param theta: Orientation (0, 90, 180, 270)
        :param as_coords: If true, return coordinates of observed obstacles
        """
        
        def get_dist_in_dir(x_dir, y_dir):
            # can only have 1 direction
            if (x_dir != 0 and y_dir != 0) or (x_dir==0 and y_dir==0):
                raise Exception(f"2 non-zero arguments provided: {x_dir}, {y_dir}")
            test_x, test_y = x, y
            index_err_or_obs = False
            while not index_err_or_obs:
                if x_dir:
                    test_x += x_dir
                else:
                    test_y += y_dir
                try:
                    if self._grid[test_y, test_x] == OBSTACLE:
                        index_err_or_obs = True
                except IndexError:
                    index_err_or_obs = True
            if x_dir != 0:
                return int(abs(test_x-x))
            else:
                return int(abs(test_y-y))
        
        
        front = rear = left = right = 0
        if theta == 0:
            front = get_dist_in_dir(1, 0)
            rear  = get_dist_in_dir(-1, 0)
            left = get_dist_in_dir(0, -1)
            right = get_dist_in_dir(0, 1)
            pos = [(x+front, y), (x, y-left), (x-rear, y), (x, y+right)]
        elif theta == 90:
            front = get_dist_in_dir(0, -1)
            rear  = get_dist_in_dir(0, 1)
            left = get_dist_in_dir(-1, 0)
            right = get_dist_in_dir(1, 0)
            pos = [(x, y-front), (x-left, y), (x, y+rear), (x+right, y)]
        elif theta == 180:
            front = get_dist_in_dir(-1, 0)
            rear  = get_dist_in_dir(1, 0)
            left = get_dist_in_dir(0, 1)
            right = get_dist_in_dir(0, -1)
            pos = [(x-front, y), (x, y+left), (x+rear, y), (x, y-right)]
        elif theta == 270:
            front = get_dist_in_dir(0, 1)
            rear  = get_dist_in_dir(0, -1)
            left = get_dist_in_dir(1, 0)
            right = get_dist_in_dir(-1, 0)
                    # front     #left       # rear       # right   
            pos = [(x, y+front), (x+left, y), (x, y-rear), (x-right, y)]
        else:
            raise Exception(f"Invalid theta value: {theta}")

        sensor = {
            "distances": [front, left, rear, right],
            "angles": [0, 90, 180, 270]
        }

        if as_coords:
            object_pos = {
                "positions": pos,
                "angles": [0, 90,180, 270]
            }
            return object_pos

        return sensor
    
class Robot:
    """
    Defines and keeps track of robot. Robot can be in an integer cell with orientation 0, 90, 180 or 270 degrees.
    
    This robot has a sensor that can measure distance in 4 directions (theta = 0 at facing direction).
    """
    def __init__(self, environment:Environment, sensor_noise:float=0.1, location:tuple=(0,0,0)):
        """
        Initialize a robot.
        
        :param location: Robot's initial location like (x, y, θ).
        :param environment: Environmet for checking, if move is possible.
        :param sensor_noise: Sensor noise per 1 unit.
        
        """
        x, y, theta = location
        self._x:int = x
        self._y:int = y
        self._theta:int = theta
        self.__environment:Environment = environment
        self.__sensor_noise:float = sensor_noise

        
        w, h = environment.width, environment.height
        if x >= w or w < 0 or y >= h or y < 0:
            raise ValueError(f"Given position is out of map: {x, y} for map with (w,h)={(w,h)}")

    @property
    def x(self):
        return self._x
    @property
    def y(self):
        return self._y
    @property
    def theta(self):
        return self._theta
    @x.setter
    def x(self, nv):
        raise Exception(f"Attribute 'x' is read-only!")
    @y.setter
    def y(self, nv):
        raise Exception(f"Attribute 'y' is read-only!") 
    @theta.setter
    def theta(self, nv):
        raise Exception(f"Attribute 'theta' is read-only!")
  
    def print_status(self):
        print(f"X: {self.x}, Y: {self.y}, Theta: {self.theta}")
    
    def turn(self, direction:str, verbose:bool=False) -> Union[Tuple[float, float, float], None]:
        """
        Turn robot left or right

        :param direction: A string like 'left' or 'right'.
        :param verbose: If True, print information about the turn.
        
        :raises ValueError: If invalid direction parameter given.
        :raises Exception: If environment is not set.

        :return diff: Tuple[dx, dy, dtheta]
        """
        if direction not in ['left', 'right']:
            raise ValueError(f"Invalid direction provided: {direction}!")
        
        current_theta = self._theta
        target_theta = None
        if direction == 'left':
            target_theta = (current_theta + 90) % 360
            diff = 90
        elif direction == 'right':
            target_theta = (current_theta - 90) % 360
            diff = -90
        
        self._theta:int = target_theta

        if verbose:
            line = "========================================="
            print(colored(line + f"\n|\tTurned from θ = {current_theta} to θ = {self._theta}\t|\n"+line, 'green'))

        return (0, 0, diff)
   
    def move(self, verbose:bool=False) -> Union[Tuple[float, float, float], None]:
        """
        Move the robot in given direction.
        
        :param verbose: Print information.

        :raises ValueError: If theta value is unexpected.

        :return diff: Tuple[dx, dy, dtheta] or None
        """
        
        # NOTE inverted y axis for printing in console
        new_x, new_y = self._x, self._y
        if self._theta == 90:
            new_y -= 1
        elif self._theta == 270:
            new_y += 1
        elif self._theta == 180:
            new_x -= 1
        elif self._theta == 0:
            new_x += 1
        else:
            raise ValueError(f"Unexpected theta value: {self._theta}!")
        
        # check, if new x,y is within bounds and not on obstacle
        line = "========================================="
        env_w, env_h = self.__environment.width, self.__environment.height
        if  new_y < env_h and new_y >= 0 and \
            new_x < env_w and new_x >=0 and \
            self.__environment.grid[new_y, new_x] != OBSTACLE:
            # clear to move
            if verbose:
                print(colored(line + f"\n|\tMoved from {(self._x, self._y)} to {(new_x, new_y)}\t|\n"+line, 'green'))
            
            
            prev_x, prev_y, prev_theta = self._x, self._y, self._theta
            self._x, self._y = new_x, new_y

            return (self._x - prev_x, self._y - prev_y, 0)
        else:
            if verbose:
                print(colored(line + f"\n|  Couldn't move from {(self._x, self._y)} to {(new_x, new_y)}  |\n"+line, 'red'))
            return None
           
    def get_perception(self, ret_coords:bool=False) -> Union[dict, Tuple[dict, dict]]:
        """
        Get robot's perception as:
        ```
        {
            distances: []
            angles: = [] 
        }
        ```

        Robot's facing direction is angle 0.

        :param ret_coords: Also return observed obstacle coordinates like:
        ```
        {
            positions: []
            angles: = [] 
        }
        ```

        :return dict | Tuple[dict, dict]: perception or (perception, coordinates)
        """

        def get_noise():
            return np.random.uniform(-self.__sensor_noise,  self.__sensor_noise)

        perception = self.__environment._get_perception_from_point(self._x, self._y, self._theta)

        # add noise to distance measurements
        perception["distances"] = [d+get_noise()*d for d in perception["distances"]]

        if ret_coords:
            coords = self.__environment._get_perception_from_point(self._x, self._y, self._theta, as_coords=True)
            return perception, coords
        
        return perception
    
# FastSLAM classes
class Landmark:
    """
    A landmark that can be observed by a particle
    """
    def __init__(self, id, mean:float, covariance:np.ndarray):
        """
        Initialize a new landmark.
        
        :param id: a *hashable* (int, tuple, etc.) value for identifying landmarks
        :param mean: mean vector
        :param covariance: covariance matrix
        
        """
        self.id = id
        self.mean:np.ndarray = mean # a vector with mean value for each dim
        self.covariance:np.ndarray = covariance

class Particle:
    """
    A single particle in fast slam system.
    """
    def __init__(self, x:int, y:int, theta:int, weight:float):
        self.x:int = x
        self.y:int = y
        self.theta:int = theta
        self.weight:float = weight
        
        self.landmarks:List[Landmark] = []
        
    def update_landmark(self, observed_landmarks:List[Landmark]) -> None:
        """
        Update particle weight (measurement model)
        If landmarks are new: initialize new landmark mean value and covariance.
        If landmark has been observed before - update it using Extended Kalman Filter.
        
        :param observed_landmarks: A list of observed landmarks.
        """
        for landmark in observed_landmarks:
            # check if landmark is known
            #landmark_is_known = any(lm.id == landmark.id for lm in self.landmarks)
            existing_landmark = next((lm for lm in self.landmarks if lm.id == landmark.id), None)
            
            if existing_landmark:               
                X_p = existing_landmark.mean
                P_p = existing_landmark.covariance
                
                H = np.eye(2)

                # observation noise
                Q = landmark.covariance
                
                # where the landmark should be based on current particle position and measurement
                Z = landmark.mean
                
                # innovation. Greater innovation -> greater loss of weight
                Y = Z - H @ X_p
                
                # innovation variance
                S = H @ P_p @ H.T + Q
                
                # Kalman gain
                K = P_p @ H.T @ np.linalg.inv(S)
                
                # update landmark
                existing_landmark.mean = X_p + K @ Y
                I = np.eye(2)
                existing_landmark.covariance = (I - K@H) @ P_p
                
                self.weight *= np.linalg.det(2*np.pi*S)**(-1/2) * np.exp(-1/2 * Y @ np.linalg.inv(S) @ Y.T)

            else:
                # add new landmark
                self.landmarks.append(deepcopy(landmark))

    def update_motion(self, motion:Tuple[float, float, float], motion_noise:float) -> None:
        """
        Update the particle's position based on robot's dx, dy, dtheta.
        
        :param motion: Motion as (dx, dy, dtheta)
        :param motion_noise: standard deviation for noise in motion
        """

        dx, dy, dtheta = motion

        self.x += dx + np.random.normal(0, motion_noise) * dx
        self.y += dy + np.random.normal(0, motion_noise) * dy
        self.theta += dtheta + np.random.normal(0, motion_noise) * (dtheta/10)
        self.theta = self.theta % 360

class FastSLAM:
    """
    Handle fast simultaneous localization and mapping.
    """
    def __init__(self, env_size:Tuple[int, int], approx_pos:Tuple[float, float, float], particle_count:int = 100):
        """
        Initialize a FastSLAM instance.
        
        :param particle_count: - Particle count.
        :param approx_pos: Generate initial particles around this position (x, y, theta).        
        :param env_size: - Size of the environment like (height, width)
        """

        # create a 2D array map representation.
        h, w = env_size
        self._map:np.ndarray = np.zeros((h, w))
        
        #self.__true_environment:Environment = environment

        # initialize new particles
        self.__init_x, self.__init_y, self.__init_theta = approx_pos
        offset = 1
        theta_offset = 30
        self.particles:List[Particle] = [
            Particle(
                np.random.uniform(self.__init_x - offset, self.__init_x + offset),
                np.random.uniform(self.__init_y - offset, self.__init_y + offset),
                np.random.uniform(self.__init_theta-theta_offset, self.__init_theta+theta_offset),
                weight=1.0) for _ in range(particle_count)]
       

        self.__init_display()
    
    def __init_display(self):
        plt.ion()
        fig_axs = plt.subplots(1, 1, figsize=(5, 5))
        self.__fig:Figure = fig_axs[0]
        self.__ax:Axes = fig_axs[1]
        title = "Fast SLAM environment"
        self.__ax.set_title(title)
        
        self.__im:AxesImage = self.__ax.imshow(self._map, cmap='Greys', vmin=0, vmax=1.0)
        self.__ax.grid(True)
        self.__ax.set_xticklabels([])
        self.__ax.set_yticklabels([])
        
        h, w = self._map.shape
        self.__ax.set_xticks([-0.5+i for i in range(w)])
        self.__ax.set_yticks([-0.5+i for i in range(h)])
        
        self.__arrows:List[FancyArrow] = []
        self.__particle_plot:PatchCollection = None
        self.__landmark_plot:List[PatchCollection] = []
        
        plt.pause(PLT_WAIT)
    
    def display(self):
        """
        Display currently mapped map with particles.
        """
        for arr in self.__arrows:
            arr.remove()
        self.__arrows.clear()
        if self.__particle_plot:
            self.__particle_plot.remove()
        
        for lm_plot in self.__landmark_plot:
            lm_plot.remove()
        self.__landmark_plot.clear()
        
        x = []
        y = []

        sizes = []
        
        for particle in self.particles:
            x.append(particle.x)
            y.append(particle.y)
            
            # plot arrow
            dx = np.cos(np.deg2rad(particle.theta))*0.2
            dy = np.sin(np.deg2rad(particle.theta+180))*0.2
            arrow = self.__ax.arrow(particle.x, particle.y, dx, dy, head_width=0.1, head_length=0.2, fc='red', ec='red')
            self.__arrows.append(arrow)
            sizes.append(20*particle.weight)

        
        self.__im.set_data(self._map)
        self.__particle_plot = self.__ax.scatter(x, y, s=sizes, color='red')
        

        self.__fig.canvas.draw()  # Update the plot
        self.__fig.canvas.flush_events()
        plt.pause(PLT_WAIT)

    def update_movement(self, diff:Tuple[float, float, float], motion_noise:float=0.2):
        """
        Update particle positions from robot movement.
        
        :param diff: Robot's displacement.
        :param motion_noise: Motion noise per one unit.
        """
        for particle in self.particles:
            particle.update_motion(motion=diff, motion_noise=motion_noise)

    def update_landmarks(self, measurements:Tuple[dict,dict], measurement_noise:float=0.1):
        """
        Update particle weights based on observations.

        :param measurement: Robot's measurements. A tuple like:
        ```
        (
            {
                distances: []
                angles: = [] 
            },
            {
                positions: []
                angles: = []
            }
        )
        ```
        :param measurement_noise: measurement noise per unit.

        """

        # For each particle construct a list of landmarks from measurements
        for p in self.particles:
            observed_landmarks:List[Landmark] = []
            distances:List[float] = []
            angles:List[float] = []

            for angle, distance, true_pos in zip(measurements[0]["angles"], measurements[0]["distances"], measurements[1]["positions"] ):
                # angle 0 is robot's facing direction.
                # true_pos is the correct (x, y) position of obstacle - used as unambiguous identificator.
                # each iteration of this loop = 1 landmark
                x = p.x + distance * np.cos(np.deg2rad(p.theta + angle))
                y = p.y - distance * np.sin(np.deg2rad(p.theta + angle))
                
                variance = np.abs(np.eye(2) * measurement_noise * distance)
                landmark = Landmark(
                    id = true_pos,
                    mean = np.array([x, y]),
                    covariance=variance
                )
                
                observed_landmarks.append(landmark)
                distances.append(distance)
                angles.append(angle)
            
            # Update landmarks for particle
            p.update_landmark(observed_landmarks=observed_landmarks)

    def resample_particles(self, normalize:bool=True):
        """
        Resample particles according to their weights using Stohastic Universal Resampling.

        :param normalize: Normalize particle weights to 0..1
        """
        particles = self.particles
        weights = np.array([p.weight for p in particles])
        
        # amount of particles to keep -> stays the same
        N = len(particles)
        
        # Normalize weights
        weights /= np.sum(weights)

        cumulative_sum = np.cumsum(weights)

        # generate pointers
        step = 1.0 / N
        start = np.random.uniform(0, step)
        pointers = start + step * np.arange(N)

        # resample particles
        resampled_particles:List[Particle] = []
        index = 0
        for p in pointers:
            while p > cumulative_sum[index]:
                index += 1
            resampled_particles.append(deepcopy(particles[index]))

        # normalize
        if normalize:
            rp_sum = np.sum([pr.weight for pr in resampled_particles])
            for particle in resampled_particles:
                particle.weight /= rp_sum + 1e-16
        
        self.particles = resampled_particles

        # update slam map
        self.update_map()

    def update_map(self):
        new_map = np.zeros_like(self._map)
        for particle in self.particles:
            for landmark in particle.landmarks:
                lm_x, lm_y = landmark.mean
                id_x, id_y = int(np.round(lm_x)), int(np.round(lm_y))
                h, w = new_map.shape
                
                if id_x < w and id_x >=0 and id_y < h and id_y >=0:
                    new_map[id_y, id_x] += particle.weight

        self._map += new_map
        # winsorize to avoid max outliers
        self._map = winsorize(self._map, limits=[0.0, 0.02])
        self._map /= np.max(self._map) + 1e-16
