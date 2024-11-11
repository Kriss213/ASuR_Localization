__author__ = "Krišs Aleksandrs Vasermans"
"""
This file contains classes used for Markov's localization study project.

Classes: Environment, Robot, MarkovLocalization
"""
from typing import List
import matplotlib
from matplotlib.axes import Axes
from matplotlib.axes._axes import Axes as Ax
from matplotlib.figure import Figure
from matplotlib.image import AxesImage
import numpy as np
import os
from termcolor import colored
import matplotlib.pyplot as plt
from matplotlib.text import Annotation
os.system('color')

OBSTACLE = 1
FREE_SPACE = 0
PLT_WAIT = 0.2
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
        self.__annotations:List[Annotation] = []# = self.__ax.annotate('',(0,0))
        plt.show()
        
    def display(self, robots: List["Robot"]) -> None:
        """
        Display environment in matplotlib window.
        
        :param robots: A list of Robot instances to place in environment
        """
        
        # Plot the environment matrix
        for ann in self.__annotations:
            ann.remove()
        self.__annotations.clear()
        
        for robot in robots:
            self.__annotations.append(
                self.__ax.annotate(
                    robot._get_direction_symbol(),
                    (robot.x, robot.y),
                    fontsize=36,
                    color='red',
                    ha='center', va='center')
            )
        

        plt.pause(PLT_WAIT)
       
    def save_map(self, file_name:str) -> None:
        """
        Save map to file.

        :param file_name: File path to save map to.
        """
        np.savetxt(file_name, self.grid, fmt='%d')

    def _get_perception_from_point(self, x:int, y:int, theta:int) -> dict:
        """
        Get distance to closest obstacle on all sides of point.
        
        :param x: X coordinate
        :param y: Y coordinate
        :param theta: Orientation (0, 90, 180, 270)
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
        elif theta == 90:
            front = get_dist_in_dir(0, -1)
            rear  = get_dist_in_dir(0, 1)
            left = get_dist_in_dir(-1, 0)
            right = get_dist_in_dir(1, 0)
        elif theta == 180:
            front = get_dist_in_dir(-1, 0)
            rear  = get_dist_in_dir(1, 0)
            left = get_dist_in_dir(0, 1)
            right = get_dist_in_dir(0, -1)
        elif theta == 270:
            front = get_dist_in_dir(0, 1)
            rear  = get_dist_in_dir(0, -1)
            left = get_dist_in_dir(1, 0)
            right = get_dist_in_dir(-1, 0)
        else:
            raise Exception(f"Invalid theta value: {theta}")

        obstacle_dists = {
            "left": left,
            "right": right,
            "front": front,
            "rear": rear,
        }
        return obstacle_dists
    
    def _get_perception_from_point_old(self, x:int, y:int, theta:int) -> list:
        """
        ### Get environment perception from specific point of view and orientation.
        ### List start from position's rear-left side.

        ## Example:
        (F - free cell, X - obstacle).\n
        +---+---+---+\n
        | X | F | X |\n
        +---+---+---+\n
        | X | ↑ | F |\n
        +---+---+---+\n
        | F | X | X |\n
        +---+---+---+\n

        Perception vector starts from top left corner and goes around the robot in clockwise direction.
        #### Example above would give:
        ```
        [0, 1, 1, 0, 1, 0, 1, 1]
        ```

        :param x: X coordinate.
        :param y: Y coordinate.
        :param theta: The direction faced (0: right, 90: up, 180: left, 270: down).

        :return percetption: A binary list. 0 - free space, 1 - obstacle.
        """

        obstacles_around_point = []

        # check cells around robot
        def check_cell(x_, y_):
            if (x_, y_) == (x, y):
                return
            
            if y_ < 0 or y_ >= self._height or x_ < 0 or x_ >= self._width:
                res = 1
            elif self._grid[y_, x_] == OBSTACLE:
                res = 1
            else:
                res=0
            obstacles_around_point.append(res)

        cut = 3
        for direction in [1, -1]:
            cut -= 1
            y_ = y - direction
            for x_ in range(x-1, x+2)[:cut:direction]:
                check_cell(x_, y_)

            x_ = x + direction
            for y_ in range(y-1, y+2)[:cut:direction]:
                check_cell(x_, y_)
            cut -= 1
        
        # rotate list
        for _ in range(int(theta / 90.0)):
            obstacles_around_point = obstacles_around_point[6:] + obstacles_around_point[:6] #rotated_obstacles
        
        return obstacles_around_point

class Robot:
    """
    Defines and keeps track of robot. Robot can be in an integer cell with orientation 0, 90, 180 or 270 degrees.
    
    This robot assumes a perception model where it can see all 8 surrounding cells.
    """
    def __init__(self, environment:Environment, location:tuple=(0,0,0), sensor_noise:float=0.1, odom_noise:float=0.1):
        """
        Initialize a robot.
        
        :param location: Robot's initial location like (x, y, θ).
        :param environment: Environmet for checking, if move is possible.
        :param sensor_noise: Chance of each cell being percieved incorrectly.
        :param odom_noise: Chance of robot not realizing it moved.
        
        """
        x, y, theta = location
        self._x:int = x
        self._y:int = y
        self._theta:int = theta
        self._previous_x:int = x
        self._previous_y:int = y
        self._previous_theta:int = theta
        self.__environment:Environment = environment
        self.sensor_noise:float = sensor_noise
        self.odom_noise:float = odom_noise
        
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
    @property
    def previous_x(self):
        return self._previous_x
    @property
    def previous_y(self):
        return self._previous_y
    @property
    def previous_theta(self):
        return self._previous_theta
    @x.setter
    def x(self, nv):
        raise Exception(f"Attribute 'x' is read-only!")
    @y.setter
    def y(self, nv):
        raise Exception(f"Attribute 'y' is read-only!") 
    @theta.setter
    def theta(self, nv):
        raise Exception(f"Attribute 'theta' is read-only!")
    @previous_x.setter
    def previous_x(self, nv):
        raise Exception(f"Attribute 'previous_x' is read-only!")
    @previous_y.setter
    def previous_y(self, nv):
        raise Exception(f"Attribute 'previous_y' is read-only!")
    @previous_theta.setter
    def previous_theta(self, nv):
        raise Exception(f"Attribute 'previous_theta' is read-only!") 

    def print_status(self):
        print(f"X: {self.x}, Y: {self.y}, Theta: {self.theta}")
        print(f"X_prev: {self.previous_x}, Y_prev: {self.previous_y}, Theta_prev: {self.previous_theta}")
        
    def _only_rotated(self) -> bool:
        return self._x == self._previous_x and \
                self._y == self._previous_y and \
                self._theta != self._previous_theta
    
    def _no_movement(self) -> bool:
        return self._x == self._previous_x and \
                self._y == self._previous_y and \
                self._theta == self._previous_theta
    
    def turn(self, direction:str, verbose:bool=False) -> bool:
        """
        Turn the robot to given direction, if possible. The Robot can only turn 90 degrees.
        
        :param direction: A string like 'up', 'down', 'left' or 'right'.
        :param verbose: If True, print information about the turn.
        
        :raises ValueError: If invalid direction parameter given.
        :raises Exception: If environment is not set.
        
        :return bool: True, if move was successfull, False otherwise.
        """
        
        if direction not in ['up', 'down', 'left', 'right']:
            raise ValueError(f"Invalid direction provided: {direction}!")
        
        dir_theta = {
            'up': 90,
            'down':270,
            'left':180,
            'right':0
        }
        target_theta = dir_theta[direction]
        current_theta = self._theta
        condition = current_theta == (target_theta+90)%360 or current_theta == (target_theta-90)%360
        if condition:
            self._previous_theta:int = current_theta
            self._previous_x:int = self._x
            self._previous_y:int = self._y
            self._theta:int = target_theta
        
        if verbose:
            line = "========================================="
            if condition:
                print(colored(line + f"\n|\tTurned from θ = {current_theta} to θ = {self._theta}\t|\n"+line, 'green'))
            else:
                print(colored(line + f"\n|\tCouldn't turn from θ = {current_theta} to θ = {self._theta}\t|\n"+line, 'red'))
        
        return condition
   
    def move(self, verbose:bool=False) -> bool:
        """
        Move the robot in given direction.
        
        :param verbose: Print information.

        :raises ValueError: If theta value is unexpected.

        :return bool: True, if move was successfull, False otherwise. 
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
            
            
            self._previous_x, self._previous_y = self._x, self._y
            self._x, self._y = new_x, new_y

            return True
        else:
            if verbose:
                print(colored(line + f"\n|  Couldn't move from {(self._x, self._y)} to {(new_x, new_y)}  |\n"+line, 'red'))
            return False
        
    def _get_direction_symbol(self) -> str:
        """
        Get an arrow indicating robots direction.
        
        :raises ValueError: If theta value is unexpected.

        :return arrow: An arrow symbol.
        """
        theta = self._theta
        if theta == 0:
            return '→'
        elif theta == 90:
            return '↑'
        elif theta == 180:
            return '←'
        elif theta == 270:
            return '↓'
        else:
            raise ValueError(f"Unexpected theta value: {theta}!")
    
    def get_perception(self) -> dict:
        """
        Get robot's perception with sensor and odometry noise.
        """
        # handle odometry error
        if np.random.choice([True, False],1,p=[1-self.odom_noise, self.odom_noise]):
            perception = self.__environment._get_perception_from_point(self._x, self._y, self._theta)
        else:
            perception = self.__environment._get_perception_from_point(self._previous_x, self._previous_y, self._theta)
            
        for key, point in perception.items():
            #cor_x, cor_y = point
            odds = [self.sensor_noise, 1-self.sensor_noise]
            if np.random.choice([True, False], 1, p=odds):
                add_sub = np.random.choice([-1, 1], 1, [0.5, 0.5])
                perception[key] = point + add_sub        
        
        return perception
        
class MarkovLocalization:
    """
    Perform Markov localization for a robot
    """
    def __init__(self, robot:Robot, environment:Environment, perception_chance:float=0.9):
        """
        Initialize Markov localization class.

        :param robot_instance: An instance of class Robot.
        :param perception_chance: Chance of accurate perception.
        :param move_chance: Chance of odometry being correct.

        """
        self.__robot:Robot = robot
        self.__environment:Environment = environment
        self.__perception_chance:float = perception_chance

        w, h = self.__environment.width, self.__environment.height
        configuration_count = h * w * 4

        init_prob = 1.0 / configuration_count

        # initialize configuration space as a 3D array
        self.__belief:np.ndarray = np.ones((h, w, 4)) * init_prob
        self.__indices:list = list(np.ndindex(self.__belief.shape))
       
        self.__init_display()

    @property
    def belief(self):
        return self.__belief
    @belief.setter
    def belief(self, nv):
        raise Exception(f"Attribute 'belief' is read-only!")
    
    def __init_display(self):
        plt.ion()
        
        fig_axs = plt.subplots(2, 2, figsize=(6, 6))
        self.__fig:Figure = fig_axs[0]
        self.__axs:Axes = fig_axs[1]
        self.__fig.suptitle("Probability distribution")
        titles = ["θ = 0° (→)", "θ = 90° (↑)", "θ = 180° (←)","θ = 270° (↓)"]
        self.__ims:List[AxesImage] = []
        for i, ax in enumerate(self.__axs.flat):
            ax:Ax = ax # for code suggestions
            ax.set_title(titles[i])
            im = ax.imshow(self.__belief[:,:,i], cmap='viridis', vmin=0, vmax=1)
            self.__ims.append(im)
            ax.grid(True)
            ax.set_xticklabels([])
            ax.set_yticklabels([])
            ax.set_xticks([-0.5+i for i in range(self.__environment.width)])
            ax.set_yticks([-0.5+i for i in range(self.__environment.height)])
            #self.__ann = self.__ax.annotate('',(0,0))
        self.__fig.colorbar(im, ax=self.__axs.ravel().tolist())
        plt.show()
        
    def display(self):
        
        for i, im in enumerate(self.__ims):
            #ax.imshow(self.__belief[:,:,i], cmap='viridis', vmin=0, vmax=1)#, cmap='Greys')
            im.set_data(self.__belief[:,:,i])
            
       
        #plt.show()
        plt.pause(PLT_WAIT)
            
    def print(self):
        """
        Prints cell probabilities for each cell
        """
        
        header = """
.---------------------------------------------------------.
| ____            _           _     _ _ _ _   _           |
||  _ \ _ __ ___ | |__   __ _| |__ (_) (_) |_(_) ___  ___ |
|| |_) | '__/ _ \| '_ \ / _` | '_ \| | | | __| |/ _ \/ __||
||  __/| | | (_) | |_) | (_| | |_) | | | | |_| |  __/\__ \|
||_|   |_|  \___/|_.__/ \__,_|_.__/|_|_|_|\__|_|\___||___/|
'---------------------------------------------------------'
"""
        print(colored(header, 'red'))
        for i in range(4):
            theta_header = "=========================================\n" + \
                    f"|\t\tθ = {(i)*90}°  \t\t|\n" + \
                    "========================================="
            print(colored(theta_header, 'yellow'))
            conf_space_layer = self.__belief[:,:,i].tolist()
            max_prob = np.max(self.__belief)

            eps = 1e-4 # if diff is smaller than eps, consider that cell matches max value
            for row in conf_space_layer:
                row_len = 0
                row_str = "| "
                row_len += len("| ")
                for i, cell in enumerate(row):                    
                    row_str += colored(f"{cell:.4f}", 'green' if np.abs(cell-max_prob) < eps else 'white')
                    row_len += len(f"{cell:.4f}")
                    if i != len(row)-1:
                        row_str += " | "
                        row_len += len(" | ")
                row_str += " |"
                row_len += len(" |")
            

                print("-"*row_len)
                print(row_str)
            print("-"*row_len)
        print(colored("="*row_len, 'red'), '\n')
    
    def __rotation_odometry_model(self, x:int, y:int, theta:int) -> float:
        """
        Calculate new belief from rotation odometry. This model assumes, that robot
        could only rotate to given state from states with +-90 deg. with 40% chance each.
        Add 10% chance for no rotation for noise.
        
        :param x: X position.
        :param y: Y position.
        :param theta: Robot orientation.
        
        :raises ValueError: If invalid theta value.
        
        :return new belief: For cell x, y, theta.
        """
        if theta not in [0, 90, 180, 270]:
            raise ValueError(f"Invalid theta value: {theta}")
        
        # it could rotate to current pos only from theta+-90, both with 45%. 10% it didn't rotate
        new_belief = 0
        prev_thetas = [(theta-90+360)%360, theta, (theta+90+360)%360]
        probs = [0.45, 0.1, 0.45]
        for prob, prev_theta in zip(probs, prev_thetas):
            prev_theta_idx = int(prev_theta / 90.0)
            prev_belief = self.__belief[y, x, prev_theta_idx]
            new_belief += prob * prev_belief
        
        return new_belief
    
    def __movement_odometry_model(self, x:int, y:int, theta:int) -> float:
        """
        Update belief with sensor model:
        
        +---+-----+---+                 +---+-----+-----+\n
        | 0 | 0   | 0 |                 | 0 | 0.0 | 0.05 |\n
        +---+-----+---+                 +---+-----+-----+\n
        | 0 | 1.0 | 0 |  (dx, dy) -->   | 0 | 0.1 | 0.85 |\n
        +---+-----+---+                 +---+-----+-----+\n
        | 0 | 0   | 0 |                 | 0 | 0.0 | 0.05|\n
        +---+-----+---+                 +---+-----+-----+\n
        
        :param x: X position.
        :param y: Y position.
        :param theta: Robot orientation.
        
        :raise ValueError: If invalid theta value encountered.
        
        :return new belief: For cell x, y, theta
        """
        #prev_theta = self.__robot.previous_theta
        theta_ind = int(theta / 90.0)
        
        prob_fw = 0.8
        prob_stay = 0.1
        prob_lf = 0.05
        prob_rf = 0.05
               
        # get adjacent cells and calculate new belief.
        if theta == 0:
            x_fw, y_fw = x, y
            x_stay, y_stay = x-1, y
            x_lf, y_lf = x, y-1
            x_rf, y_rf = x, y+1
        elif theta == 90:
            x_fw, y_fw = x, y
            x_stay, y_stay = x, y+1
            x_lf, y_lf = x-1, y
            x_rf, y_rf = x+1, y
        elif theta == 180:
            x_fw, y_fw = x, y
            x_stay, y_stay = x+1, y
            x_lf, y_lf = x, y+1
            x_rf, y_rf = x, y-1
        elif theta == 270:
            x_fw, y_fw = x, y
            x_stay, y_stay = x, y-1
            x_lf, y_lf = x+1, y
            x_rf, y_rf = x-1, y
        else:
            raise ValueError(f"Invalid theta value: {theta}")
        
        # dict matching point and probablity it came from it:
        # fw is the new point robot came to
        point_probs = {
            (x_fw, y_fw): prob_stay, # e.g. probability that robot got to new point by already being there
            (x_stay, y_stay): prob_fw,
            (x_lf, y_lf): prob_lf,
            (x_rf, y_rf): prob_rf
        }
        new_belief = 0
        # sum all possible ways to get to xy from adjacent cells
        for point, prob in point_probs.items():
            try:
                x_, y_ = point
                # get probability from adjacent cell
                # where it is possibly coming from
                prev_belief = self.__belief[y_, x_, theta_ind]
                
                # add the chance it came from there
                new_belief += prev_belief * prob
            except IndexError:
                # likely a point is out of bounds
                continue
        # set the new belief in new belief array
        return new_belief
        
    def __predict(self) -> None:
        """
        Prediction step. Updates belief space based on odometry.
        """
        # if no movement was done, return
        if self.__robot._no_movement():
            return

        # new belief array
        new_beliefs = np.zeros_like(self.__belief)
                
        for y, x, theta_ind in self.__indices:
            theta = int(theta_ind * 90)
            
            # check if robot's previous pos == current pos
            # it means it just rotated without moving
            if self.__robot._only_rotated():
                
                # apply odometry model for only rotating
                new_belief = self.__rotation_odometry_model(x, y, theta)
                   
            else:
                # robot moved
                new_belief = self.__movement_odometry_model(x, y, theta)
            
            
            new_beliefs[y,x,theta_ind] = new_belief
            
        self.__belief = new_beliefs.copy()
    
    def __correct(self, print_robot_perception:bool=False) -> None:
        """
        Correct belief space based on sensor perception.
        
        :param print_robot_perception: Print robot's perception.
        
        """
        # get robot perception
        robot_perception = self.__robot.get_perception()
        
        if print_robot_perception:
            x,y,theta = self.__robot.x, self.__robot.y, self.__robot.theta
            title_str = colored(f"""======================================================
| Robot's perception at position X: {x}, Y: {y}, θ: {theta} |
======================================================""",'red')
            print(title_str)
            for dir, dist in robot_perception.items():
                #print(dir, dist)
                
                print(f"{dir}:\t{int(dist)}")
            print(colored('======================================================','red'))

        for y, x, theta_ind in self.__indices:
            theta = int(theta_ind * 90)
            point_perception = self.__environment._get_perception_from_point(x, y, theta)

            if robot_perception == point_perception:
                self.__belief[y,x,theta_ind] *= self.__perception_chance
            else:
                self.__belief[y,x,theta_ind] *= (1-self.__perception_chance)        
        
    def update(self,print_robot_perception:bool=False) -> None:
        """
        Update belief space probabilities.
        
        :param print_robot_perception: Print robot's perception.
        
        """

        self.__predict()
        # normalize
        self.__belief /= (np.sum(self.__belief) + 1e-16)
        
        self.__correct(print_robot_perception)
        
        # normalize
        self.__belief /= (np.sum(self.__belief) + 1e-16)
        