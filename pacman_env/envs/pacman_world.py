from enum import Enum
import gymnasium as gym
from gymnasium import spaces
import numpy as np

class Actions(Enum):
    right = 0
    up = 1
    left = 2
    down = 3

class Cells(Enum):
    empty = 0
    pacman = 1
    pip = 2
    wall = 3
    ghost = 4


class PacManWorld(gym.Env):
    metadata = {"render_modes": ["human"], "render_fps": 4}
    def __init__(self, render_mode=None, maze=None):
        # Placeholder
        if maze == None:
            maze = [[1, 2, 2, 2, 2],
                    [2, 2, 3, 3, 2],
                    [2, 2, 3, 3, 2],
                    [2, 3, 3, 3, 2],
                    [2, 2, 2, 2, 4]] 
    
        self.maze = maze
        self.observation_space = spaces.Dict({           
            "grid": spaces.Box(low = 0, high = 4, shape=maze.shape, dtype=np.int8), # State of the world
            "direction": spaces.Discrete(4) # Pacman's current direction
        })

        self.action_space = spaces.Discrete(4)

        self.action_to_direction = {
            Actions.right.value: np.array([1, 0]),
            Actions.up.value: np.array([0, 1]),
            Actions.left.value: np.array([-1, 0]),
            Actions.down.value: np.array([0, -1]),
        }
        # For faster updating after step
        self.pacman_loc = (0, 0) # Search for '1' in grid?
        self.pacman_dir = Actions.down # Arbitrary? 

        assert render_mode is None or render_mode in self.metadata["render_modes"]
        self.render_mode = render_mode

        
        return
    
    def reset(self):
        return
    
    def step(self, action):
        # Update ghosts based on Pacmans current location
        self._update_ghosts()

        # Update Pacman based on the chosen action
        self.grid[self.pacman_loc] = Cells.empty
        direction = self.action_to_direction[action]
        new_loc = self.pacman_loc + direction

        if self._valid_loc(new_loc): # Next loc in-bounds, not blocked by wall
            self.pacman_loc = new_loc
            self.pacman_dir = action
        else: # Invalid action, continue in same direction
            direction = self.action_to_direction[self.pacman_dir]
            new_loc = self.pacman_loc + direction
            if self._valid_loc(new_loc): #Check that can keep moving in this direction
                self.pacman_loc = new_loc
            # If can't keep moving, Pacman location doesn't change

        # Calculating score
        score = 0
        if self.grid[self.pacman_loc] == Cells.pip:
            score = 1
        self.grid[self.pacman_loc] = Cells.pacman

        # Determining if env has terminated (if ghosts caught Pacman or if all pips are eaten)
        # Should do this earlier (before score)?
        # Given that ghost is faster, need to make sure it doesn't skip over pacman
        # Ex. 
        # 0 P G -> G 0 P because ghost moves 2 left, pacman moves 1 right
        terminated = False
        
        # observation is data given to algorithm
        # score is reward
        # terminated is if episode is over
        # 4th arg is if we truncate the episode 
        # 5th is extra info for debugging
        return observation, score, terminated, False, None
    
    def render(self):
        return
    
    def close(self):
        return
    
    def _valid_loc(self, new_loc):
        if not (new_loc[0] >= 0 and new_loc[0] < self.maze.shape[0] and new_loc[1] >= 0 and new_loc[1] < self.maze.shape[1]):
            return False #Out of bounds
        if self.grid[new_loc] == Cells.wall:
            return False #Can't move into wall
        return True
    
    # This function should update the position of the ghosts 
    def _update_ghosts(self):
        # Ghost should make the move that minimizes direct distance?
        # I.e. if ghost is x from pacman horizontally, y from pacman vertically
        # If x > y moves horizontally, if y > x moves vertically
        return 0
