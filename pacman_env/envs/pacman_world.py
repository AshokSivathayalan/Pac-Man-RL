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
    def __init__(self, render_mode=None, grid=None):
        # Placeholder
        if grid == None:
            self.grid = np.array([[1, 2, 2, 2, 2],
                    [2, 2, 3, 3, 2],
                    [2, 2, 3, 3, 2],
                    [2, 3, 3, 3, 2],
                    [2, 2, 2, 2, 4]]) 
        else:
            self.grid = grid
         
        # The state information given to the algorithm
        self.observation_space = spaces.Dict({           
            "grid": spaces.Box(low = 0, high = 4, shape=self.grid.shape, dtype=np.int8), # State of the world
            "direction": spaces.Discrete(4) # Pacman's current direction
        })
         
        # The actions the algorithm can take
        self.action_space = spaces.Discrete(4)
        self.action_to_direction = {
            Actions.right.value: np.array([0, 1]),
            Actions.up.value: np.array([-1, 0]),
            Actions.left.value: np.array([0, -1]),
            Actions.down.value: np.array([11, 0]),
        }

        # For faster updating after step
        self.pacman_loc = np.argwhere(self.grid == Cells.pacman.value)[0]
        self.pacman_dir = Actions.down # Arbitrary? 
        self.ghost_loc = np.argwhere(self.grid == Cells.ghost.value)[0]
        self.remaining_pips = np.sum(self.grid == Cells.pip.value)

        # For graphics
        assert render_mode is None or render_mode in self.metadata["render_modes"]
        self.render_mode = render_mode

    
    def reset(self):
        return
    
    def step(self, action):
        terminated = False
    
        # Update ghosts based on Pacmans current location
        self._update_ghosts()

        # Update Pacman based on the chosen action
        self._update_pacman(action)

        # Calculating score
        score = 0
        if self.grid[self.pacman_loc] == Cells.pip.value:
            score = 1
            self.remaining_pips -= 1
        self.grid[self.pacman_loc] = Cells.pacman.value

        # Determining if env has terminated (if ghosts caught Pacman or if all pips are eaten)
        # Should do this earlier (before score)?
        # Given that ghost is faster, need to make sure it doesn't skip over pacman
        # Ex. 
        # 0 P G -> G 0 P because ghost moves 2 left, pacman moves 1 right
        # If all pips eaten, done game
        if self.remaining_pips == 0:
            terminated = True
        # If ghosts caught pacman, done game
        
        observation = self._get_obs()
        # observation is data given to algorithm
        # score is reward
        # terminated is if episode is over
        # 4th arg is if we truncate the episode (lasted too long & cut it short) 
        # 5th is extra info for debugging(?) 
        return observation, score, terminated, False, None
    
    def render(self):
        return
    
    def close(self):
        return
    
    def _valid_loc(self, new_loc):
        if not (new_loc[0] >= 0 and new_loc[0] < self.grid.shape[0] and new_loc[1] >= 0 and new_loc[1] < self.grid.shape[1]):
            return False #Out of bounds
        if self.grid[tuple(new_loc)] == Cells.wall.value:
            return False #Can't move into wall
        return True
    
    def _update_pacman(self, action):
        self.grid[self.pacman_loc] = Cells.empty.value # Cell we leave becomes empty
        movement = self.action_to_direction[action]
        new_loc = self.pacman_loc + movement

        if self._valid_loc(new_loc): # Next loc in-bounds, not blocked by wall
            self.pacman_loc = new_loc
            self.pacman_dir = action
        else: # Invalid action, continue in same direction
            movement = self.action_to_direction[self.pacman_dir]
            new_loc = self.pacman_loc + movement
            if self._valid_loc(new_loc): #Check that can keep moving in this direction
                self.pacman_loc = new_loc
            # If can't keep moving in old direction, Pacman location doesn't change

    # This function should update the position of the ghosts 
    def _update_ghosts(self):
        # Ghost should make the move that minimizes direct distance?
        # I.e. if ghost is x from pacman horizontally, y from pacman vertically
        # If x > y moves horizontally, if y > x moves vertically
        return 0
    
    # Function to determine if ghost caught pacman
    def _caught_pacman(self):
        return False
    
    def _get_obs(self):
        return {"grid": self.grid.copy(), "direction": self.pacman_dir.value}
