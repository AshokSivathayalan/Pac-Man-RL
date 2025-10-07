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
        if grid == None:
            self._generate_grid()
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
            Actions.down.value: np.array([1, 0]),
        }
        # Track some positions, stats to make updates easier
        self._update_stats()

        # For graphics
        assert render_mode is None or render_mode in self.metadata["render_modes"]
        self.render_mode = render_mode

    
    def reset(self, seed=None, options=None):
        super().reset(seed=seed) # Required by library
        # New grid, update stats accordingly
        self._generate_grid()
        self._update_stats()

        # Returns observations, auxiliary info (5th val returned in step)
        return self._get_obs(), {}
    
    # Update grid after action chosen
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
            # Adding new pips if they drop too low
            if self.remaining_pips < self.pip_threshold:
                self._regenerate_pellets()
        self.grid[self.pacman_loc] = Cells.pacman.value

        # Determining if env has terminated (if ghosts caught Pacman)
        # Should do this earlier (before score)?
        # Given that ghost is faster, need to make sure it doesn't skip over pacman
        # Ex. 
        # 0 P G -> G 0 P because ghost moves 2 left, pacman moves 1 right
        # If ghosts caught pacman, done game
        if self._caught_pacman():
            terminated = True
        
        observation = self._get_obs()
        # observation is data given to algorithm
        # score is reward
        # terminated is if episode is over
        # 4th arg is if we truncate the episode (lasted too long & cut it short) 
        # 5th is extra info for debugging(?) 
        return observation, score, terminated, False, {}
    
    def render(self):
        return
    
    def close(self):
        # Only stuff we have to do here is if the graphics/rendering needs to be closed
        return
    
    def _valid_loc(self, new_loc):
        if not (new_loc[0] >= 0 and new_loc[0] < self.grid.shape[0] and new_loc[1] >= 0 and new_loc[1] < self.grid.shape[1]):
            return False #Out of bounds
        if self.grid[tuple(new_loc)] == Cells.wall.value:
            return False #Can't move into wall
        return True
    
    def _update_pacman(self, action):
        self.grid[tuple(self.pacman_loc)] = Cells.empty.value # Cell we leave becomes empty
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

    # A function to update tracked values about state
    def _update_stats(self):
         # For faster updating after step
        self.pacman_loc = np.argwhere(self.grid == Cells.pacman.value)[0]
        self.pacman_dir = Actions.down.value # Arbitrary? 
        self.ghost_loc = np.argwhere(self.grid == Cells.ghost.value)[0]
        self.remaining_pips = np.sum(self.grid == Cells.pip.value)
        self.pip_threshold = (self.remaining_pips//5) + 1 # Fifth of starting pips, rounded up (can change later)

    # A function to add more pellets once they drop below a threshold
    def _regenerate_pellets(self):
        # Get all empty spaces, pick some at random and add pips
        # Add half of pip_threshold? i.e. repopulate 10% of space at random
        empty = np.argwhere(self.grid == Cells.empty.value)
        n_add = self.pip_threshold//2 + 1
        chosen = empty[np.random.choice(len(empty), n_add, replace=False)]
        self.grid[chosen[:,0], chosen[:,1]] = Cells.pip.value

    # Function to refresh grid after episode finished
    # This is a placeholder
    def _generate_grid(self):
        self.grid = np.array(
                   [[1, 2, 2, 2, 2],
                    [2, 2, 3, 3, 2],
                    [2, 2, 3, 3, 2],
                    [2, 3, 3, 3, 2],
                    [2, 2, 2, 2, 4]]) 