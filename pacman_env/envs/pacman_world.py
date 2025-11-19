from enum import Enum
import gymnasium as gym
from gymnasium import spaces
import numpy as np
import math
import pygame

class Actions(Enum):
    right = 0
    up = 1
    left = 2
    down = 3

    def opposite(self):
        return Actions((self.value+2)%4)

class Cells(Enum):
    empty = 0
    pacman = 1
    pip = 2
    wall = 3
    ghost = 4
    dead = 5 # Maybe good idea for rendering, algorithm will only see this at episode end so should have no impact

class PacManWorld(gym.Env):
    metadata = {"render_modes": ["human"], "render_fps": 4}
    def __init__(self, render_mode=None, grid=None):
        self.steps_since_pellet = 0
        self.last_pos = None
        self.stay_streak = 0
        if grid == None:
            self._generate_grid()
        else:
            self.grid = grid

        # The state information given to the algorithm
        self.observation_space = spaces.Dict({           
            "grid": spaces.Box(low = 0, high = 5, shape=self.grid.shape, dtype=np.int8), # State of the world
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
        self.np_random, _ = gym.utils.seeding.np_random(seed)
        # New grid, update stats accordingly
        self._generate_grid()
        self._update_stats()
        self.steps_since_pellet = 0
        self.last_pos = None
        self.stay_streak = 0

        # Returns observations, auxiliary info (5th val returned in step)
        return self._get_obs(), {}
    
    # Update grid after action chosen
    def step(self, action):
        truncated = False

        reward = -0.1 # small living cost to discourage camping
        d0 = self.nearest_pip_dist()
        
        # Update Pacman based on the chosen action
        self._update_pacman(action)
        if self._caught_pacman(): # If pac-man moves into ghost
            self.grid[tuple(self.pacman_loc)] = Cells.dead.value
            # Return early
            return self._get_obs(), -5, True, False, {}
        
        d1 = self.nearest_pip_dist()

        # Update ghosts based on Pacmans current location
        # This also determines if ghosts caught pacman and episode has terminated
        if self.ghost_had_pip:
            self.grid[tuple(self.ghost_loc)] = Cells.pip.value
        else:
            self.grid[tuple(self.ghost_loc)] = Cells.empty.value
        self._update_ghosts()
        if self.terminated:
            return self._get_obs(), -5, True, False, {}
        self.ghost_had_pip = (self.grid[tuple(self.ghost_loc)] == Cells.pip.value)
        self.grid[tuple(self.ghost_loc)] = Cells.ghost.value

        # Calculating score
        if self.grid[tuple(self.pacman_loc)] == Cells.pip.value:
            reward += 1.0
            self.remaining_pips -= 1
            self.steps_since_pellet = 0
            # Adding new pips if they drop too low
            if self.remaining_pips < self.pip_threshold:
                self._regenerate_pips()
        else:
            self.steps_since_pellet += 1
        self.grid[tuple(self.pacman_loc)] = Cells.pacman.value
        
        # penalize same itle streaks to avoid camping
        pos = tuple(self.pacman_loc)
        if pos == self.last_pos:
            self.stay_streak += 1
        else:
            self.stay_streak = 0
        self.last_pos = pos
        
        if self.stay_streak >= 3:
            reward += -0.05 * self.stay_streak
        
        if self.steps_since_pellet >= 100:
            reward += -1.0
            truncated = True
            
        reward += 0.02*(d0-d1)
        
        # observation is data given to algorithm
        # score is reward
        # 3rd is if episode is over
        # 4th arg is if we truncate the episode (lasted too long & cut it short) 
        # 5th is extra info for debugging(?) 
        return self._get_obs(), reward, False, truncated, {}
    
    def render(self):
        if self.render_mode not in ("human", "rgb_array"):
            return
        return self._render_frame()

    def _render_frame(self):

        H, W = self.grid.shape

        if getattr(self, "_tile", None) is None:
            # Size of one cell in pixels we can change this later on once we decide how big we want
            self._tile = 48

        surf_w, surf_h = W * self._tile, H * self._tile

        if self.render_mode == "human":
            if getattr(self, "_window", None) is None:
                pygame.init()
                pygame.display.init()
                self._window = pygame.display.set_mode((surf_w, surf_h))
                pygame.display.set_caption("Pacman (Gymnasium)")
            if getattr(self, "_clock", None) is None:
                self._clock = pygame.time.Clock()

        # Offscreen surface we draw on every frame
        canvas = pygame.Surface((surf_w, surf_h))
        canvas.fill((10, 10, 10))  # background

        # Colors
        WALL   = (40, 40, 140)
        PELLET = (255, 215, 0) 
        PACMAN = (255, 255, 0)
        GHOST  = (220, 50, 50)
        DEAD   = (220, 220, 220)
        FLOOR  = (30, 30, 30)
        GRID   = (50, 50, 50)

        tag_by_val = {
            Cells.empty.value:  "empty",
            Cells.wall.value:   "wall",
            Cells.pip.value:    "pellet",
            Cells.ghost.value:  "ghost",
            Cells.pacman.value: "pacman",
            Cells.dead.value:   "dead",
        }

        # Draw grid
        t = self._tile
        r2 = t // 2

        for r in range(H):
            for c in range(W):
                x, y = c * t, r * t
                tag = tag_by_val.get(int(self.grid[r, c]), "empty")

                # floor tile
                pygame.draw.rect(canvas, FLOOR, (x, y, t, t))

                if tag == "wall":
                    pygame.draw.rect(canvas, WALL, (x, y, t, t))
                elif tag == "pellet":
                    # small dot
                    pygame.draw.circle(canvas, PELLET, (x + r2, y + r2), max(3, t // 8))
                elif tag == "ghost":
                    # ghost body (circle) + base
                    pygame.draw.circle(canvas, GHOST, (x + r2, y + r2), t // 3)
                    pygame.draw.rect(canvas, GHOST, (x + t // 6, y + r2, t * 2 // 3, t // 3))
                elif tag == "pacman":
                    # pacman circle (no mouth cut for simplicity)
                    pygame.draw.circle(canvas, PACMAN, (x + r2, y + r2), t // 3)
                elif tag == "dead":
                    pygame.draw.line(canvas, DEAD, (x + t // 4, y + t // 4), (x + 3 * t // 4, y + 3 * t // 4), 3)
                    pygame.draw.line(canvas, DEAD, (x + 3 * t // 4, y + t // 4), (x + t // 4, y + 3 * t // 4), 3)

        # grid lines
        for i in range(H + 1):
            pygame.draw.line(canvas, GRID, (0, i * t), (surf_w, i * t), 1)
        for j in range(W + 1):
            pygame.draw.line(canvas, GRID, (j * t, 0), (j * t, surf_h), 1)

        if self.render_mode == "human":
            # Keep window responsive
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    self._user_quit = True

            self._window.blit(canvas, (0, 0))
            pygame.display.update()
            self._clock.tick(self.metadata.get("render_fps", 10))
            return  # human returns None

        else:  # "rgb_array"
            arr = pygame.surfarray.array3d(canvas)        # (W, H, 3)
            arr = np.transpose(arr, (1, 0, 2)).copy()     # -> (H, W, 3)
            return arr

    def close(self):
        try:
            import pygame
            if getattr(self, "_window", None) is not None:
                pygame.display.quit()
            pygame.quit()
        except Exception:
            pass
        self._window = None
        self._clock = None
    
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
            self.pacman_dir = Actions(action)
        else: # Invalid action, continue in same direction
            movement = self.action_to_direction[self.pacman_dir.value]
            new_loc = self.pacman_loc + movement
            if self._valid_loc(new_loc): #Check that can keep moving in this direction
                self.pacman_loc = new_loc
            # If can't keep moving in old direction, Pacman location doesn't change

    # This function should update the position of the ghosts 
    # Ghost in Pacman only change direction when at an intersection
    # Seem to never flip direction
    # Could make ghost nondeterministic with epsilon chance to pick random direction?
    def _update_ghosts(self):
        # Make 2 steps separately so ghost can go around corners
        for _ in range(2):
            # Figure out ghost next direction
            possible_dirs = self._get_ghost_moves()
            if len(possible_dirs) > 1: # At an intersection
                self.ghost_dir = self._optimal_ghost_dir(possible_dirs)
                #recalculate direction based on pacman loc (move closer to pacman)
            elif len(possible_dirs) == 0: # means at dead-end (can always backtrack)
                self.ghost_dir = self.ghost_dir.opposite()
            else: # Move in only possible non-backwards direction
                self.ghost_dir = possible_dirs[0]
            # Update ghost location (we already know it is a valid location)
            self.ghost_loc = self.ghost_loc + self.action_to_direction[self.ghost_dir.value]
            # At each move, check if hit Pacman
            if self._caught_pacman():
                self.terminated = True
                self.grid[tuple(self.pacman_loc)] = Cells.dead.value
                return # If caught pacman, don't make second move
    
    # Function to determine if ghost caught pacman
    def _caught_pacman(self):
        return np.array_equal(self.ghost_loc, self.pacman_loc)
    
    def _get_obs(self):
        return {"grid": self.grid.copy(), "direction": self.pacman_dir.value}

    # A function to update tracked values about state
    def _update_stats(self):
         # For faster updating after step
        self.pacman_loc = np.argwhere(self.grid == Cells.pacman.value)[0].copy() # Only one Pacman, so just take first
        self.pacman_dir = Actions.down # Doesn't matter, AI will choose direction before first step
        self.ghost_loc = np.argwhere(self.grid == Cells.ghost.value)[0].copy() # Only one ghost, so just take first
        self.ghost_dir = Actions.down # Arbitrary?
        self.ghost_had_pip = True # Need to know if the ghost was in a cell with a pip so we can replace it when it moves out
        self.remaining_pips = np.sum(self.grid == Cells.pip.value)
        self.pip_threshold = (self.remaining_pips//5) + 1 # Fifth of starting pips, rounded up (can change later)
        self.terminated = False # To abort step early

    # A function to add more pips once they drop below a threshold
    def _regenerate_pips(self):
        # Get all empty spaces, pick some at random and add pips
        # Add half of pip_threshold? i.e. repopulate 10% of space at random
        empty = np.argwhere(self.grid == Cells.empty.value)
        n_add = self.pip_threshold//2 + 1
        chosen = empty[self.np_random.choice(len(empty), n_add, replace=False)]
        self.grid[chosen[:,0], chosen[:,1]] = Cells.pip.value

    # Returns the possible directions the ghost can move
    def _get_ghost_moves(self):
        moves = []
        for action in Actions:
            # Pac-man ghosts don't do 180s
            # We exclude this action since we won't consider it unless it is the only option
            if action == self.ghost_dir.opposite():
                continue
            next_in_dir = self.ghost_loc + self.action_to_direction[action.value]
            if self._valid_loc(next_in_dir):
                moves.append(action)
        return moves
    
    def _optimal_ghost_dir(self, dirs):
        xdist = self.ghost_loc[1] - self.pacman_loc[1]
        ydist = self.ghost_loc[0] - self.pacman_loc[0]
        # Prefer to minimize larger delta
        preference_order = []
        if abs(xdist) > abs(ydist):
            if xdist > 0: # Pacman to left, so move left  if possible
                preference_order.append(Actions.left)
            else:
                preference_order.append(Actions.right)

            if ydist > 0: #Pacman up, so try to move up
                preference_order.append(Actions.up)
                preference_order.append(Actions.down)
            else:
                preference_order.append(Actions.down)
                preference_order.append(Actions.up)
        else:
            if ydist > 0: # Pacman up, so move up  if possible
                preference_order.append(Actions.up)
            else:
                preference_order.append(Actions.down)
            
            if xdist > 0: #Pacman left, so try to move up
                preference_order.append(Actions.left)
                preference_order.append(Actions.right)
            else:
                preference_order.append(Actions.right)
                preference_order.append(Actions.left)
        # Get best possible direction
        # Only call function when multiple directions are possible, so we only need 3 in preference_order
        for dir in preference_order:
            if dir in dirs:
                return dir
            
    # helper function to make one step towards pellet strictly better than standing still
    def nearest_pip_dist(self):
        ps = np.argwhere(self.grid == Cells.pip.value)
        if len(ps) == 0: return 0
        pr, pc = self.pacman_loc
        return int(np.min(np.abs(ps[:,0]-pr) + np.abs(ps[:,1]-pc)))

    # Function to refresh grid after episode finished
    # This is a placeholder
    def _generate_grid(self):
        self.grid = np.array(
                #    [[1, 2, 2, 2, 2, 2],
                #     [2, 3, 3, 3, 3, 2],
                #     [2, 2, 2, 2, 3, 2],
                #     [2, 2, 2, 2, 2, 2],
                #     [3, 3, 3, 3, 2, 2],
                #     [2, 2, 2, 2, 4, 2]],
                    [[2,2,3,3,3,2,2,2,3,3,3,2,2,2,4],
                    [2,2,3,2,2,2,3,2,2,2,3,2,3,2,2],
                    [3,3,3,2,3,2,3,3,3,2,3,2,3,2,2],
                    [2,2,2,2,3,2,2,2,3,2,2,2,3,2,2],
                    [2,3,3,2,3,3,3,2,3,3,3,2,3,3,2],
                    [2,2,3,2,2,2,2,2,2,2,2,2,3,2,2],
                    [2,3,3,3,3,2,3,3,3,3,3,2,3,2,2],
                    [2,2,2,2,2,2,2,2,2,2,3,2,2,2,2],
                    [2,3,3,3,3,3,3,3,3,2,3,3,3,3,2],
                    [2,3,2,2,2,2,2,2,3,2,2,2,2,3,2],
                    [2,3,2,3,3,3,3,2,3,3,3,3,2,3,2],
                    [2,2,2,2,2,2,2,2,2,2,2,3,2,2,2],
                    [2,3,3,2,3,2,3,3,3,3,2,3,3,3,2],
                    [2,2,2,2,3,2,2,2,2,2,2,2,2,2,2],
                    [1,2,2,2,2,2,2,2,2,3,2,2,2,2,2]],
                    dtype=np.int8) 
