import pickle, numpy as np
from pacman_env.envs.pacman_world import Cells

def encode_state(obs):
    # find pacman and ghost
    grid = obs["grid"]
    pac = np.argwhere(grid == Cells.pacman.value)
    ghost = np.argwhere(grid == Cells.ghost.value)
    if len(pac) == 0 or len(ghost) == 0:
        return("missing", 0)
    
    # pacman (r, c), ghost (r,c)
    pr, pc = int(pac[0][0]), int(pac[0][1])
    gr, gc = int(ghost[0][0]), int(ghost[0][1])
 
    # pellet density (how full the baord is)
    pip_dir_bucket = nearest_pellet_dir(grid, (pr, pc))
    ghost_bucket = ghost_dist((pr, pc), (gr, gc))
    
    # hashable tuple
    return(pr, pc, pip_dir_bucket, ghost_bucket)

def load_q(path):
    with open(path, "rb") as f:
        raw = pickle.load(f)
    return {k: np.array(v, dtype=float) for k, v in raw.items()}

# nearest pellet direction bucket to give better directional signal
def nearest_pellet_dir(grid, pac_rc):
    pr, pc = pac_rc
    ps = np.argwhere(grid == Cells.pip.value)
    if len(ps) == 0:
        return 4
    
    # nearest pellet
    dist = np.abs(ps[:,0]-pr) + np.abs(ps[:,1]-pc)
    r, c = ps[int(np.argmin(dist))]
    dr, dc = r - pr, c - pc
    
    # 0=N, 1=E, 2=S, 3=W, 4=none
    if abs(dr) >= abs(dc):
        return 0 if dr < 0 else 2
    else:
        return 1 if dc > 0 else 3
    
# detect ghost distance
def ghost_dist(pac_rc, ghost_rc):
    d = abs(pac_rc[0]-ghost_rc[0]) + abs(pac_rc[1]-ghost_rc[1])
    return 0 if d <= 1 else 1 if d <= 2 else 2 if d <= 4 else 3
