import gymnasium
import pacman_env
import pygame
from qpolicy import encode_state, load_q
import pickle
import numpy as np

Q = load_q("models/qlearning.pkl") # import Q-table

env = gymnasium.make('pacman_env/PacManWorld', render_mode="human")

observation, info = env.reset()
# env.render()

total_reward = 0
done = False

while not done:
    s = encode_state(observation)
    
    # choose best action
    if s in Q:
        action = int(np.argmax(Q[s]))
    else: 
        action = env.action_space.sample()
    # obs, r, term, trunc, _ = env.step(action)
    observation, reward, terminated, truncated, info = env.step(action)

    total_reward += reward
    done = terminated or truncated
    print(observation["grid"])

    print(f"total reward: {total_reward}")

    env.render()  # updates the same window each step
    done = terminated or truncated or getattr(env, "_user_quit", False)

# this lets the window stay open at the end (just to look at it)
hold = True
clock = pygame.time.Clock()
while hold:
    # keep window responsive
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            hold = False

    env.render()
    clock.tick(30)  # keep ~30 fps
env.close()

