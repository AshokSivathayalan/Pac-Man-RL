import gymnasium
import pacman_env

env = gymnasium.make('pacman_env/PacManWorld')

observation, info = env.reset()

total_reward = 0
done = False

while not done:
    action = env.action_space.sample()
    observation, reward, terminated, truncated, info = env.step(action)

    total_reward += reward
    done = terminated or truncated
    print(observation["grid"])

print(f"total reward: {total_reward}")
env.close()

