import numpy as np
from traffic_env import TrafficEnv

env = TrafficEnv(render_mode=None) # faster purely to test logic

state, _ = env.reset()
print("Initial state shape:", state.shape)

for i in range(50):
    # random phases (0-3)
    action = np.random.randint(0, 2, size=env.num_tls)
    state, reward, terminated, truncated, _ = env.step(action)
    print(f"Step {i} | Reward: {reward}")

    if terminated or truncated:
        break

env.close()
print("Testing done!")
