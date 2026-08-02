from stable_baselines3 import PPO
from traffic_env import TrafficEnv
import time

def main():
    # Evaluate with human rendering
    env = TrafficEnv(render_mode="human")

    print("Loading best trained model...")
    try:
        model = PPO.load("best_traffic_model/best_model")
    except Exception:
        print("Best model not found, loading final model...")
        model = PPO.load("traffic_model_final")

    state, _ = env.reset()

    # Slow-demo settings: adjust STEP_DELAY (seconds) to control speed
    STEP_DELAY = 0.5   # 0.5 s per step  ← change this value to go faster/slower

    for step in range(300):
        action, _ = model.predict(state, deterministic=True)
        state, reward, terminated, truncated, _ = env.step(action)

        print(f"[Step {step:>3}/300]  Reward: {reward:+.1f}")

        time.sleep(STEP_DELAY)   # pause so the GUI is visible

        if terminated or truncated:
            break

    env.close()
    print("Demo finished")

if __name__ == "__main__":
    main()
