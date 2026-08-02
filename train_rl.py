from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import EvalCallback
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.vec_env import SubprocVecEnv
from traffic_env import TrafficEnv

def main():
    # Wrap environment in a Subprocess Vectorized Env for 8 simultaneous cores (stable)
    env = make_vec_env(TrafficEnv, n_envs=8, env_kwargs={"render_mode": None}, vec_env_cls=SubprocVecEnv)
    
    # We still need a single environment just for evaluation callbacks
    eval_env = TrafficEnv(render_mode=None)

    eval_callback = EvalCallback(
        eval_env,
        best_model_save_path='./best_traffic_model/',
        log_path='./logs/',
        eval_freq=1000,
        deterministic=True,
        render=False
    )

    model = PPO(
        "MlpPolicy",
        env,
        verbose=1,
        learning_rate=0.0003,
        n_steps=512,
        batch_size=64,
    )

    print("Training started...")
    model.learn(total_timesteps=50000, callback=eval_callback)
    
    # Save the final model too
    model.save("traffic_model_final")
    print("MODEL SAVED ✅")

if __name__ == "__main__":
    main()
