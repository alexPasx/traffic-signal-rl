from pathlib import Path

from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import CheckpointCallback
from stable_baselines3.common.env_checker import check_env
from stable_baselines3.common.monitor import Monitor

from fair_traffic_env import FairTrafficEnv


ROOT = Path(__file__).resolve().parent
TOTAL_STEPS = 200_000


def main():
    models_dir = ROOT / "models"
    logs_dir = ROOT / "logs"

    models_dir.mkdir(exist_ok=True)
    logs_dir.mkdir(exist_ok=True)

    # Έλεγχος του νέου περιβάλλοντος.
    test_env = FairTrafficEnv(duration=60)

    try:
        check_env(test_env, warn=True)
        print("Έλεγχος νέου περιβάλλοντος: OK")
    finally:
        test_env.close()

    env = Monitor(
        FairTrafficEnv(
            duration=600,
            wait_threshold=30.0,
            fairness_weight=2.0,
        ),
        filename=str(logs_dir / "ppo_fair"),
    )

    model = PPO(
        policy="MlpPolicy",
        env=env,
        learning_rate=0.0003,
        n_steps=2048,
        batch_size=64,
        gamma=0.99,
        policy_kwargs=dict(
            net_arch=dict(
                pi=[64, 64],
                vf=[64, 64],
            )
        ),
        seed=42,
        device="cpu",
        verbose=1,
    )

    checkpoint = CheckpointCallback(
        save_freq=50_000,
        save_path=str(models_dir / "fair_checkpoints"),
        name_prefix="ppo_fair",
    )

    try:
        model.learn(
            total_timesteps=TOTAL_STEPS,
            callback=checkpoint,
        )

        model.save(str(models_dir / "ppo_fair"))
        print("\nΑποθηκεύτηκε το models/ppo_fair.zip")

    except KeyboardInterrupt:
        model.save(str(models_dir / "ppo_fair_interrupted"))
        print("\nΑποθηκεύτηκε ενδιάμεσο μοντέλο.")

    finally:
        env.close()


if __name__ == "__main__":
    main()