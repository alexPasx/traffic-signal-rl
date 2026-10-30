from pathlib import Path

from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import CheckpointCallback
from stable_baselines3.common.monitor import Monitor

from traffic_env import TrafficEnv


ROOT = Path(__file__).resolve().parent
TOTAL_STEPS = 200_000


def main():
    models_dir = ROOT / "models"
    logs_dir = ROOT / "logs"

    models_dir.mkdir(exist_ok=True)
    logs_dir.mkdir(exist_ok=True)

    env = Monitor(
        TrafficEnv(duration=600),
        filename=str(logs_dir / "ppo_wait"),
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
        save_path=str(models_dir / "wait_checkpoints"),
        name_prefix="ppo_wait",
    )

    print(f"Έναρξη εκπαίδευσης: {TOTAL_STEPS:,} βήματα.")
    print("Κάθε επεισόδιο διαρκεί 600 δευτερόλεπτα προσομοίωσης.")

    try:
        model.learn(
            total_timesteps=TOTAL_STEPS,
            callback=checkpoint,
        )

        model.save(str(models_dir / "ppo_wait"))
        print("\nΑποθηκεύτηκε το models/ppo_wait.zip")

    except KeyboardInterrupt:
        model.save(str(models_dir / "ppo_wait_interrupted"))
        print("\nΑποθηκεύτηκε ενδιάμεσο μοντέλο.")

    finally:
        env.close()


if __name__ == "__main__":
    main()