from stable_baselines3.common.env_checker import check_env

from traffic_env import TrafficEnv


def main():
    env = TrafficEnv(duration=60)

    try:
        check_env(env, warn=True)
        print("Έλεγχος Gymnasium / Stable-Baselines3: OK")

        observation, info = env.reset(
            seed=42,
            options={"sumo_seed": 42},
        )

        print("Σχήμα εισόδου:", observation.shape)
        print("Ενέργειες:", env.action_space.n)

        total_reward = 0.0

        while True:
            action = env.action_space.sample()

            observation, reward, terminated, truncated, info = env.step(
                action
            )

            total_reward += reward

            if terminated or truncated:
                break

        print("Τελευταίες πληροφορίες:", info)
        print(f"Συνολική ανταμοιβή: {total_reward:.2f}")

    finally:
        env.close()


if __name__ == "__main__":
    main()