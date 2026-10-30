from pathlib import Path

import numpy as np
import pandas as pd
import traci
from stable_baselines3 import PPO

from traffic_env import TrafficEnv


ROOT = Path(__file__).resolve().parent
DURATION = 600
SEED = 42


def main():
    model_path = ROOT / "models" / "ppo_wait.zip"

    if not model_path.is_file():
        raise FileNotFoundError(
            f"Δεν βρέθηκε το μοντέλο: {model_path}"
        )

    results_dir = ROOT / "results"
    results_dir.mkdir(exist_ok=True)

    model = PPO.load(str(model_path), device="cpu")
    env = TrafficEnv(duration=DURATION)

    waiting = {}
    arrived_ids = set()
    history = []
    teleports = 0
    total_reward = 0.0

    try:
        observation, info = env.reset(
            seed=42,
            options={"sumo_seed": SEED},
        )

        while True:
            action, _ = model.predict(
                observation,
                deterministic=True,
            )

            observation, reward, terminated, truncated, info = env.step(
                int(action.item())
            )

            total_reward += reward

            for vehicle_id in traci.simulation.getDepartedIDList():
                waiting.setdefault(vehicle_id, 0.0)

            vehicles = traci.vehicle.getIDList()
            stopped = 0

            for vehicle_id in vehicles:
                waiting.setdefault(vehicle_id, 0.0)

                if traci.vehicle.getSpeed(vehicle_id) < 0.1:
                    waiting[vehicle_id] += 1.0
                    stopped += 1

            arrived_ids.update(
                traci.simulation.getArrivedIDList()
            )

            teleports += (
                traci.simulation.getStartingTeleportNumber()
            )

            pending = len(
                traci.simulation.getPendingVehicles()
            )

            now = traci.simulation.getTime()

            history.append({
                "time_s": now,
                "phase": traci.trafficlight.getPhase("t"),
                "vehicles_in_network": len(vehicles),
                "stopped_vehicles": stopped,
                "pending_insertion": pending,
                "arrived_total": len(arrived_ids),
            })

            if int(now) % 60 == 0:
                print(
                    f"Χρόνος: {now:.0f} s | "
                    f"Σταματημένα: {stopped} | "
                    f"Ολοκληρωμένες διαδρομές: {len(arrived_ids)}"
                )

            if terminated or truncated:
                break

        remaining = len(traci.vehicle.getIDList())

    finally:
        env.close()

    waits = np.array(list(waiting.values()), dtype=float)

    summary = {
        "controller": "rl_wait",
        "seed": SEED,
        "duration_s": DURATION,
        "entered_vehicles": len(waiting),
        "arrived_vehicles": len(arrived_ids),
        "remaining_vehicles": remaining,
        "pending_insertion": pending,
        "teleport_events": teleports,
        "mean_observed_wait_s": (
            float(waits.mean()) if len(waits) else 0.0
        ),
        "p95_observed_wait_s": (
            float(np.percentile(waits, 95)) if len(waits) else 0.0
        ),
        "max_observed_wait_s": (
            float(waits.max()) if len(waits) else 0.0
        ),
        "mean_stopped_vehicles": float(np.mean([
            row["stopped_vehicles"] for row in history
        ])),
        "total_reward": total_reward,
    }

    pd.DataFrame(history).to_csv(
        results_dir / "rl_wait_history.csv",
        index=False,
    )

    pd.DataFrame([summary]).to_csv(
        results_dir / "rl_wait_summary.csv",
        index=False,
    )

    print("\n--- RL: ΣΥΝΟΛΙΚΗ ΑΝΑΜΟΝΗ ---")

    for key, value in summary.items():
        if isinstance(value, float):
            print(f"{key}: {value:.2f}")
        else:
            print(f"{key}: {value}")

    print("\nΑποτελέσματα στον φάκελο results.")


if __name__ == "__main__":
    main()