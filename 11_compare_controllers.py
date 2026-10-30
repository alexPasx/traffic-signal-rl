import os
from pathlib import Path

import sumo

os.environ["SUMO_HOME"] = str(sumo.SUMO_HOME)

import numpy as np
import pandas as pd
import sumo_rl
import traci
from stable_baselines3 import PPO

from adaptive_controller import AdaptiveController
from traffic_env import TrafficEnv
from fair_traffic_env import FairTrafficEnv


ROOT = Path(__file__).resolve().parent
RESULTS = ROOT / "results" / "comparison"
DURATION = 600

# Διαφορετικά seeds από τη δοκιμή με seed 42.
SEEDS = range(1000, 1020)

CONTROLLERS = ["fixed", "adaptive", "rl_wait", "rl_fair"]

SCENARIO = (
    Path(sumo_rl.__file__).resolve().parent
    / "nets"
    / "2way-single-intersection"
)


def start_sumo(seed):
    traci.start([
        str(Path(sumo.SUMO_HOME) / "bin" / "sumo.exe"),
        "--net-file",
        str(SCENARIO / "single-intersection.net.xml"),
        "--route-files",
        str(SCENARIO / "single-intersection-vhvh.rou.xml"),
        "--seed", str(seed),
        "--step-length", "1",
        "--no-step-log", "true",
    ])


def evaluate(name, seed, model=None):
    env = None
    vehicles_data = {}
    arrived = set()

    stopped_sum = 0
    pending_sum = 0
    teleports = 0

    try:
        if name == "rl_wait":
            env = TrafficEnv(duration=DURATION)

        elif name == "rl_fair":
            env = FairTrafficEnv(duration=DURATION)

        if env is not None:
            observation, _ = env.reset(
                seed=seed,
                options={"sumo_seed": seed},
            )
        else:
            start_sumo(seed)

        controller = (
            AdaptiveController("t")
            if name == "adaptive"
            else None
        )

        for _ in range(DURATION):
            if env is not None:
                action, _ = model.predict(
                    observation,
                    deterministic=True,
                )

                observation, _, terminated, truncated, _ = env.step(
                    int(action.item())
                )

            else:
                if controller is not None:
                    controller.update()

                traci.simulationStep()

            # Καταγράφουμε κάθε όχημα από την είσοδό του.
            for vehicle_id in traci.simulation.getDepartedIDList():
                route = traci.vehicle.getRoute(vehicle_id)
                origin_edge = route[0]

                vehicles_data[vehicle_id] = {
                    "vehicle_id": vehicle_id,
                    "origin_edge": origin_edge,
                    "direction": origin_edge.split("_")[0],
                    "observed_wait_s": 0.0,
                }

            stopped = 0

            for vehicle_id in traci.vehicle.getIDList():
                if vehicle_id not in vehicles_data:
                    route = traci.vehicle.getRoute(vehicle_id)
                    origin_edge = route[0]

                    vehicles_data[vehicle_id] = {
                        "vehicle_id": vehicle_id,
                        "origin_edge": origin_edge,
                        "direction": origin_edge.split("_")[0],
                        "observed_wait_s": 0.0,
                    }

                if traci.vehicle.getSpeed(vehicle_id) < 0.1:
                    vehicles_data[vehicle_id]["observed_wait_s"] += 1
                    stopped += 1

            arrived.update(
                traci.simulation.getArrivedIDList()
            )

            stopped_sum += stopped

            pending_sum += len(
                traci.simulation.getPendingVehicles()
            )

            teleports += (
                traci.simulation.getStartingTeleportNumber()
            )

        remaining = len(traci.vehicle.getIDList())
        pending_final = len(
            traci.simulation.getPendingVehicles()
        )

    finally:
        if env is not None:
            env.close()
        elif traci.isLoaded():
            traci.close()

    vehicle_rows = []

    for vehicle_id, data in vehicles_data.items():
        vehicle_rows.append({
            "controller": name,
            "seed": seed,
            **data,
            "arrived": vehicle_id in arrived,
        })

    waits = np.array([
        row["observed_wait_s"] for row in vehicle_rows
    ])

    summary = {
        "controller": name,
        "seed": seed,
        "entered": len(vehicle_rows),
        "arrived": len(arrived),
        "remaining": remaining,
        "pending_final": pending_final,
        "teleport_events": teleports,
        "mean_wait_s": float(waits.mean()) if len(waits) else 0,
        "p95_wait_s": float(np.percentile(waits, 95)) if len(waits) else 0,
        "max_wait_s": float(waits.max()) if len(waits) else 0,
        "percent_over_30s": (
            float(100 * np.mean(waits > 30)) if len(waits) else 0
        ),
        "mean_stopped": stopped_sum / DURATION,
        "pending_vehicle_seconds": pending_sum,
        "total_wait_vehicle_seconds": stopped_sum + pending_sum,
    }

    return summary, vehicle_rows


def main():
    RESULTS.mkdir(parents=True, exist_ok=True)

    models = {
        "rl_wait": PPO.load(
            str(ROOT / "models" / "ppo_wait.zip"),
            device="cpu",
        ),
        "rl_fair": PPO.load(
            str(ROOT / "models" / "ppo_fair.zip"),
            device="cpu",
        ),
    }

    summaries = []
    all_vehicles = []

    total_runs = len(CONTROLLERS) * len(SEEDS)
    completed = 0

    for name in CONTROLLERS:
        for seed in SEEDS:
            summary, vehicle_rows = evaluate(
                name,
                seed,
                model=models.get(name),
            )

            summaries.append(summary)
            all_vehicles.extend(vehicle_rows)

            # Αποθήκευση μετά από κάθε ολοκληρωμένη δοκιμή.
            pd.DataFrame(summaries).to_csv(
                RESULTS / "episodes.csv",
                index=False,
            )

            pd.DataFrame(all_vehicles).to_csv(
                RESULTS / "vehicles.csv",
                index=False,
            )

            completed += 1

            print(
                f"[{completed}/{total_runs}] "
                f"{name} | seed={seed} | "
                f"αναμονή={summary['mean_wait_s']:.2f} s | "
                f"ολοκλήρωσαν={summary['arrived']}"
            )

    episodes = pd.DataFrame(summaries)
    vehicles = pd.DataFrame(all_vehicles)

    metrics = [
        "mean_wait_s",
        "p95_wait_s",
        "max_wait_s",
        "percent_over_30s",
        "arrived",
        "remaining",
        "pending_final",
        "total_wait_vehicle_seconds",
    ]

    aggregate = episodes.groupby("controller")[metrics].agg(
        ["mean", "std"]
    )

    aggregate.to_csv(RESULTS / "summary.csv")

    # Πρώτα υπολογίζουμε ανά κατεύθυνση και seed.
    directional = (
        vehicles.groupby(["controller", "seed", "direction"])
        ["observed_wait_s"]
        .agg(
            vehicle_count="count",
            mean_wait_s="mean",
            p95_wait_s=lambda x: x.quantile(0.95),
            max_wait_s="max",
        )
        .reset_index()
    )

    directional.to_csv(
        RESULTS / "direction_by_seed.csv",
        index=False,
    )

    direction_summary = (
        directional.groupby(["controller", "direction"])
        [["mean_wait_s", "p95_wait_s", "max_wait_s"]]
        .mean()
    )

    direction_summary.to_csv(
        RESULTS / "direction_summary.csv"
    )

    print("\n--- ΜΕΣΕΣ ΤΙΜΕΣ ΣΤΑ 20 SEEDS ---")
    print(
        episodes.groupby("controller")[metrics]
        .mean()
        .round(2)
        .to_string()
    )

    print("\n--- ΑΝΑ ΚΑΤΕΥΘΥΝΣΗ: ΜΕΣΟΣ ΟΡΟΣ ΤΩΝ SEEDS ---")
    print(direction_summary.round(2).to_string())

    print(f"\nΑποθηκεύτηκαν στο: {RESULTS}")


if __name__ == "__main__":
    main()