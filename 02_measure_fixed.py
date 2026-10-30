import os
from pathlib import Path

import sumo

os.environ["SUMO_HOME"] = str(sumo.SUMO_HOME)

import numpy as np
import pandas as pd
import sumo_rl
import traci


ROOT = Path(__file__).resolve().parent

SCENARIO = (
    Path(sumo_rl.__file__).resolve().parent
    / "nets"
    / "2way-single-intersection"
)

DURATION = 600
SEED = 42


def main():
    results_dir = ROOT / "results"
    results_dir.mkdir(exist_ok=True)

    command = [
        str(Path(sumo.SUMO_HOME) / "bin" / "sumo.exe"),
        "--net-file",
        str(SCENARIO / "single-intersection.net.xml"),
        "--route-files",
        str(SCENARIO / "single-intersection-vhvh.rou.xml"),
        "--seed", str(SEED),
        "--step-length", "1",
        "--no-step-log", "true",
        "--tripinfo-output",
        str(results_dir / "fixed_trips.xml"),
        "--tripinfo-output.write-unfinished", "true",
    ]

    # Αναμονή που παρατηρήσαμε για κάθε όχημα.
    waiting = {}
    arrived_ids = set()
    teleports = 0
    history = []
    phases = []

    traci.start(command)

    try:
        # Διαβάζουμε το ενεργό πρόγραμμα κάθε σηματοδότη.
        for signal_id in traci.trafficlight.getIDList():
            program_id = traci.trafficlight.getProgram(signal_id)

            programs = traci.trafficlight.getAllProgramLogics(signal_id)
            active = next(
                program
                for program in programs
                if program.programID == program_id
            )

            print(f"\nΣηματοδότης: {signal_id}")
            print(f"Πρόγραμμα: {program_id}")

            for index, phase in enumerate(active.phases):
                print(
                    f"Φάση {index}: "
                    f"{phase.duration:.0f} s | {phase.state}"
                )

                phases.append({
                    "signal": signal_id,
                    "program": program_id,
                    "phase": index,
                    "duration_s": phase.duration,
                    "state": phase.state,
                })

        while traci.simulation.getTime() < DURATION:
            traci.simulationStep()

            # Καταγράφουμε όσα οχήματα εισήλθαν στο δίκτυο.
            for vehicle_id in traci.simulation.getDepartedIDList():
                waiting.setdefault(vehicle_id, 0.0)

            vehicles = traci.vehicle.getIDList()
            stopped = 0

            for vehicle_id in vehicles:
                waiting.setdefault(vehicle_id, 0.0)

                if traci.vehicle.getSpeed(vehicle_id) < 0.1:
                    waiting[vehicle_id] += 1.0
                    stopped += 1

            arrived_ids.update(traci.simulation.getArrivedIDList())
            teleports += traci.simulation.getStartingTeleportNumber()

            history.append({
                "time_s": traci.simulation.getTime(),
                "vehicles_in_network": len(vehicles),
                "stopped_vehicles": stopped,
                "pending_insertion": len(
                    traci.simulation.getPendingVehicles()
                ),
                "arrived_total": len(arrived_ids),
            })

        remaining = len(traci.vehicle.getIDList())
        pending = len(traci.simulation.getPendingVehicles())

    finally:
        traci.close()

    waits = np.array(list(waiting.values()), dtype=float)

    summary = {
        "controller": "fixed",
        "seed": SEED,
        "duration_s": DURATION,
        "entered_vehicles": len(waiting),
        "arrived_vehicles": len(arrived_ids),
        "remaining_vehicles": remaining,
        "pending_insertion": pending,
        "teleport_events": teleports,
        "mean_observed_wait_s": float(waits.mean()) if len(waits) else 0,
        "p95_observed_wait_s": float(np.percentile(waits, 95)) if len(waits) else 0,
        "max_observed_wait_s": float(waits.max()) if len(waits) else 0,
        "mean_stopped_vehicles": float(np.mean([
            row["stopped_vehicles"] for row in history
        ])),
    }

    pd.DataFrame(phases).to_csv(
        results_dir / "fixed_phases.csv", index=False
    )

    pd.DataFrame(history).to_csv(
        results_dir / "fixed_history.csv", index=False
    )

    pd.DataFrame([summary]).to_csv(
        results_dir / "fixed_summary.csv", index=False
    )

    print("\n--- ΣΤΑΘΕΡΟ ΠΡΟΓΡΑΜΜΑ ---")

    for key, value in summary.items():
        if isinstance(value, float):
            print(f"{key}: {value:.2f}")
        else:
            print(f"{key}: {value}")

    print("\nΑποτελέσματα στον φάκελο results.")


if __name__ == "__main__":
    main()