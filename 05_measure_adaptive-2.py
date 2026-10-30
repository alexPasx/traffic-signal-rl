import os
from pathlib import Path

import sumo

os.environ["SUMO_HOME"] = str(sumo.SUMO_HOME)

import numpy as np
import pandas as pd
import sumo_rl
import traci

from adaptive_controller import AdaptiveController


ROOT = Path(__file__).resolve().parent

SCENARIO = (
    Path(sumo_rl.__file__).resolve().parent
    / "nets"
    / "2way-single-intersection"
)

DURATION = 600
SEED = 42
SIGNAL_ID = "t"

GREEN_PHASES = {0, 2, 4, 6}
YELLOW_PHASES = {1, 3, 5, 7}


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
        str(results_dir / "adaptive_trips.xml"),
        "--tripinfo-output.write-unfinished", "true",
    ]

    waiting = {}
    arrived_ids = set()
    teleports = 0

    history = []
    phase_history = []

    traci.start(command)

    try:
        controller = AdaptiveController(signal_id=SIGNAL_ID)

        last_phase = traci.trafficlight.getPhase(SIGNAL_ID)
        phase_started = traci.simulation.getTime()

        def record_phase_change():
            nonlocal last_phase, phase_started

            current_phase = traci.trafficlight.getPhase(SIGNAL_ID)
            now = traci.simulation.getTime()

            if current_phase != last_phase:
                phase_history.append({
                    "phase": last_phase,
                    "start_s": phase_started,
                    "end_s": now,
                    "duration_s": now - phase_started,
                    "complete": True,
                })

                last_phase = current_phase
                phase_started = now

        print("Εκτέλεση προσαρμοζόμενου ελεγκτή.")
        print(f"Διάρκεια: {DURATION} s | Seed: {SEED}")

        while traci.simulation.getTime() < DURATION:
            # Η δική μας απόφαση μπορεί να αλλάξει αμέσως τη φάση.
            controller.update()
            record_phase_change()

            # Προχωράμε ένα δευτερόλεπτο στην προσομοίωση.
            traci.simulationStep()

            # Ελέγχουμε και τις αυτόματες μεταβάσεις του SUMO.
            record_phase_change()

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

            now = traci.simulation.getTime()
            pending_now = len(
                traci.simulation.getPendingVehicles()
            )

            history.append({
                "time_s": now,
                "phase": traci.trafficlight.getPhase(SIGNAL_ID),
                "vehicles_in_network": len(vehicles),
                "stopped_vehicles": stopped,
                "pending_insertion": pending_now,
                "arrived_total": len(arrived_ids),
            })

            if int(now) % 60 == 0:
                print(
                    f"Χρόνος: {now:.0f} s | "
                    f"Σταματημένα: {stopped} | "
                    f"Ολοκληρωμένες διαδρομές: {len(arrived_ids)}"
                )

        end_time = traci.simulation.getTime()
        remaining = len(traci.vehicle.getIDList())
        pending = len(traci.simulation.getPendingVehicles())

        # Η τελευταία φάση δεν έχει παρατηρηθεί μέχρι το τέλος της.
        # Δεν τη χρησιμοποιούμε στον έλεγχο χρονικών ορίων.
        phase_history.append({
            "phase": last_phase,
            "start_s": phase_started,
            "end_s": end_time,
            "duration_s": end_time - phase_started,
            "complete": False,
        })

    finally:
        traci.close()

    waits = np.array(list(waiting.values()), dtype=float)

    phase_df = pd.DataFrame(phase_history)
    complete_phases = phase_df[phase_df["complete"]].copy()

    green = complete_phases[
        complete_phases["phase"].isin(GREEN_PHASES)
    ]

    yellow = complete_phases[
        complete_phases["phase"].isin(YELLOW_PHASES)
    ]

    bad_green = green[
        (green["duration_s"] < controller.min_green - 1e-6)
        | (green["duration_s"] > controller.max_green + 1e-6)
    ]

    bad_yellow = yellow[
        ~np.isclose(yellow["duration_s"], 2.0)
    ]

    # Ελέγχουμε ότι διατηρήθηκε η κυκλική σειρά 0→1→...→7→0.
    observed_phases = phase_df["phase"].to_list()

    sequence_errors = sum(
        following != (previous + 1) % 8
        for previous, following in zip(
            observed_phases,
            observed_phases[1:],
        )
    )

    summary = {
        "controller": "adaptive",
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
        "green_duration_violations": len(bad_green),
        "yellow_duration_violations": len(bad_yellow),
        "phase_sequence_errors": sequence_errors,
    }

    pd.DataFrame(history).to_csv(
        results_dir / "adaptive_history.csv",
        index=False,
    )

    phase_df.to_csv(
        results_dir / "adaptive_phase_history.csv",
        index=False,
    )

    pd.DataFrame([summary]).to_csv(
        results_dir / "adaptive_summary.csv",
        index=False,
    )

    print("\n--- ΠΡΟΣΑΡΜΟΖΟΜΕΝΟ ΠΡΟΓΡΑΜΜΑ ---")

    for key, value in summary.items():
        if isinstance(value, float):
            print(f"{key}: {value:.2f}")
        else:
            print(f"{key}: {value}")

    print("\n--- ΔΙΑΡΚΕΙΕΣ ΟΛΟΚΛΗΡΩΜΕΝΩΝ ΦΑΣΕΩΝ ---")

    print(
        complete_phases.groupby("phase")["duration_s"]
        .agg(["count", "min", "mean", "max"])
        .round(2)
        .to_string()
    )

    print(f"\nΠράσινες εκτός ορίων: {len(bad_green)}")
    print(f"Κίτρινες εκτός ορίων: {len(bad_yellow)}")
    print(f"Λάθη σειράς φάσεων: {sequence_errors}")

    if not bad_green.empty:
        print("\nΠράσινες με παραβίαση:")
        print(bad_green.to_string(index=False))

    if not bad_yellow.empty:
        print("\nΚίτρινες με παραβίαση:")
        print(bad_yellow.to_string(index=False))

    print("\nΑποτελέσματα στον φάκελο results.")


if __name__ == "__main__":
    main()