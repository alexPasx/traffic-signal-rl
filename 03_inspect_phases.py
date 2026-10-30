import os
from pathlib import Path

import sumo

os.environ["SUMO_HOME"] = str(sumo.SUMO_HOME)

import pandas as pd
import sumo_rl
import traci


ROOT = Path(__file__).resolve().parent

SCENARIO = (
    Path(sumo_rl.__file__).resolve().parent
    / "nets"
    / "2way-single-intersection"
)


def main():
    command = [
        str(Path(sumo.SUMO_HOME) / "bin" / "sumo.exe"),
        "--net-file",
        str(SCENARIO / "single-intersection.net.xml"),
        "--route-files",
        str(SCENARIO / "single-intersection-vhvh.rou.xml"),
        "--seed", "42",
        "--no-step-log", "true",
    ]

    rows = []
    traci.start(command)

    try:
        signal_id = "t"

        # Κάθε θέση του state αντιστοιχεί σε ομάδα συνδέσεων.
        links = traci.trafficlight.getControlledLinks(signal_id)

        program_id = traci.trafficlight.getProgram(signal_id)

        program = next(
            item
            for item in traci.trafficlight.getAllProgramLogics(signal_id)
            if item.programID == program_id
        )

        for phase_index, phase in enumerate(program.phases):
            # Εξετάζουμε μόνο τις πράσινες φάσεις.
            if not any(color in "Gg" for color in phase.state):
                continue

            incoming_lanes = set()

            print(f"\n--- ΠΡΑΣΙΝΗ ΦΑΣΗ {phase_index} ---")
            print(f"Κατάσταση: {phase.state}")

            for link_index, color in enumerate(phase.state):
                if color not in "Gg":
                    continue

                for incoming, outgoing, via in links[link_index]:
                    incoming_lanes.add(incoming)

                    print(f"  {incoming} → {outgoing}")

                    rows.append({
                        "phase": phase_index,
                        "signal_index": link_index,
                        "color": color,
                        "incoming_lane": incoming,
                        "outgoing_lane": outgoing,
                        "via_lane": via,
                    })

            print(
                "Μοναδικές εισερχόμενες λωρίδες:",
                ", ".join(sorted(incoming_lanes)),
            )

    finally:
        traci.close()

    results_dir = ROOT / "results"
    results_dir.mkdir(exist_ok=True)

    pd.DataFrame(rows).to_csv(
        results_dir / "phase_connections.csv",
        index=False,
    )

    print("\nΑποθηκεύτηκε το results/phase_connections.csv")


if __name__ == "__main__":
    main()