import os
from pathlib import Path

import sumo

# Ρυθμίζουμε τη διαδρομή μέσα στον κώδικα,
# ώστε να λειτουργεί και από νέο terminal.
os.environ["SUMO_HOME"] = str(sumo.SUMO_HOME)

import sumo_rl
import traci


ROOT = Path(__file__).resolve().parent

# Εντοπίζουμε το σενάριο μέσα στην εγκατεστημένη βιβλιοθήκη.
SCENARIO = (
    Path(sumo_rl.__file__).resolve().parent
    / "nets"
    / "2way-single-intersection"
)

NET_FILE = SCENARIO / "single-intersection.net.xml"
ROUTE_FILE = SCENARIO / "single-intersection-vhvh.rou.xml"

SUMO_GUI = Path(sumo.SUMO_HOME) / "bin" / "sumo-gui.exe"


def main():
    for path in [NET_FILE, ROUTE_FILE, SUMO_GUI]:
        if not path.is_file():
            raise FileNotFoundError(f"Δεν βρέθηκε: {path}")

    results_dir = ROOT / "results"
    results_dir.mkdir(exist_ok=True)

    command = [
        str(SUMO_GUI),
        "--net-file", str(NET_FILE),
        "--route-files", str(ROUTE_FILE),
        "--seed", "42",
        "--step-length", "1",
        "--delay", "100",
        "--start",
        "--tripinfo-output", str(results_dir / "fixed_demo_trips.xml"),
        "--tripinfo-output.write-unfinished", "true",
    ]

    traci.start(command)

    try:
        print("Φανάρια:", traci.trafficlight.getIDList())
        print("Εκτέλεση σταθερού προγράμματος για 600 δευτερόλεπτα.")

        while traci.simulation.getTime() < 600:
            traci.simulationStep()

            time = int(traci.simulation.getTime())

            if time % 60 == 0:
                vehicles = traci.vehicle.getIDList()

                stopped = sum(
                    traci.vehicle.getSpeed(vehicle_id) < 0.1
                    for vehicle_id in vehicles
                )

                print(
                    f"Χρόνος: {time:3d} s | "
                    f"Οχήματα στο δίκτυο: {len(vehicles)} | "
                    f"Σταματημένα: {stopped}"
                )

    finally:
        traci.close()

    print("Η προσομοίωση ολοκληρώθηκε.")


if __name__ == "__main__":
    main()