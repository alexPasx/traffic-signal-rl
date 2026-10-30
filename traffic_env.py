import os
from pathlib import Path

import sumo

os.environ["SUMO_HOME"] = str(sumo.SUMO_HOME)

import gymnasium as gym
import numpy as np
import sumo_rl
import traci

from gymnasium import spaces
from adaptive_controller import AdaptiveController


SCENARIO = (
    Path(sumo_rl.__file__).resolve().parent
    / "nets"
    / "2way-single-intersection"
)


class TrafficEnv(gym.Env):
    def __init__(self, duration=600):
        super().__init__()

        self.duration = duration
        self.signal_id = "t"
        self.connection = None

        self.lanes = [
            "n_t_0", "n_t_1",
            "s_t_0", "s_t_1",
            "e_t_0", "e_t_1",
            "w_t_0", "w_t_1",
        ]

        # 0: διατήρηση πράσινου
        # 1: ολοκλήρωση πράσινου
        self.action_space = spaces.Discrete(2)

        # 8 ουρές + 8 πλήθη οχημάτων + 8 ενδείξεις φάσης
        # + χρόνος φάσης + χρόνος επεισοδίου + εκκρεμείς εισαγωγές.
        self.observation_space = spaces.Box(
            low=0.0,
            high=np.inf,
            shape=(27,),
            dtype=np.float32,
        )

    def observation(self):
        queues = [
            traci.lane.getLastStepHaltingNumber(lane) / 20.0
            for lane in self.lanes
        ]

        counts = [
            traci.lane.getLastStepVehicleNumber(lane) / 20.0
            for lane in self.lanes
        ]

        phase_vector = np.zeros(8, dtype=np.float32)
        phase_vector[self.controller.phase] = 1.0

        now = traci.simulation.getTime()

        elapsed = (
            now - self.controller.phase_started
        ) / self.controller.max_green

        progress = now / self.duration

        pending = len(
            traci.simulation.getPendingVehicles()
        ) / 20.0

        return np.array(
            queues
            + counts
            + phase_vector.tolist()
            + [elapsed, progress, pending],
            dtype=np.float32,
        )

    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed)
        self.close()

        # Διαφορετικό seed προσομοίωσης σε κάθε νέο επεισόδιο.
        simulation_seed = int(
            self.np_random.integers(0, 1_000_000)
        )

        # Για αξιολόγηση μπορούμε να ζητήσουμε συγκεκριμένο SUMO seed.
        if options is not None and "sumo_seed" in options:
            simulation_seed = int(options["sumo_seed"])

        command = [
            str(Path(sumo.SUMO_HOME) / "bin" / "sumo.exe"),
            "--net-file",
            str(SCENARIO / "single-intersection.net.xml"),
            "--route-files",
            str(SCENARIO / "single-intersection-vhvh.rou.xml"),
            "--seed", str(simulation_seed),
            "--step-length", "1",
            "--no-step-log", "true",
        ]

        traci.start(command)
        self.connection = traci.getConnection()

        # Χρησιμοποιούμε τα ήδη ελεγμένα χρονικά όρια
        # και τη συνάρτηση μετάβασης του ελεγκτή μας.
        # Δεν καλούμε τη δική του update() με τους κανόνες ουρών.
        self.controller = AdaptiveController(
            signal_id=self.signal_id
        )

        self.finished = False

        return self.observation(), {
            "sumo_seed": simulation_seed,
        }

    def step(self, action):
        if self.finished:
            raise RuntimeError(
                "Το επεισόδιο τελείωσε. Κάλεσε reset()."
            )

        action = int(action)

        if not self.action_space.contains(action):
            raise ValueError(f"Μη έγκυρη ενέργεια: {action}")

        now = traci.simulation.getTime()
        elapsed = now - self.controller.phase_started
        phase = self.controller.phase

        if phase in self.controller.phase_lanes:
            # Στο ανώτατο όριο η αλλαγή είναι υποχρεωτική.
            must_switch = elapsed >= self.controller.max_green

            requested_switch = (
                action == 1
                and elapsed >= self.controller.min_green
            )

            if must_switch or requested_switch:
                self.controller.switch_to(phase + 1)

        else:
            # Το κίτρινο ολοκληρώνεται ανεξάρτητα από την ενέργεια.
            if elapsed >= self.controller.yellow_duration:
                self.controller.switch_to((phase + 1) % 8)

        traci.simulationStep()

        vehicles = traci.vehicle.getIDList()

        stopped = sum(
            traci.vehicle.getSpeed(vehicle_id) < 0.1
            for vehicle_id in vehicles
        )

        pending = len(
            traci.simulation.getPendingVehicles()
        )

        # Ένα βήμα = ένα δευτερόλεπτο.
        # Μετράμε και την αναμονή πριν από την είσοδο στο δίκτυο.
        reward = -float(stopped + pending) / 100.0

        truncated = (
            traci.simulation.getTime() >= self.duration
        )

        self.finished = truncated

        info = {
            "time_s": traci.simulation.getTime(),
            "stopped": stopped,
            "pending_insertion": pending,
            "phase": self.controller.phase,
            "arrived_this_step": (
                traci.simulation.getArrivedNumber()
            ),
        }

        return (
            self.observation(),
            reward,
            False,
            truncated,
            info,
        )

    def close(self):
        if self.connection is not None:
            try:
                self.connection.close()
            finally:
                self.connection = None