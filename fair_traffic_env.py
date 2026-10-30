import numpy as np
import traci
from gymnasium import spaces

from traffic_env import TrafficEnv


class FairTrafficEnv(TrafficEnv):
    def __init__(
        self,
        duration=600,
        wait_threshold=30.0,
        fairness_weight=2.0,
    ):
        super().__init__(duration=duration)

        self.wait_threshold = wait_threshold
        self.fairness_weight = fairness_weight

        self.vehicle_wait = {}

        self.observation_space = spaces.Box(
            low=0.0,
            high=np.inf,
            shape=(35,),
            dtype=np.float32,
        )

    def reset(self, *, seed=None, options=None):
        self.vehicle_wait = {}

        return super().reset(
            seed=seed,
            options=options,
        )

    def observation(self):
        original = super().observation()

        lane_waits = []

        for lane in self.lanes:
            vehicles = traci.lane.getLastStepVehicleIDs(lane)

            max_wait = max(
                (
                    self.vehicle_wait.get(vehicle_id, 0.0)
                    for vehicle_id in vehicles
                ),
                default=0.0,
            )

            lane_waits.append(
                max_wait / self.wait_threshold
            )

        return np.concatenate([
            original,
            np.array(lane_waits, dtype=np.float32),
        ]).astype(np.float32)

    def step(self, action):
        _, base_reward, terminated, truncated, info = super().step(
            action
        )

        vehicles = traci.vehicle.getIDList()

        # Αθροιστικός χρόνος ακινησίας από την είσοδο
        # κάθε οχήματος στο τρέχον επεισόδιο.
        for vehicle_id in vehicles:
            self.vehicle_wait.setdefault(vehicle_id, 0.0)

            if traci.vehicle.getSpeed(vehicle_id) < 0.1:
                self.vehicle_wait[vehicle_id] += 1.0

        # Μόνο τα οχήματα που παραμένουν τώρα στο δίκτυο
        # συνεισφέρουν στην πρόσθετη ποινή.
        excess = sum(
            max(
                0.0,
                self.vehicle_wait[vehicle_id]
                - self.wait_threshold,
            ) / self.wait_threshold
            for vehicle_id in vehicles
        )

        fairness_penalty = (
            self.fairness_weight * excess / 100.0
        )

        reward = base_reward - fairness_penalty

        info = dict(info)
        info["base_reward"] = float(base_reward)
        info["fairness_penalty"] = float(fairness_penalty)

        info["vehicles_over_threshold"] = sum(
            self.vehicle_wait[vehicle_id] > self.wait_threshold
            for vehicle_id in vehicles
        )

        return (
            self.observation(),
            float(reward),
            terminated,
            truncated,
            info,
        )