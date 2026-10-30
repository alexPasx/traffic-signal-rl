import traci


class AdaptiveController:
    def __init__(self, signal_id="t"):
        self.signal_id = signal_id

        self.min_green = 5.0
        self.max_green = 40.0
        self.yellow_duration = 2.0
        self.queue_margin = 3

        self.phase_lanes = {
            0: ("n_t_0", "s_t_0"),
            2: ("n_t_1", "s_t_1"),
            4: ("e_t_0", "w_t_0"),
            6: ("e_t_1", "w_t_1"),
        }

        self.phase = traci.trafficlight.getPhase(
            self.signal_id
        )

        self.phase_started = traci.simulation.getTime()

        # Αποτρέπουμε τις αυτόματες αλλαγές φάσεων.
        # Όλες οι μεταβάσεις θα γίνονται από την update().
        traci.trafficlight.setPhaseDuration(
            self.signal_id,
            100_000.0,
        )

    def queue_size(self, phase):
        return sum(
            traci.lane.getLastStepHaltingNumber(lane)
            for lane in self.phase_lanes[phase]
        )

    def switch_to(self, next_phase):
        traci.trafficlight.setPhase(
            self.signal_id,
            next_phase,
        )

        traci.trafficlight.setPhaseDuration(
            self.signal_id,
            100_000.0,
        )

        self.phase = next_phase
        self.phase_started = traci.simulation.getTime()

    def update(self):
        now = traci.simulation.getTime()
        elapsed = now - self.phase_started

        # Κίτρινο: κρατά ακριβώς 2 δευτερόλεπτα.
        if self.phase not in self.phase_lanes:
            if elapsed >= self.yellow_duration:
                self.switch_to((self.phase + 1) % 8)

            return

        # Πράσινο: δεν επιτρέπεται αλλαγή πριν τα 5 s.
        if elapsed < self.min_green:
            return

        queues = {
            phase: self.queue_size(phase)
            for phase in self.phase_lanes
        }

        current_queue = queues[self.phase]

        largest_other_queue = max(
            queue
            for phase, queue in queues.items()
            if phase != self.phase
        )

        reached_limit = elapsed >= self.max_green
        empty = current_queue == 0

        other_is_busier = (
            largest_other_queue
            >= current_queue + self.queue_margin
        )

        if reached_limit or empty or other_is_busier:
            # Πάντα στην επόμενη κίτρινη, ποτέ απευθείας
            # σε άλλο πράσινο.
            self.switch_to(self.phase + 1)