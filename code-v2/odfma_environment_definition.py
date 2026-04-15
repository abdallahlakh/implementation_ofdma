

# ============================================================================
# OFDMA ENVIRONMENT WITH STATIONS
# ============================================================================

class OFDMAEnvironment:
    
    def __init__(self, n_stations: int = 5, bandwidth_mhz: int = 20, ru_size: int = 26):
        self.channel = OFDMAChannel(bandwidth_mhz=bandwidth_mhz, ru_size=ru_size)
        self.n_stations = n_stations
        self.n_rus = len(self.channel.rus)
        
        # Station state: queue_size, deadline_pressure, total_success
        self.station_queues = np.ones(n_stations, dtype=np.int32) * 5  # Initial queue
        self.station_deadlines = np.ones(n_stations, dtype=np.float32) * 10
        self.station_success = np.zeros(n_stations, dtype=np.float32)
        
        self.step_count = 0
        self.max_steps = 300
        
    def reset(self):
        """Reset environment."""
        self.channel = OFDMAChannel(bandwidth_mhz=20, ru_size=26)
        self.station_queues = np.ones(self.n_stations, dtype=np.int32) * 5
        self.station_deadlines = np.ones(self.n_stations, dtype=np.float32) * 10
        self.station_success = np.zeros(self.n_stations, dtype=np.float32)
        self.step_count = 0
        return self._get_state()
    


    #this is the global state of the enviroment 
    def _get_state(self) -> np.ndarray:
        """Get current state representation."""

        state = []
        for ru in self.channel.rus:
            alloc_status = 0.0 if ru.allocated_to is None else 1.0
            ruqi = ru.RUqi

            state.append([alloc_status, ruqi])

        queue_features = self.station_queues.astype(np.float32) / 20.0
        deadline_features = self.station_deadlines.astype(np.float32) / 10.0
        success_total = float(np.sum(self.station_success)) + 1e-6
        success_features = (self.station_success.astype(np.float32) / success_total)
        global_features = np.array([
            np.mean(queue_features),
            np.mean(deadline_features),
            self.step_count / float(self.max_steps),
        ], dtype=np.float32)

        return np.concatenate([
            np.array(state, dtype=np.float32).flatten(),
            queue_features,
            deadline_features,
            success_features,
            global_features,
        ]).astype(np.float32)
    


    def step(self, actions: np.ndarray) -> tuple:
        '''
        station_success: long-term cumulative history 
        per station.
        allocation_count: short-term current-step (or recent)
        allocation result per station.
        
        '''

        self.step_count += 1
        
        # Reset allocations
        for ru in self.channel.rus:
            ru.release()

        # Count successful and failed allocation attempts.
        allocation_count = np.zeros(self.n_stations, dtype=np.int32)
        allocation_reward = 0.0
        attempted_allocations = 0
        failed_allocations = 0

        #important note here i use np.random.rand to allocate just for simplcity u need to define a number as threshold to be compared with ruqi
        # Apply actions
        for ru_id, action in enumerate(actions):
            if action > 0:  # Action 0 means don't allocate
                station_id = int(action - 1)  # Convert to 0-indexed
                if 0 <= station_id < self.n_stations:
                    attempted_allocations += 1
                    ruqi = self.channel.rus[ru_id].RUqi

                    # Small transmission attempt cost.
                    allocation_reward -= 0.03

                    # Success probability is RUQI-driven rather than hard threshold.
                    if np.random.rand() < ruqi:
                        self.channel.allocate_ru(ru_id, station_id)
                        allocation_count[station_id] += 1
                        allocation_reward += 0.8 + 0.8 * ruqi
                    else:
                        failed_allocations += 1
                        allocation_reward -= 0.05
        
        # Update station state
        for i in range(self.n_stations):
            if allocation_count[i] > 0:
                # Successful transmissions
                self.station_success[i] += allocation_count[i]
                self.station_queues[i] = max(0, self.station_queues[i] - allocation_count[i])
  #this means that each station gets 0.5 new packets on average at each step and the queue size will stop at 20 for simplicity

            # Queue arrivals (Poisson-like)
            arrivals = np.random.poisson(0.4)
            self.station_queues[i] = min(20, self.station_queues[i] + arrivals)
            
            #here we will decrease the deadline pressure by 1 at each step and reset it to 10 if the station transmitted successfully

            # Deadline countdown
            self.station_deadlines[i] = max(0, self.station_deadlines[i] - 1)
            
            # Reset deadline if station transmitted successfully
            if allocation_count[i] > 0:
                self.station_deadlines[i] = 10
        
        # Compute reward
        # Reward = successful allocations - queue backlog - fairness
        queue_penalty = np.sum(self.station_queues) * 0.04
        fairness_penalty = np.std(self.station_success + 1e-6) * 0.02
        deadline_pressure_penalty = np.mean(1.0 - (self.station_deadlines / 10.0)) * 0.1

        reward = allocation_reward - queue_penalty - fairness_penalty - deadline_pressure_penalty
        reward = float(np.clip(reward, -6.0, 6.0))
        
        # Update state
        next_state = self._get_state()
        
        done = self.step_count >= self.max_steps

        successful_allocations = int(np.sum(allocation_count))
        throughput = float(successful_allocations)
        collision_rate = (
            float(failed_allocations / attempted_allocations)
            if attempted_allocations > 0
            else 0.0
        )
        
        info = {
            "allocation_count": allocation_count,
            "queues": self.station_queues.copy(),
            "deadlines": self.station_deadlines.copy(),
            "fairness": np.std(self.station_success),
            "successful_allocations": successful_allocations,
            "attempted_allocations": int(attempted_allocations),
            "failed_allocations": int(failed_allocations),
            "throughput": throughput,
            "collision_rate": collision_rate,
        }
        
        return next_state, reward, done, info
