import numpy as np

#here i define a full controled OFDMA channel that has inside it resource units and inside resource unit we have the subcarriers .
class ResourceUnit:
    def __init__(self, ru_id, subcarriers):
        self.ru_id = ru_id
        self.subcarriers = subcarriers
        self.allocated_to = None  # station ID or None if free
        self.RUqi = np.random.rand()  # random CQI value between 0 and 1

    def allocate(self, station_id):
        if self.allocated_to is None:
            self.allocated_to = station_id
            return True
        return False

    def release(self):
        self.allocated_to = None

    def get_subcarriers(self):
        return self.subcarriers

    def __repr__(self):
        return f"RU-{self.ru_id}: {len(self.subcarriers)} subcarriers, RUQI={self.RUqi:.2f}, allocated to {self.allocated_to}"

class OFDMAChannel:
    def __init__(self, bandwidth_mhz=20, ru_size=26):
        self.bandwidth_mhz = bandwidth_mhz
        self.subcarriers = int(bandwidth_mhz * 12)
        self.ru_size = ru_size
        self.rus = self._create_rus()

    def _create_rus(self):
        num_rus = self.subcarriers // self.ru_size
        rus = []
        for i in range(num_rus):
            start = i * self.ru_size
            end = start + self.ru_size
            rus.append(ResourceUnit(i, list(range(start, end))))
        return rus

    def allocate_ru(self, ru_id, station_id):
        return self.rus[ru_id].allocate(station_id)

    def release_ru(self, ru_id):
        self.rus[ru_id].release()

    def status(self):
        return [str(ru) for ru in self.rus]

    def get_state(self):
       # RL state: each RU is [allocation_status, RU_QI]
        state = []
        for ru in self.rus:
            alloc = 0 if ru.allocated_to is None else 1
            state.append([alloc, ru.RUqi])
        return np.array(state) 
          # shape = (num_rus, 2)


if __name__ == "__main__":
    # Example usage with allocations
    channel = OFDMAChannel(bandwidth_mhz=20, ru_size=26)

    print("\nInitial RUs:")
    for ru in channel.status():
        print(ru)

    # Allocate RU-0 to STA-1
    channel.allocate_ru(0, "STA-1")

    # Allocate RU-2 to STA-2
    channel.allocate_ru(2, "STA-2")

    # Allocate RU-3 to STA-3
    channel.allocate_ru(3, "STA-3")

    print("\nAfter allocations:")
    for ru in channel.status():
        print(ru)

    # Show RL state vector
    print("\nRL State Vector:", channel.get_state())
