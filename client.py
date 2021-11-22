from shared_tools import SharedStore_uint8
store = SharedStore_uint8("check", 0, False)

import os
meta = f'client {os.getpid()}:'

import numpy as np
data = {str(i): np.ones((1920, 1080, 3), dtype=np.uint8) for i in range(7)}

from time import time as now, sleep
class Locker:
    def __init__(self, period):
        self.period = period
        self.next_update = now() + self.period

    def reset(self):
        self.next_update = now()

    def wait(self):
        time_to_wait = self.next_update - now()
        if time_to_wait > 0:
            sleep(time_to_wait)
        elif time_to_wait < -self.period:
            self.reset()
        self.next_update += self.period

from time import time as now, sleep
locker = Locker(0.1)
timestamps = [now()]
for i in range(250):
    locker.wait()
    read_start = now()
    store.get_dict(data)
    read_end = now()
    if len(timestamps) > 100:
        timestamps.pop(0)
    timestamps.append(now())
    print(meta, (timestamps[-1] - timestamps[0]) / 100, read_end - read_start)
