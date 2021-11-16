from shared_tools import SharedStore_uint8
store = SharedStore_uint8("check", 400, True)

import os
meta = f'server {os.getpid()}:'

import numpy as np

data = {str(i): np.ones(1920 * 1080 * 3, dtype=np.uint8) for i in range(9)}
print(data)

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

from time import time as now
timestamps = [now()]
locker = Locker(0.1)
for i in range(100):
    locker.wait()
    write_start = now()
    store.insert_dict(data)
    if len(timestamps) > 100:
        timestamps.pop(0)
    write_end = now()
    timestamps.append(write_end)
    print(meta, (timestamps[-1] - timestamps[0]) / 100, write_end - write_start)

store.finalize()

# 0.009 per write 9 images FullHD, but without sync
