import os
import numpy as np
from time import time as now, sleep
from shared_store import SharedStore_uint8


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


with SharedStore_uint8("check", 100, True) as store:
    data = {str(i): np.ones((1920, 1080, 3), dtype=np.uint8) for i in range(11)}
    meta = f'server {os.getpid()}:'
    locker = Locker(0.1)

    timestamps = [now()]
    for i in range(150):
        locker.wait()
        write_start = now()
        store.insert_dict(data)
        write_end = now()
        if len(timestamps) > 100:
            timestamps.pop(0)
        timestamps.append(now())
        print(meta, (timestamps[-1] - timestamps[0]) / 100, write_end - write_start)
