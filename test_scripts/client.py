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


with SharedStore_uint8("check", 0, False) as store:
    data = {str(i): None for i in range(11)}
    meta = f'client {os.getpid()}:'
    locker = Locker(0.1)

    store.get_dict_init(data)

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
