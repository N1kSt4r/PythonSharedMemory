from shared_tools import SharedStore_uint8
store = SharedStore_uint8("check", 200, False)

import os
meta = f'client {os.getpid()}:'

import numpy as np
data = {str(i): np.ones(1920 * 1080 * 3, dtype=np.uint8) for i in range(9)}

from time import time as now, sleep
timestamps = [now()]
for i in range(1000):
    store.get_dict(data)
    if len(timestamps) > 100:
        timestamps.pop(0)
    timestamps.append(now())
    print(meta, (timestamps[-1] - timestamps[0]) / 100)
