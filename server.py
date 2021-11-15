from shared_tools import SharedStore_uint8
store = SharedStore_uint8("check", 400, True)

import os
meta = f'server {os.getpid()}:'

import numpy as np

data = {str(i): np.ones(1920 * 1080 * 3, dtype=np.uint8) for i in range(9)}
print(data)

from time import time as now
timestamps = [now()]
for i in range(2000):
    store.insert_dict(data)
    if len(timestamps) > 100:
        timestamps.pop(0)
    timestamps.append(now())
    print(meta, (timestamps[-1] - timestamps[0]) / 100)

store.finalize()

# 0.009 per write 9 images FullHD, but without sync
