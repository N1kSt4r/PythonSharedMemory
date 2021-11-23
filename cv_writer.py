from shared_store import SharedStore_uint8
from time import sleep
import numpy as np

store = SharedStore_uint8("check", 400, True)
for i in range(100000):
    store.insert_dict({"check": np.full((416, 416, 3), 0 if i % 2 == 0 else 255, dtype=np.uint8)})
    print(0 if i % 2 == 0 else 255)
    sleep(1/30)
    
store.finalize()
