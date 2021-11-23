from shared_store import SharedStore_uint8
import numpy as np
import cv2

data = {"check": np.full((416, 416, 3), 120, dtype=np.uint8)}

store = SharedStore_uint8("check", 400, False)
for i in range(10000):
    store.get_dict(data)
    cv2.imshow("check", data["check"])
    cv2.waitKey(150)

store.finalize()
