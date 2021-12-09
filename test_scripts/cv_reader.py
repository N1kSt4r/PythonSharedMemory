from shared_store import SharedStore_uint8
import numpy as np
import cv2


fps = 5
with SharedStore_uint8("check", 400, False) as store:
    data = {"check": None}
    store.get_dict_init(data)

    for i in range(10000):
        store.get_dict(data)
        cv2.imshow("check", data["check"])
        cv2.waitKey(1000 // fps)
        # Epilepsy warning
