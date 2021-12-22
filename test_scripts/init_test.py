from shared_store import SharedStore_uint8, SharedStore_fp32

with SharedStore_uint8("1", 1, True) as store1:
    print('uint8 1 inited')
    with SharedStore_uint8("2", 1, True) as store2:
        print('uint8 2 inited')
        with SharedStore_fp32("1", 1, True) as store3:
            print('fp32 1 inited')
            pass