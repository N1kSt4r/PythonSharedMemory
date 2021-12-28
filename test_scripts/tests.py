import pytest
import numpy as np
from time import sleep
from multiprocessing import Process
from shared_store import SharedStore_uint8, SharedStore_fp32


def test_insert():
    array = np.ones((5, 5))
    with SharedStore_uint8("server", 1, True) as server:
        server.insert_dict({'check': array})
        with pytest.raises(RuntimeError, match='Mismatch data shape'):
            server.insert_dict({'check': array[:2]})
        with pytest.raises(RuntimeError, match='Non contiguous data'):
            server.insert_dict({'another_check': array[:, 2:]})


def test_release():
    array = np.ones((5, 5))
    for i in range(3):
        with SharedStore_uint8("server", 1, True) as server:
            server.insert_dict({'check': array})


def test_double_init():
    with SharedStore_uint8("1", 1, True) as store1:
        print('uint8 1 inited')
        with SharedStore_uint8("2", 1, True) as store2:
            print('uint8 2 inited')
            with pytest.raises(RuntimeError, match='Double init 1 store'):
                with SharedStore_fp32("1", 1, True) as store3:
                    print('fp32 1 inited')


def test_invalidation_timeout():

    def server():
        with SharedStore_uint8("store", 1, True) as server:
            server.insert_dict({'check': np.ones((5, 5))})
            sleep(7)

    server = Process(target=server)
    server.start()

    with SharedStore_uint8("store", 1, False) as client:
        check = {'check': None}
        client.get_dict_init(check)
        assert type(check['check']) == np.ndarray
        assert check['check'].shape == (5, 5)

        sleep(5)
        with pytest.raises(RuntimeError, match='Too old last update'):
            client.get_dict(check)

        server.join(3)
        sleep(1)

        with pytest.raises(RuntimeError, match='Too old last update'):
            client.get_dict(check)


def test_not_inited():
    with pytest.raises(RuntimeError, match='Shared store not_exist is not inited in client mode'):
        with SharedStore_fp32("not_exist", 1, False):
            pass
