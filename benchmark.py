import os
from time import sleep
from multiprocessing import Process


def run(count_clients):
    server = Process(target=lambda: os.system("python3 server.py"))
    server.start()
    sleep(2)

    clients = [Process(target=lambda: os.system("python3 client.py"))
               for _ in range(count_clients)]
    for client in clients:
        client.start()

    server.join()
    for client in clients:
        client.join()


run(1)
