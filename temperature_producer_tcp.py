import os
import json
import asyncio
import logging
import socket
import random
import time
import threading


class TemperatureProducerTcp(object):

    def __init__(self):
        self.socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self.ip = 'localhost'
        self.port = 50000

    def get_temp(self) -> int:
        return random.randint(0, 35)

    def run(self):
        self.socket.bind((self.ip, self.port))
        self.socket.listen(1)
        logging.info('Listening on {}:{}'.format(self.ip, self.port))

        while True:
            conn, addr = self.socket.accept()
            # event_loop.create_task(self.handle_connection(conn, addr))
            logging.info('Received connection from {}'.format(addr))
            t = threading.Thread(target=self.worker, args=(conn, addr))
            t.start()

    def worker(self, conn, addr):
        logging.info('Worker started')
        asyncio.set_event_loop(asyncio.new_event_loop())
        event_loop = asyncio.get_event_loop()
        event_loop.run_until_complete(self.handle_connection(conn, addr))

    async def handle_connection(self, conn, addr):
        logging.info('Accepted connection from {}'.format(addr))
        while True:
            data_dict = {
                'temp': self.get_temp(),
                'location': 'Bedroom',
                'timestamp': int(time.time())
            }
            conn.sendall(str.encode(json.dumps(data_dict)))
            await asyncio.sleep(1)
        conn.close()


if __name__ == "__main__":
    logging.basicConfig(format='[%(asctime)s]%(levelname)s:%(message)s',
                        datefmt='%Y-%m-%d %H:%M:%S',
                        level=logging.INFO)

    p = TemperatureProducerTcp()

    event_loop = asyncio.get_event_loop()
    event_loop.create_task(p.run())

