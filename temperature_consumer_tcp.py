import os
import json
import socket
import logging

class TemperatureConsumerTcp(object):

    def __init__(self):
        self.socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self.server_ip = 'localhost'
        self.server_port = 50000

    def run(self):
        self.socket.connect((self.server_ip, self.server_port))
        logging.info('Connected to {}:{}'.format(self.server_ip, self.server_port))

        while True:
            data = self.socket.recv(1024)
            data_dict = json.loads(data)
            logging.info('Received {}'.format(data_dict))

        self.socket.close()


if __name__ == "__main__":
    logging.basicConfig(format='[%(asctime)s]%(levelname)s:%(message)s',
                        datefmt='%Y-%m-%d %H:%M:%S',
                        level=logging.INFO)

    c = TemperatureConsumerTcp()
    c.run()

