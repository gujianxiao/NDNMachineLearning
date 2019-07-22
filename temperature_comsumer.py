import asyncio
import logging
import random
import os
import pandas as pd
import random
import time
from datetime import datetime
from pyndn import Face, Name, Data, Interest, Blob
from pyndn.security import KeyChain
from asyncndn import fetch_data_packet
from incremental_learner import IncrementalLearner

class TemperatureConsumer(object):
    def __init__(self, prefix: Name):
        self.face = Face()
        self.keychain = KeyChain()
        self.running = True
        self.prefix = prefix
        self.learner = IncrementalLearner()
        self.face.setCommandSigningInfo(self.keychain, self.keychain.getDefaultCertificateName())
        self.face.registerPrefix(self.prefix, None,
                                 lambda prefix: logging.error("Prefix registration failed: %s", prefix))
        self.empty_data_frame = pd.DataFrame([], columns=['Time', 'DistrictCode', 'TypeCode', 'Popularity'])
        self.data_frame = self.empty_data_frame
        self.batch_size = 10
        
        event_loop = asyncio.get_event_loop()
        event_loop.create_task(self.face_loop())
    
    async def face_loop(self):
        while self.running:
            self.face.processEvents()
            await asyncio.sleep(0.001)

    def set_batchsize(self, batch_size: int):
        """
        Call this function to set the training batch size
        """
        self.batch_size = batch_size
    
    async def send_temp_interest(self):
        """
        Send a temperature interest to the producer
        """
        interest_name = Name(self.prefix).append(str(int(time.time()) - 1))
        interest = Interest(interest_name)
        interest.interestLifetimeMilliseconds = 4000

        logging.info('Fetching {}'.format(str(interest.getName())))
        print('Fetching {}'.format(str(interest.getName())))

        data = await fetch_data_packet(self.face, interest)
        if isinstance(data, Data):
            self.process_temp_data(data)
        else:
            logging.info('Failed to fetch {}'.format(str(interest.getName())))
    
    def process_temp_data(self, data: Data):
        """
        Parse the received Data packet containing temperature info
        """
        content_bytes = data.getContent().toBytes()
        temperature = int.from_bytes(content_bytes, byteorder='little')
        logging.info('Received {}: {} degrees'.format(str(data.getName()), temperature))
        print('Received {}: {} degrees'.format(str(data.getName()), temperature))

        self.data_frame = self.data_frame.append({
            'Time': datetime.now().strftime('%Y-%m-%d-%H'),
            'DistrictCode': random.randint(0, 5),
            'TypeCode': random.randint(0, 5),
            'Popularity': temperature
        }, ignore_index=True)

        # If collected a batch of data, perform incremental learning on it
        print('len: {}'.format(len(self.data_frame)))
        if len(self.data_frame) >= self.batch_size + 24:
            csv_name = str(int(time.time())) + '.csv'
            csv_path = os.path.join('data', csv_name)
            self.data_frame.to_csv(csv_path, index=False)
            self.data_frame = self.data_frame[-24:]
            
            logging.info('Start incremental training on batch {}'.format(csv_name))

            self.learner.load_data(csv_path)
            self.learner.train_once()

    async def run(self):
        """
        Periodically send interest for temperature info
        """
        while self.running:
            print('Running')
            await self.send_temp_interest()
            await asyncio.sleep(1)


def main():
    prefix = Name('/home')
    sensor = TemperatureConsumer(prefix)

    event_loop = asyncio.get_event_loop()
    event_loop.run_until_complete(sensor.run())


if __name__== "__main__":
    logging.basicConfig(format='[%(asctime)s]%(levelname)s:%(message)s',
                        datefmt='%Y-%m-%d %H:%M:%S',
                        level=logging.INFO)
    main()