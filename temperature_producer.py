import argparse
import asyncio
import logging
import random
import time
from typing import Optional
from pyndn import Face, Name, Data, Interest, Blob
from pyndn.security import KeyChain
from pyndn.encoding import ProtobufTlv
from asyncndn import fetch_data_packet
from command.repo_command_parameter_pb2 import RepoCommandParameterMessage
from command.repo_command_response_pb2 import RepoCommandResponseMessage

use_repo = False

class TemperatureProducer(object):
    """
    A temperature sensor that publishes data periodically.
    Temperature Data packets have the name of /<prefix>/<timestamp>
    """
    def __init__(self, prefix: Name, repo_name: Optional[Name]):
        self.prefix = prefix
        self.repo_name = repo_name
        self.face = Face()
        self.keychain = KeyChain()
        self.running = True
        self.name_str_to_data = dict()
        self.face.setCommandSigningInfo(self.keychain, self.keychain.getDefaultCertificateName())
        self.face.registerPrefix(self.prefix, None,
                                 lambda prefix: logging.error("Prefix registration failed: %s", prefix))
        self.face.setInterestFilter(self.prefix, self.on_interest)

        event_loop = asyncio.get_event_loop()
        event_loop.create_task(self.face_loop())

    async def send_cmd_interest(self):
        event_loop = asyncio.get_event_loop()
        face_task = event_loop.create_task(self.face_loop())

        parameter = RepoCommandParameterMessage()
        for compo in self.prefix:
            parameter.repo_command_parameter.name.component.append(compo.getValue().toBytes())
        parameter.repo_command_parameter.start_block_id = int(time.time())
        parameter.repo_command_parameter.end_block_id = parameter.repo_command_parameter.start_block_id
        param_blob = ProtobufTlv.encode(parameter)

        # Prepare cmd interest
        name = Name(self.repo_name).append("insert").append(Name.Component(param_blob))
        interest = Interest(name)
        interest.canBePrefix = True
        self.face.makeCommandInterest(interest)

        logging.info("Express interest: {}".format(interest.getName()))
        ret = await fetch_data_packet(self.face, interest)

        if not isinstance(ret, Data):
            logging.warning("Insertion failed")
        else:
            # Parse response
            response = RepoCommandResponseMessage()
            try:
                ProtobufTlv.decode(response, ret.content)
                logging.info('Insertion command accepted: status code {}'
                             .format(response.repo_command_response.status_code))
            except RuntimeError as exc:
                logging.warning('Response decoding failed', exc)

    async def face_loop(self):
        while self.running:
            self.face.processEvents()
            await asyncio.sleep(0.001)

    def get_temp(self) -> int:
        return random.randint(0, 35)

    def publish_temp_packet(self):
        data_name = Name(self.prefix).append(str(int(time.time())))
        data = Data(data_name)

        temp = self.get_temp()
        content_blob = Blob(temp.to_bytes(2, byteorder='little'))
        data.setContent(content_blob)
        data.metaInfo.setFreshnessPeriod(1000000)

        logging.info('Publish temp data {}, {} degree'.format(data.getName(), temp))

        self.keychain.sign(data)
        self.name_str_to_data[str(data.getName())] = data
        if use_repo is True:
            event_loop = asyncio.get_event_loop()
            event_loop.create_task(self.send_cmd_interest())
            logging.info("send repo insertion command")

    def on_interest(self, _prefix, interest: Interest, face, _filter_id, _filter):
        name = str(interest.getName())
        if name in self.name_str_to_data:
            self.face.putData(self.name_str_to_data[name])
            logging.info('Serve data: {}'.format(name))

    async def run(self):
        """
        Need to publish data with period of at least 1 second, otherwise Data
        packets are not immutable
        """
        while self.running:
            self.publish_temp_packet()
            await asyncio.sleep(1)


def main():
    parser = argparse.ArgumentParser(description='segmented insert client')
    parser.add_argument('-r', '--repo_enable', help='when with this option, Repo will be used.', action='store_true')
    args = parser.parse_args()
    if args.repo_enable:
        global use_repo
        use_repo = True

    prefix = Name('/home')
    sensor = TemperatureProducer(prefix, Name('testrepo'))

    event_loop = asyncio.get_event_loop()
    event_loop.run_until_complete(sensor.run())


if __name__== "__main__":
    logging.basicConfig(format='[%(asctime)s]%(levelname)s:%(message)s',
                        datefmt='%Y-%m-%d %H:%M:%S',
                        level=logging.INFO)
    main()
