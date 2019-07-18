from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import logging
import asyncio
from scipy import misc
import cv2
import numpy as np
import os
import time
import pickle
from sys import platform as _platform
from pyndn import Face, Name, Data, Interest
from pyndn.security import KeyChain
from pyndn.encoding import ProtobufTlv
from asyncndn import fetch_data_packet
from command.repo_command_parameter_pb2 import RepoCommandParameterMessage
from command.repo_command_response_pb2 import RepoCommandResponseMessage


MAX_BYTES_IN_DATA_PACKET = 2000
VIDEO_STREAM_NAME='/local_manager/building_1/camera_1/video/'
dict1 = dict()


def prepare_data(filePath, keychain: KeyChain):
    """
    Shard file into data packets.
    """
    temp_data={}
    logging.info('preparing data for {}'.format(filePath))
    print('preparing data for {}'.format(filePath))
    with open(filePath, 'rb') as binary_file:
        b_array = bytearray(binary_file.read())

    if len(b_array) == 0:
        logging.warning("File is 0 bytes")
        return

    n_packets = int((len(b_array) - 1) / MAX_BYTES_IN_DATA_PACKET + 1)
    print('There are {} packets in total'.format(n_packets))
    seq = 0
    for i in range(0, len(b_array), MAX_BYTES_IN_DATA_PACKET):
        data = Data(Name(VIDEO_STREAM_NAME).append(filePath.split('.')[0]).append(str(seq)))
        data.setContent(b_array[i: min(i + MAX_BYTES_IN_DATA_PACKET, len(b_array))])
        data.metaInfo.setFinalBlockId(Name.Component.fromSegment(n_packets - 1))
        keychain.signWithSha256(data)
        temp_data[str(data.getName())] = data
        seq += 1
    print('{} packets prepared: {}'.format(n_packets, str(Name(VIDEO_STREAM_NAME).append(filePath.split('.')[0]))))
    return temp_data


def on_register_failed(prefix):
    logging.error("Prefix registration failed: %s", prefix)


def on_interest(prefix, interest: Interest, face, _filter_id, _filter):
    logging.info('On interest: {}'.format(interest.getName()))
    print('On interest: {}'.format(interest.getName()))
    if str(interest.getName()[:-1]) in dict1:

        face.putData(dict1[str(interest.getName()[:-1])][str(interest.getName())])
        logging.info('Serve data: {}'.format(interest.getName()))
    else:
        logging.info('Data does not exist: {}'.format(interest.getName()))


def prepare_packets(cur_time: int, keychain: KeyChain):
    key = VIDEO_STREAM_NAME + str(cur_time)
    dict1[key] = prepare_data(str(cur_time) + '.avi', keychain)


def remove_outdated_data(cur_time: int):
    """
    Evict outdated packets from memory and disk.
    """
    if VIDEO_STREAM_NAME + str(cur_time - 10) in dict1:
        dict1.pop(VIDEO_STREAM_NAME + str(cur_time - 10))
        logging.info('Remove data packet: {}'.format(cur_time - 10))
    
    if os.path.exists(str(cur_time - 10) + '.avi'):
        os.remove(str(cur_time - 10) + '.avi')
        logging.info('Removed file: {}'.format(cur_time - 10))


async def capture_video_chunk(duration: int, cap, fourcc) -> str:
    """
    Capture a video chunk of given duration, then save to disk.
    Return the timestamp of the captured video.
    """
    cur_time = int(time.time()) + duration

    if _platform == "linux" or _platform == "linux2":
        filename = str(cur_time) + '.avi'
        out = cv2.VideoWriter(filename, fourcc, 25, (640, 480))
    elif _platform == "darwin":
        frame_width = int(cap.get(3))
        frame_height = int(cap.get(4))
        print("W, H: {}, {}".format(frame_width, frame_height))
        out = cv2.VideoWriter(filename, cv2.VideoWriter_fourcc('M','J','P','G'), 10, 
                            (frame_width, frame_height))
    
    while int(time.time()) < cur_time:
        await asyncio.sleep(0)
        ret, frame = cap.read()
        out.write(frame)
        cv2.imshow('frame', frame)
        
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    
    return cur_time


async def main():

    # Set up face
    async def face_loop():
        nonlocal face, running
        while running:
            face.processEvents()
            await asyncio.sleep(0.001)

    face = Face()
    running = True
    face_event = event_loop.create_task(face_loop())

    # register prefix in local NFD with it own name: /local_manager/building_1/camera_1
    keychain = KeyChain()
    face.setCommandSigningInfo(keychain, keychain.getDefaultCertificateName())
    prefix_id = face.registerPrefix(Name(VIDEO_STREAM_NAME), None, on_register_failed)
    filter_id = face.setInterestFilter(Name(VIDEO_STREAM_NAME), on_interest)
    print('Registered prefix ID {}'.format(prefix_id))
    print('Registered filter ID {}'.format(filter_id))

    cap = cv2.VideoCapture(0)
    fourcc = cv2.VideoWriter_fourcc(*'XVID')

    # Capture a video every second
    while True:
        cur_time = await capture_video_chunk(duration=1, cap=cap, fourcc=fourcc)
        prepare_packets(cur_time=cur_time, keychain=keychain)
        remove_outdated_data(cur_time)

    cap.release()
    cv2.destroyAllWindows()

    running = False
    await face_event


if __name__ == "__main__":
    logging.basicConfig(format='[%(asctime)s]%(levelname)s:%(message)s',
                        datefmt='%Y-%m-%d %H:%M:%S',
                        level=logging.INFO)

    event_loop = asyncio.get_event_loop()
    event_loop.run_until_complete(main())
