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
from pyndn import Face, Name, Data, Interest
from pyndn.security import KeyChain
from pyndn.encoding import ProtobufTlv
from asyncndn import fetch_data_packet
from command.repo_command_parameter_pb2 import RepoCommandParameterMessage
from command.repo_command_response_pb2 import RepoCommandResponseMessage


MAX_BYTES_IN_DATA_PACKET = 2000
print('Start capture')
video_stream_name='/local_manager/building_1/camera_1/'

#deal with network layer
#segmenation function
def prepare_data(filePath):
    """
    Shard file into data packets.
    """
    temp_data={}
    logging.info('preparing data')
    with open(filePath, 'rb') as binary_file:
        b_array = bytearray(binary_file.read())

    if len(b_array) == 0:
        logging.warning("File is 0 bytes")
        return

    n_packets = int((len(b_array) - 1) / MAX_BYTES_IN_DATA_PACKET + 1)
    print('There are {} packets in total'.format(n_packets))
    seq = 0
    for i in range(0, len(b_array), MAX_BYTES_IN_DATA_PACKET):
        print(i)
        data = Data(Name(os.path.splitext(filePath)[0]).append(str(seq)))
        data.setContent(b_array[i: min(i + MAX_BYTES_IN_DATA_PACKET, len(b_array))])
        data.metaInfo.setFinalBlockId(Name.Component.fromSegment(n_packets - 1))
        keychain.signWithSha256(data)
        temp_data[str(data.getName())] = data
        seq += 1
    return temp_data

def on_register_failed(prefix):
    logging.error("Prefix registration failed: %s", prefix)

def on_interest(self, _prefix, interest: Interest, face, _filter_id, _filter):
    logging.info('On interest: {}'.format(interest.getName()))
    if str(interest.getName())[0:44] in dict1:

        self.face.putData(dict1[str(interest.getName())[0:44]][str(interest.getName())])
        logging.info('Serve data: {}'.format(interest.getName()))
    else:
        logging.info('Data does not exist: {}'.format(interest.getName()))

# register prefix in local NFD with it own name: /local_manager/building_1/camera_1
face = Face()
keychain = KeyChain()
face.setCommandSigningInfo(keychain, keychain.getDefaultCertificateName())
face.registerPrefix(video_stream_name, None, on_register_failed)
face.setInterestFilter(video_stream_name, on_interest)

video_capture = cv2.VideoCapture(0)
c = 0
print('Start save the video')
fourcc = cv2.VideoWriter_fourcc(*'XVID')

c=1
dict1 = {}
while True:
    curTime = int(time.time()) + 1
    out = cv2.VideoWriter(str(curTime) + '.avi', fourcc, 25, (640, 480))
    while curTime>int(time.time()):

        ret, frame = video_capture.read(0)
        # calc fps
        out.write(frame)
        cv2.imshow('Video', frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
    #     prepare the network layer packet
    dict1[video_stream_name + str(curTime)]=prepare_data(str(curTime) + '.avi')
    c+=1
    # delete the out-of-date data
    if os.path.exists(str(curTime - 10) + '.avi'):
        os.remove(str(curTime - 10) + '.avi')
        dict1.pop(video_stream_name +str(curTime - 10))

    else:
        print('file not exist or error!')
    if c==11:
        c=1


video_capture.release()
cv2.destroyAllWindows()

