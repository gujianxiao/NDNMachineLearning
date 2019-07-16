from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import tensorflow as tf
from scipy import misc
import cv2
import numpy as np
import facenet
import detect_face
import os
import time
import pickle


print('Start capture')


video_capture = cv2.VideoCapture(0)
c = 0


print('Start save the video')

fourcc = cv2.VideoWriter_fourcc(*'XVID')

c=1
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
    c+=1
    if os.path.exists(str(curTime - 10) + '.avi'):
        print('ready to delete ' + str(curTime - 10) + '.avi')
        os.remove(str(curTime - 10) + '.avi')
    else:
        print('file not exist or error!')
    if c==11:
        c=1


video_capture.release()
cv2.destroyAllWindows()
