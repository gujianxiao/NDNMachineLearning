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
import argparse
import asyncio
import logging
from pyndn import Face, Name, Data, Interest
from pyndn.security import KeyChain
from pyndn.encoding import ProtobufTlv
from asyncndn import fetch_data_packet
from asyncndn import fetch_segmented_data
from command.repo_command_parameter_pb2 import RepoCommandParameterMessage
from command.repo_command_response_pb2 import RepoCommandResponseMessage

modeldir = './model/20170511-185253.pb'
classifier_filename = './class/classifier.pkl'
npy='./npy'
train_img="./train_img"
video_stream_name='/local_manager/building_1/camera_1/video'


async def main():

    async def face_loop():
        nonlocal face, running
        while running:
            face.processEvents()
            await asyncio.sleep(0.001)
    face = Face()
    running = True
    face_event = event_loop.create_task(face_loop())


    with tf.Graph().as_default():
        gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=0.6)
        sess = tf.Session(config=tf.ConfigProto(gpu_options=gpu_options, log_device_placement=False))
        with sess.as_default():
            pnet, rnet, onet = detect_face.create_mtcnn(sess, npy)

            minsize = 20  # minimum size of face
            threshold = [0.6, 0.7, 0.7]  # three steps's threshold
            factor = 0.709  # scale factor
            margin = 44
            frame_interval = 3
            batch_size = 1000
            image_size = 182
            input_image_size = 160

            HumanNames = os.listdir(train_img)
            HumanNames.sort()

            print('Loading Modal')
            facenet.load_model(modeldir)
            images_placeholder = tf.get_default_graph().get_tensor_by_name("input:0")
            embeddings = tf.get_default_graph().get_tensor_by_name("embeddings:0")
            phase_train_placeholder = tf.get_default_graph().get_tensor_by_name("phase_train:0")
            embedding_size = embeddings.get_shape()[1]


            classifier_filename_exp = os.path.expanduser(classifier_filename)
            with open(classifier_filename_exp, 'rb') as infile:
                (model, class_names) = pickle.load(infile,encoding='iso-8859-1')

            c = 0
            # initialized the face of NFD
            keychain = KeyChain()
            face.setCommandSigningInfo(keychain, keychain.getDefaultCertificateName())
            name = Name(video_stream_name)

            print('Start Recognition')
            prevTime = 0


            def after_fetched(data: Data):
                nonlocal recv_window, b_array, seq_to_bytes_unordered
                """
                Reassemble data packets in sequence.
                """
                if not isinstance(data, Data):
                    return
                try:
                    seq = int(str(data.getName()).split('/')[-1])
                    print('seq: {}'.format(seq))
                except ValueError:
                    logging.warning('Sequence nufetch_segmented_datamber decoding error')
                    return

                # Temporarily store out-of-order packets
                if seq <= recv_window:
                    return
                elif seq == recv_window + 1:
                    b_array.extend(data.getContent().toBytes())
                    print('saved packet: seq {}'.format(seq))
                    recv_window += 1
                    while recv_window + 1 in seq_to_bytes_unordered:
                        b_array.extend(seq_to_bytes_unordered[recv_window + 1])
                        seq_to_bytes_unordered.pop(recv_window + 1)
                        print('saved packet: seq {}'.format(recv_window + 1))
                        recv_window += 1
                else:
                    logging.info('Received out of order packet: seq {}'.format(seq))
                    seq_to_bytes_unordered[seq] = data.getContent().toBytes()


            while True:
                # fetch segment data callback

                recv_window = -1
                b_array = bytearray()
                seq_to_bytes_unordered = dict()  # Temporarily save out-of-order packets
                semaphore = asyncio.Semaphore(5)

                # to request the real-time video from the specific camera
                # construct Interest name
                name = Name(video_stream_name)
                curTime = str(int(time.time()) - 2)  # calc fps
                name.append(curTime)
                temp_file_name = str(name[-1]) + '.avi'

                print('Fetching')
                await fetch_segmented_data(face, name,
                                           start_block_id=0, end_block_id=None,
                                           semaphore=semaphore, after_fetched=after_fetched)
                print('Fetched')

                if len(b_array) == 0:
                    print('no data back')
                    continue
                else:
                    print('fetch {}    bytes of data'.format(len(b_array)))
                with open(os.path.join('result', temp_file_name), 'wb') as f:
                    f.write(b_array)


                video_capture = cv2.VideoCapture(os.path.join('result', temp_file_name))

                print('Start Recognition')
                prevTime = 0
                while True:
                    ret, frame = video_capture.read()

                    frame = cv2.resize(frame, (0, 0), fx=0.5, fy=0.5)  # resize frame (optional)

                    curTime = time.time() + 1  # calc fps
                    timeF = frame_interval

                    if (c % timeF == 0):
                        find_results = []

                        if frame.ndim == 2:
                            frame = facenet.to_rgb(frame)
                        frame = frame[:, :, 0:3]
                        bounding_boxes, _ = detect_face.detect_face(frame, minsize, pnet, rnet, onet, threshold, factor)
                        nrof_faces = bounding_boxes.shape[0]
                        print('Detected_FaceNum: %d' % nrof_faces)

                        if nrof_faces > 0:
                            det = bounding_boxes[:, 0:4]
                            img_size = np.asarray(frame.shape)[0:2]

                            cropped = []
                            scaled = []
                            scaled_reshape = []
                            bb = np.zeros((nrof_faces, 4), dtype=np.int32)

                            for i in range(nrof_faces):
                                emb_array = np.zeros((1, embedding_size))

                                bb[i][0] = det[i][0]
                                bb[i][1] = det[i][1]
                                bb[i][2] = det[i][2]
                                bb[i][3] = det[i][3]

                                # inner exception
                                if bb[i][0] <= 0 or bb[i][1] <= 0 or bb[i][2] >= len(frame[0]) or bb[i][3] >= len(frame):
                                    print('Face is very close!')
                                    continue

                                cropped.append(frame[bb[i][1]:bb[i][3], bb[i][0]:bb[i][2], :])
                                cropped[i] = facenet.flip(cropped[i], False)
                                scaled.append(misc.imresize(cropped[i], (image_size, image_size), interp='bilinear'))
                                scaled[i] = cv2.resize(scaled[i], (input_image_size, input_image_size),
                                                       interpolation=cv2.INTER_CUBIC)
                                scaled[i] = facenet.prewhiten(scaled[i])
                                scaled_reshape.append(scaled[i].reshape(-1, input_image_size, input_image_size, 3))
                                feed_dict = {images_placeholder: scaled_reshape[i], phase_train_placeholder: False}
                                emb_array[0, :] = sess.run(embeddings, feed_dict=feed_dict)
                                predictions = model.predict_proba(emb_array)
                                print(predictions)
                                best_class_indices = np.argmax(predictions, axis=1)
                                best_class_probabilities = predictions[
                                    np.arange(len(best_class_indices)), best_class_indices]
                                # print("predictions")
                                print(best_class_indices, ' with accuracy ', best_class_probabilities)

                                # print(best_class_probabilities)
                                if best_class_probabilities > 0.53:
                                    cv2.rectangle(frame, (bb[i][0], bb[i][1]), (bb[i][2], bb[i][3]), (0, 255, 0),
                                                  2)  # boxing face

                                    # plot result idx under box
                                    text_x = bb[i][0]
                                    text_y = bb[i][3] + 20
                                    print('Result Indices: ', best_class_indices[0])
                                    print(HumanNames)
                                    for H_i in HumanNames:
                                        if HumanNames[best_class_indices[0]] == H_i:
                                            result_names = HumanNames[best_class_indices[0]]
                                            cv2.putText(frame, result_names, (text_x, text_y),
                                                        cv2.FONT_HERSHEY_COMPLEX_SMALL,
                                                        1, (0, 0, 255), thickness=1, lineType=2)
                        else:
                            print('Alignment Failure')
                    # c+=1
                    cv2.imshow('Video_2', frame)

                    if cv2.waitKey(1) & 0xFF == ord('q'):
                        break

                video_capture.release()
                cv2.destroyAllWindows()

    running = False
    await face_event


if __name__ == '__main__':
    logging.basicConfig(format='[%(asctime)s]%(levelname)s:%(message)s',
                        datefmt='%Y-%m-%d %H:%M:%S',
                        level=logging.INFO)

    event_loop = asyncio.get_event_loop()
    event_loop.run_until_complete(main())

