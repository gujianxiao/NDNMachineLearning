"""
Incremental learning on temperature data
"""

import os
import tensorflow as tf
import numpy as np
import pandas as pd
import tflearn
import logging
from datetime import datetime
from excel import data_prepare,preprocess_excel_data


class IncrementalLearner(object):

    def __init__(self):
        self.model_path = os.path.join('model', 'popModel')
    
    def load_data(self, data_path: str):
        tf.reset_default_graph()
        preprocess_excel_data(data_path)
        train_data, train_label = data_prepare(type='train')
        self.train_data = train_data
        self.train_label = train_label
        print(self.train_data.shape)
        print(self.train_label.shape)
    
    def train_once(self):
        net = tflearn.input_data(shape=[None, 24, 38])
        net = tflearn.lstm(net, 128, return_seq=True)
        net = tflearn.lstm(net, 128)
        net = tflearn.fully_connected(net, 1, activation='linear')
        net = tflearn.regression(net, optimizer='adam',batch_size=0.01,learning_rate=0.0005,
                                 loss='mean_square', name="output1")
        
        model = tflearn.DNN(net, tensorboard_dir='./log/',tensorboard_verbose=2)
        
        if os.path.exists(self.model_path + '.meta'):
            model.load(self.model_path)
            logging.info('Continue training on existing model ...')

        model.fit(self.train_data, self.train_label, n_epoch=10, validation_set=0.0,
                  show_metric=True, shuffle=True, batch_size=min(72, len(self.train_data)),
                  run_id='popularity-1')
        
        model.save(self.model_path)


if __name__ == "__main__":
    learner = IncrementalLearner()
    # learner.load_data(os.path.join('data', 'NY-Data-Preprocess-to-03-02.csv'))
    
    learner.load_data(os.path.join('data', '1563757821.csv'))
    learner.train_once()

    learner.load_data(os.path.join('data', '1563757836.csv'))
    learner.train_once()


