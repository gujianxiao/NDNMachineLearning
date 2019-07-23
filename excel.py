#coding=utf-8
import random
import os
import numpy as np
import pandas as pd
import tensorflow as tf
import pickle
from sklearn.externals import joblib
from sklearn.preprocessing import *
from datetime import datetime

def preprocess_excel_data(data_name='./data/NY-Data-Preprocess-to-03-02.csv'):
    data = pd.read_csv(data_name)
    data = data.fillna(0)

    keys = data.sort_values(['Time']).groupby(['DistrictCode', 'TypeCode']).groups


    # print(data)
    num_rows = len(data)
    Hour = np.zeros(shape=[num_rows], dtype=np.int32)
    Day = np.zeros(shape=[num_rows], dtype=np.int32)
    Month = np.zeros(shape=[num_rows], dtype=np.int32)
    Label = np.zeros(shape=[num_rows], dtype=np.int32)
    District = np.zeros(shape=[num_rows], dtype=np.int32)
    Type = np.zeros(shape=[num_rows], dtype=np.int32)
    # Popularity = np.zeros(shape=[num_rows], dtype=np.int32)

    idx=0
    for key, values in keys.items():
        if key != 'null':
            n=len(values)
            for i in range(n):
                value = values[i]
                # print(value)
                t = datetime.strptime(data.ix[value, 'Time'].strip(), '%Y-%m-%d-%H')


                Day[idx] = t.day
                Hour[idx] = t.hour
                Month[idx] = t.month

                District[idx] = data.ix[value,'DistrictCode']
                Type[idx] = data.ix[value,'TypeCode']

                try:
                    Label[idx] = data.ix[value,'Popularity']
                except ValueError:
                    Label[idx] = 0
                # if i == 0:
                #     value = values[-1]
                #     # print(value)
                #     try:
                #         Popularity[idx] = data.ix[value, 'Popularity']
                #     except ValueError:
                #         Popularity[idx] = 0
                # else:
                #     value = values[i-1]
                #     try:
                #         Popularity[idx] = data.ix[value, 'Popularity']
                #     except ValueError:
                #         Popularity[idx] = 0

                idx += 1



    district_embeddings = tf.get_variable(
        name='district_embeddings',
        shape=[6, 3],  # 3, 2, 5
        dtype=tf.float32
    )
    type_embeddings = tf.get_variable(
        name='type_embeddings',
        shape=[6, 3],  # 3 2 5
        dtype=tf.float32
    )
    month_embeddings = tf.get_variable(
        name='month_embeddings',
        shape=[13, 6],  # 6 3 10

        dtype=tf.float32
    )
    day_embeddings = tf.get_variable(
        name='day_embeddings',
        shape=[32, 16],  # 16 8 24
        dtype=tf.float32
    )
    hour_embeddings = tf.get_variable(
        name='hour_embeddings',
        shape=[25, 10],  # 10 6 18
        dtype=tf.float32
    )

    # print(Day)
    # print(Month)
    # print(Hour)
    # print(District)
    # print(Label)
    # print(Popularity)

    ndata = tf.concat([
        tf.nn.embedding_lookup(day_embeddings, Day),
        tf.nn.embedding_lookup(month_embeddings, Month),
        tf.nn.embedding_lookup(hour_embeddings, Hour),
        tf.nn.embedding_lookup(district_embeddings, District),
        tf.nn.embedding_lookup(type_embeddings, Type),
        # Popularity.reshape(len(Popularity), 1),
        Month.reshape(len(Month), 1),
        Day.reshape(len(Day), 1)
    ], axis=1)
    # print(ndata)

    init_op = tf.global_variables_initializer()
    saver = tf.train.Saver()
    with tf.Session() as sess:

        sess.run(init_op)
        save_path = saver.save(sess, "./Variable/model.ckpt")
        print("Model saved in file: ", save_path)
        # saver.restore(sess, "/Variable/model.ckpt")
        ndata = ndata.eval(session=sess)
        # print("****" * 20)
        # print(ndata)
        # print(len(Label))
        Label = np.array(Label)
        Label = Label.reshape(len(Label), 1)
        data = np.save('./feat-data/origin_data_all_popularity.npy', np.asarray(ndata))
        labels = np.save('./feat-data/labels_all_popularity.npy', np.asarray(Label))
        # print(np.asarray(ndata).shape)
        # print(np.asarray(Label).shape)


def data_prepare(type=None):
    data = np.load('./feat-data/origin_data_all_popularity.npy').astype('float')
    labels = np.load('./feat-data/labels_all_popularity.npy').astype('float')
    '''
    pm10_model_predict_all_labels = np.load('../data/pm10_model_predict_all_labels.npy')
    combined_data = []
    for ori_data,pm10_data in zip(data[-len(pm10_model_predict_all_labels):],pm10_model_predict_all_labels):
        combined_data.append(np.append(ori_data,pm10_data))
    data = np.asarray(combined_data)
    labels = labels[-len(pm10_model_predict_all_labels):]
    '''
    ori_data = data
    #
    # print(data.shape, ori_data.shape)
    # print(data, data.shape)
    rnn_data = []
    rnn_labels = []
    for i in range(len(labels) - 24):
        rnn_data.append(data[i:i + 24])
        rnn_labels.append(labels[i:i + 24])

    rnn_data = np.asarray(rnn_data)
    rnn_labels = np.asarray(rnn_labels)
    if type == 'train':

        data = []
        labels = []
        for sample, label in zip(rnn_data, rnn_labels):

            if sample[0, -2] != 12. or sample[0, -1] != 8.:
                data.append(sample[:, :-2])
                # label[-1]=to_cls_label(float(label[-1]))
                labels.append(label[-1])
                print(sample[0, -1])
#                labels.append(to_cls_label(float(label[-1])))
        data = np.array(data)
        labels = np.array(labels)
    elif type == 'test':

        data = []
        labels = []
        for sample, label in zip(rnn_data, rnn_labels):
            # print('print sample[0,-1]')
            # print(sample[0, -1])
            # print('print sample over')
            if  sample[0, -1] == 8.and sample[0, -2] == 12.:
                data.append(sample[:, :-2])
                # label[-1]=to_cls_label(float(label[-1]))
                labels.append(label[-1])

                # print(sample[:, :])
                # print("***"*20)
                # print(sample[:, :-1])
                # print("----" * 20)

#                labels.append(to_cls_label(float(label[-1])))
        data = np.array(data)
        labels = np.array(labels)
    elif type == 'all':
        data = []
        labels = []
        for sample, label in zip(rnn_data, rnn_labels):
            # print sample[:,-1]
            data.append(sample[:, :-2])
            labels.append(label[-1])

        data = np.array(data)
        labels = np.array(labels)

    else:
        return None

    # print(data.shape, labels.shape)
    # print(labels)
    return data, labels




def preprocess_data():
    data = pd.read_csv("./data/NY-Data-Preprocess-to-03-02.csv")
    data = data.fillna(0)
    data = data.sort_values(['Time', 'DistrictCode', 'TypeCode'])
    num_rows = len(data)
    Hour = np.zeros(shape=[num_rows], dtype=np.int32)
    Day = np.zeros(shape=[num_rows], dtype=np.int32)
    Month = np.zeros(shape=[num_rows], dtype=np.int32)
    Label = np.zeros(shape=[num_rows], dtype=np.int32)
    District = np.zeros(shape=[num_rows], dtype=np.int32)
    Type = np.zeros(shape=[num_rows], dtype=np.int32)
    Keys = np.zeros(shape=[num_rows,3], dtype=np.int32)
    for idx, row in data.iterrows():
        time = datetime.strptime(row['Time'].strip(), '%Y-%m-%d-%H')
        Day[idx] = time.day
        Hour[idx] = time.hour
        Month[idx] = time.month

        District[idx] = row['DistrictCode']
        Type[idx] = row['TypeCode']
        Keys[idx] = [time.hour,row['DistrictCode'],row['TypeCode']]
        try:
            Label[idx] = row['Popularity']
        except ValueError:
            Label[idx] = 0

    district_embeddings = tf.get_variable(
        name='district_embeddings',
        shape=[6, 3],  # 3, 2 ，5
        dtype=tf.float32
    )
    type_embeddings = tf.get_variable(
        name='type_embeddings',
        shape=[6, 3],  # 3 2 5
        dtype=tf.float32
    )
    month_embeddings = tf.get_variable(
        name='month_embeddings',
        shape=[13, 6],  # 6 3 10

        dtype=tf.float32
    )
    day_embeddings = tf.get_variable(
        name='day_embeddings',
        shape=[32, 16],  # 16 8 24
        dtype=tf.float32
    )
    hour_embeddings = tf.get_variable(
        name='hour_embeddings',
        shape=[25, 10],  # 10 6 18
        dtype=tf.float32
    )

    # print(Day)
    # print(Month)
    # print(Hour)
    # print(District)

    ndata = tf.concat([
        tf.nn.embedding_lookup(day_embeddings, Day),
        tf.nn.embedding_lookup(month_embeddings, Month),
        tf.nn.embedding_lookup(hour_embeddings, Hour),
        tf.nn.embedding_lookup(district_embeddings, District),
        tf.nn.embedding_lookup(type_embeddings, Type),
        Day.reshape(len(Day), 1)
    ], axis=1)
    # print(ndata)

    init_op = tf.global_variables_initializer()
    saver = tf.train.Saver()
    with tf.Session() as sess:
        sess.run(init_op)
        # save_path = saver.save(sess, "./Variable/model.ckpt")
        # print("Model saved in file: ", save_path)
        # saver.restore(sess, "./Variable/model.ckpt")

        ndata = ndata.eval(session=sess)
        # print("****" * 20)
        # print(ndata)
        # print(len(Label))

        rnn_data = []
        rnn_labels = []
        rnn_keys = []

        for i in range(len(Label) - 24):
            if Day[i]==28 and Month[i]==11:
                rnn_data.append(ndata[i-23:i+1])
                rnn_labels.append(Label[i-23:i+1])
                rnn_keys.append(Keys[i-23:i+1])

        rnn_data = np.asarray(rnn_data)
        rnn_labels = np.asarray(rnn_labels)
        featureToEmbed = {}
        labelToEmbed = {}


        for sample, label , key in zip(rnn_data, rnn_labels, rnn_keys):
            keys=tuple(key[-1])
            value1=[list(sample[:, :-1])]
            value2=label[-1]

            featureToEmbed[keys]=value1
            labelToEmbed[keys]=value2
        # print(featureToEmbed)
        # print(labelToEmbed)
        # a=[17,8,1]
        # a=tuple(a)
        # print(featureToEmbed[a])
        # print(labelToEmbed[a])

        # np.save('./feat-data/dict_features.npy', featureToEmbed)
        # np.save('./feat-data/dict_labels.npy', labelToEmbed)
        # print(len(featureToEmbed))
        # print(len(labelToEmbed))
        return featureToEmbed,labelToEmbed

def oneDayData(file):
    # features,labels=preprocess_data()
    path = 'd:/Poisson generate data/%d/NY Data Generate.csv' % file
    data = pd.read_csv(path)
    # data = data.sort_values(['Time', 'ArriveTime'])
    num_rows = len(data)
    # Hour = np.zeros(shape=[num_rows], dtype=np.int32)
    # Day = np.zeros(shape=[num_rows], dtype=np.int32)
    # Month = np.zeros(shape=[num_rows], dtype=np.int32)
    # Label = np.zeros(shape=[num_rows], dtype=np.int32)
    # District = np.zeros(shape=[num_rows], dtype=np.int32)
    # Type = np.zeros(shape=[num_rows], dtype=np.int32)
    # Keys = np.zeros(shape=[num_rows, 3], dtype=np.int32)
    Keys = []
    for idx, row in data.iterrows():
        time = datetime.strptime(row['Time'].strip(), '%Y-%m-%d-%H')
        # Day[idx] = time.day
        # Hour[idx] = time.hour
        # Month[idx] = time.month
        #
        # District[idx] = row['DistrictCode']
        # Type[idx] = row['TypeCode']
        Keys.append([time.hour, row['DistrictCode'], row['TypeCode']])
        # Label[idx] = row['Popularity']
    # print(Keys)
    return Keys





if __name__ == '__main__':
    preprocess_excel_data()
   # preprocess_data()
   # data_prepare (type = 'test')

   # pass


