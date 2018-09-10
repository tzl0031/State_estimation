# import tensorflow as tf
from keras.models import Sequential
import pandas as pd
import numpy as np
from keras.layers import Dense, Dropout, Activation, Flatten
from sklearn.preprocessing import MinMaxScaler


DATA_TEST = 'measurement_test'
LABEL_TEST = 'state_test'
DATA_TRAIN = 'measurement'
LABEL_TRAIN = 'state'

# case 14. bus=14, line = 17, gen=5, load=11
# case 30，bus=30, line = 41, gen=
# case 118, bus=118, line = 177, gen=9, synchronous_condenser=35, transformer=9, loads=91

def load_data(test=1, case_num=9):
    data_filename = 'data/' + DATA_TEST + '_' + str(case_num) if test else 'data/' + DATA_TRAIN + '_' + str(case_num)
    df = pd.read_csv('data/measurement_9bus_test.csv', header=None)
    # data transform
    # data = np.concatenate([p_bus, p0, p1])
    # drop generator vol and ref angle


    # data = MinMaxScaler().fit_transform(df) - 0.5
    df.drop(df.columns[18:], axis=1, inplace=True)

    df = df.values
    # min = df.min(axis=0)
    # max = df.max(axis=0)
    # df = MinMaxScaler().fit_transform(df) - 0.5

    noise = np.random.normal(0, 1, size=df.shape)
    data = df + noise
    print(data.shape)
    # print(min, max)
    return data


def load_label(test=1, case_num=9):
    label_filename = 'data/' + LABEL_TEST+'_'+str(case_num) if test else 'data/' + LABEL_TRAIN+'_'+str(case_num)
    df = pd.read_csv("data/state_9bus_test.csv", header=None)
    # cols = [0, 1, 2, 9]
    df.drop(df.columns[:10], axis=1, inplace=True)
    print(df.shape)

    # label transform

    return df.values


class StateEstimation:
    def __init__(self):
        train_data = load_data(0)
        train_label = load_label(0)
        self.test_data = load_data()
        self.test_label = load_label()

        VALIDATION_SIZE = 100
        self.validation_data = train_data[:VALIDATION_SIZE, :]
        self.validation_label = train_label[:VALIDATION_SIZE, :]
        self.train_data = train_data[VALIDATION_SIZE:, :]
        self.train_label = train_label[VALIDATION_SIZE:, :]

class Model9:
    def __init__(self, restore, session=None):
        self.input_size = 18
        self.output_size = 8

        model = Sequential()
        model.add(Dense(6, input_dim=self.input_size))
        model.add(Activation('relu'))
        model.add(Dense(4))
        model.add(Activation('relu'))
        model.add(Dense(self.output_size))
        model.load_weights(restore)

        self.model = model

    def predict(self, data):
        return self.model(data)


class Model14:
    def __init__(self, restore, session=None):
        self.input_size = 48
        self.output_size = 28

        model = Sequential()
        model.add(Dense(80, input_dim=self.input_size))
        model.add(Activation('sigmoid'))
        model.add(Dense(self.output_size))
        model.load_weights(restore)

        self.model = model

    def predict(self, data):
        return self.model(data)


class Model30:
    def __init__(self, restore, session=None):
        self.input_size = 112
        self.output_size = 60

        model = Sequential()
        model.add(Dense(80, input_dim=self.input_size))
        model.add(Activation('sigmoid'))
        model.add(Dense(self.output_size))
        model.load_weights(restore)

        self.model = model

    def predict(self, data):
        return self.model(data)








