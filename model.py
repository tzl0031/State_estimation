# import tensorflow as tf
from keras.models import Sequential
import pandas as pd
import numpy as np
from keras.layers import Dense, Dropout, Activation, Flatten
from sklearn.preprocessing import MinMaxScaler
import keras.backend as K



# case 14. bus=14, line = 17, gen=5, load=11
# case 30，bus=30, line = 41, gen=
# case 118, bus=118, line = 177, gen=9, synchronous_condenser=35, transformer=9, loads=91

def load_data(test=1, case_num=9):
    file_name = 'data/measurement_' + str(case_num)+ "bus_test.csv" if test else 'data/measurement_' + str(case_num)+"bus_train.csv"
    df = pd.read_csv(file_name, header=None)
    noise = np.random.normal(0, 0.1, size=df.shape)
    df = df + noise
    scaler = MinMaxScaler(feature_range=(-1, 1))
    meas = scaler.fit_transform(df)
    # data transform
    # data = np.concatenate([p_bus, p0, p1])
    # drop generator vol and ref angle
    # meas = np.clip(meas + noise, -1, 1)
    return meas


def load_label(test=1, case_num=9):
    file_name = 'data/state_' + str(case_num) + "bus_test.csv" if test else 'data/state_'+ str(case_num)+"bus_train.csv"
    df = pd.read_csv(file_name, header=None)
    # cols = [0, 1, 2, 9]
    # df.drop(df.columns[:10], axis=1, inplace=True)
    scaler = MinMaxScaler(feature_range=(-1, 1))
    state = scaler.fit_transform(df)
    # print(df.shape)

    # label transform

    return state


class StateEstimation:
    def __init__(self):
        train_data = load_data(0)
        train_label = load_label(0)
        self.test_data = load_data()
        self.test_label = load_label()

        VALIDATION_SIZE = 1000
        self.validation_data = train_data[:VALIDATION_SIZE, :]
        self.validation_label = train_label[:VALIDATION_SIZE, :]
        self.train_data = train_data[VALIDATION_SIZE:, :]
        self.train_label = train_label[VALIDATION_SIZE:, :]

class Model9:
    def __init__(self, restore, session=None):
        self.input_size = 6
        self.output_size = 18

        model = Sequential()
        model.add(Dense(12, input_dim=self.input_size))
        model.add(Activation('sigmoid'))
        model.add(Dense(24))
        model.add(Activation('sigmoid'))
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








