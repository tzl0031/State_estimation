# import tensorflow as tf
from keras.models import Sequential
import pandas as pd
from keras.layers import Dense, Dropout, Activation, Flatten
from sklearn.preprocessing import StandardScaler


DATA_TEST = 'measurements_test'
LABEL_TEST = 'state_test'
DATA_TRAIN = 'measurements'
LABEL_TRAIN = 'state'


def load_data(test=1, case_num=14):
    data_filename = DATA_TEST + '_' + str(case_num) if test else DATA_TRAIN + '_' + str(case_num)

    data = pd.read_csv(data_filename, header=None)
    # data transform
    data = StandardScaler().fit_transform(data)

    return data


def load_label(test=1, case_num=14):
    label_filename = LABEL_TEST+'_'+str(case_num) if test else LABEL_TRAIN+'_'+str(case_num)
    label = pd.read_csv(label_filename, header=None)
    # label transform
    label = StandardScaler().fit_transform(label)

    return label


class StateEstimation:
    def __init__(self):
        train_data = load_data(0)
        train_label = load_label(0)
        self.test_data = load_data()
        self.test_label = load_label()

        VALIDATION_SIZE = 5000

        self.validation_data = train_data[:VALIDATION_SIZE, :]
        self.validation_label = train_label[:VALIDATION_SIZE, :]
        self.train_data = train_data[VALIDATION_SIZE:, :]
        self.train_label = train_label[VALIDATION_SIZE:, :]


class Model:
    def __init__(self, restore, session=None):
        self.input_size = 21
        self.output_size = 17

        model = Sequential()
        model.add(Dense(20, input_dim=self.input_size))
        model.add(Activation('relu'))
        model.add(Dense(self.output_size))
        model.load_weights(restore)

        self.model = model

    def predict(self, data):
        return self.model(data)






