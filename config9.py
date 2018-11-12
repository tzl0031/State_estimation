from model import StateEstimation
import tensorflow as tf
import numpy as np
from keras.models import Sequential, Model
from keras.layers import Dense, Activation, Input
from keras.optimizers import SGD, Adam
from keras.wrappers.scikit_learn import KerasRegressor
from sklearn.model_selection import cross_val_score, KFold
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
import matplotlib.pyplot as plt
from model import *

from math import *
from keras import backend as K

import pypsa
import numpy as np
from pypower.api import case9, case14, case30
from math import *
import pandas as pd
import tensorflow as tf
from math import *
import keras.backend as K


df_meas = pd.read_csv('data/measurement_9bus_train.csv', header=None)

max_meas = K.constant(np.max(df_meas.values, axis=0))
min_meas = K.constant(np.min(df_meas.values, axis=0))
max_meas_np = np.max(df_meas.values, axis=0)
min_meas_np = np.min(df_meas.values, axis=0)



df_state = pd.read_csv('data/state_9bus_train.csv', header=None)
max_state = K.constant(np.max(df_state.values, axis=0))
min_state = K.constant(np.min(df_state.values, axis=0))
max_state_np = np.max(df_state.values, axis=0)
min_state_np = np.min(df_state.values, axis=0)



class Bus9:
    def __init__(self):
        G = pd.read_csv('bus_config/Ybus_9_real', header=None)
        B = pd.read_csv('bus_config/Ybus_9_imag', header=None)
        self.G = G.values
        self.B = B.values
        self.num_bus = 9
        self.num_lines = 9
        # self.network = pypsa.Network()
        # ppc = case9()
        # self.network.import_from_pypower_ppc(ppc)
        # self.network.pf()
        # self.bus_num = np.array(self.network.lines[['bus0', 'bus1']].values)

    def gather_cols(self, params, indices):
        p_shape = params.shape
        p_flat = K.reshape(params, [-1])
        i_flat = K.reshape(K.reshape(K.arange(0, 1) * p_shape[1], [-1, 1]) + indices, [-1])
        result = K.reshape(tf.gather(p_flat, i_flat), [1, -1])
        # print(result.shape)
        return result

    def estimated_np(self, state):
        #  inject generator vol mags and angle for ref bus.

        # state = np.insert(state, 6, 0, axis=1)
        batch_size = state.shape[0]

        # generator_mag = np.ones((batch_size, 9))
        # state = np.concatenate((generator_mag, state), axis=1)
        P_bus = np.zeros((batch_size, self.num_bus))
        Q_bus = np.zeros((batch_size, self.num_bus))
        # P_line = np.zeros((batch_size, self.num_bus, self.num_bus))
        # Q_line = np.zeros((batch_size, self.num_bus, self.num_bus))

        # batch_estimated_measurement = np.zeros((batch_size, 54))
        state_restore = (state + 1) * (max_state_np - min_state_np) / 2 + min_state_np

        # physical laws
        V = state_restore[:, :self.num_bus] * 10
        A = state_restore[:, self.num_bus:]

        # estimated_measurement = np.zeros(36)
        G = self.G
        B = self.B

        # [k, 9]
        # print(V.shape, A.shape)
        # P_bus = K.zeros((A.shape[0], 9))
        # [k, 9]
        # Q_bus = K.zeros((A.shape[0], 9))
        # [k, 9]
        # A_
        # print(A.shape)
        A = A.reshape(batch_size, self.num_bus, 1)
        # print(np.tile(A, (1, 1, self.num_bus)).shape)
        # print(np.transpose(np.tile(A, (1, 1, self.num_bus)), (0, 2, 1)).shape)
        A_ = np.transpose(np.tile(A, (1, 1, self.num_bus)), (0, 2, 1)) - np.tile(A, (1, 1, self.num_bus))
        cos_ = np.cos(A_ * pi / 180)
        sin_ = np.sin(A_ * pi / 180)

        term_1_P = G * cos_ + B * sin_
        term_1_Q = G * sin_ - B * cos_
        for i in range(batch_size):
            P_bus[i] = np.dot(V[i], term_1_P[i])
            Q_bus[i] = np.dot(V[i], term_1_Q[i])

        P_bus = (V * P_bus)
        Q_bus = (V * Q_bus)
        # print(P_bus, Q_bus)

        # # P0, P1 = self.line_matrix_converter_np(P_line, batch_size)
        # Q0, Q1 = self.line_matrix_converter_np(Q_line, batch_size)
        # # col = [4, 6, 8]
        batch_estimated_measurement = np.concatenate([P_bus[:, [4, 6, 8]], Q_bus[:, [4, 6, 8]]], axis=1)

        ans = (batch_estimated_measurement - min_meas_np) / (max_meas_np - min_meas_np) * 2 - 1

        return ans

        # return scaler_meas.fit_transform(batch_estimated_measurement)

    def line_matrix_converter_np(self, line, batch_size):

        P0 = np.zeros((batch_size, self.num_bus))
        P1 = np.zeros((batch_size, self.num_bus))
        # print(batch_size, self.num_bus)
        for k in range(batch_size):
            for i in range(self.num_bus):
                x = int(self.bus_num[i, 0]) - 1
                y = int(self.bus_num[i, 1]) - 1
                # print(x, y)
                P0[k, i] = line[k, x, y]
                P1[k, i] = line[k, y, x]
        # print(P0, P1)

        return P0, P1

    def estimated(self, state, batch_size):
        # print(state.shape)

        # batch_size = state.shape[0]

        # generator_mag = K.ones((batch_size, 3))
        # ang_ref = K.zeros((batch_size, 1))
        # ref_ang = tf.Variable(tf.zeros((batch_size, 1)))
        # state = K.concatenate([generator_mag, state[:, :6], ang_ref, state[:, 6:]], axis=-1)

        # print(state.shape)
        state_restore = (state + 1) * (max_state - min_state) / 2 + min_state


        V = state_restore[:, :self.num_bus] * 10
        # [k, 9]

        A = state_restore[:, self.num_bus:]
        # [k, 9]
        # print(V.shape, A.shape)
        # P_bus = K.zeros((A.shape[0], 9))
        # [k, 9]
        # Q_bus = K.zeros((A.shape[0], 9))
        # [k, 9]
        # A_
        # print(K.permute_dimensions(K.repeat(A, 9), [0, 2, 1]).shape)
        # print(K.repeat(A, 9).shape)
        A_ = K.permute_dimensions(K.repeat(A, self.num_bus), [0, 2, 1]) - K.repeat(A, self.num_bus)
        G = K.constant(self.G, dtype=tf.float32)
        B = K.constant(self.B, dtype=tf.float32)
        cos_ = K.cos(A_ * pi / 180)
        sin_ = K.sin(A_ * pi / 180)

        term_1_P = G * cos_ + B * sin_
        term_1_Q = G * sin_ - B * cos_
        P_bus = (V * K.batch_dot(V, term_1_P, axes=[1, 2]))
        Q_bus = (V * K.batch_dot(V, term_1_Q, axes=[1, 2]))

        idx = [4, 6, 8]


        batch_estimated_measurement = K.concatenate([self.gather_cols(P_bus, idx), self.gather_cols(Q_bus, idx)], axis=1)
        # print(batch_estimated_measurement.shape)

        ans = (batch_estimated_measurement - min_meas) / (max_meas - min_meas) * 2 - 1
        # print(K.eval(ans))

        return ans



data = StateEstimation()

bus = Bus9()

BATCH_SIZE = 50

def train9(data, bus, file_name, params, num_epoch=100, batch_size=64, init=None):


    def custom_loss_wrapper(input_tensor):
        def custom_loss(y_true, y_pred):
            x_pred = bus.estimated(y_pred, BATCH_SIZE)
            return K.mean(K.square(input_tensor - x_pred))
        return custom_loss


    adam = Adam(lr=0.01)
    input_tensor = Input(shape=(6, ))
    hidden = Dense(params[0], activation='sigmoid')(input_tensor)
    hidden2 = Dense(params[1], activation='sigmoid')(hidden)
    out = Dense(18)(hidden2)
    model = Model(input_tensor, out)
    model.compile(loss='mse', optimizer=adam)
    # custom_loss_wrapper(input_tensor)#

    # X = data.train_data[:100]
    # y = data.train_label[:100]
    # print(model.test_on_batch(X, y))
    history = model.fit(data.train_data, data.train_label,
              batch_size=batch_size,
              epochs=num_epoch,
              validation_data=(data.validation_data, data.validation_label),
              shuffle=True, verbose=2)
    if file_name is not None:
        model.save(file_name)
    return history


# history_9 = train9(data, bus, 'models/train_9', [16, 40], num_epoch=100, batch_size=BATCH_SIZE)
# #
# with tf.Session() as sess:
#     model = Model9('models/train_9', sess)
#
# plt.plot(history_9.history['loss'])
# plt.plot(history_9.history['val_loss'])
# plt.xlabel('loss')
# plt.ylabel('epoch')
# plt.title('model loss')
# plt.legend(['train', 'val'], loc='upper right')
# plt.show()
#
model = Model9('models/train_9')

def restore_state(state):
    return (state + 1) * (max_state_np - min_state_np) / 2 + min_state_np

def restore_meas(meas):
    return (meas + 1) * (max_meas_np - min_meas_np) / 2 + min_meas_np

def scale_meas(meas):
    return (meas - min_meas_np) / (max_meas_np - min_meas_np) * 2 -1


bus = Bus9()
# real state
x = data.test_label[:10].copy()
x_ = data.test_label[:10].copy()
z = data.test_data[:10].copy()




# print("### add 0.1 to any angle will result in how much diff in meas ", np.sqrt(np.sum(np.square(z - z_), 1)))
# x = np.array([np.concatenate((V, A))])

# print("state: ", x)

# measurement

# z = np.array([np.concatenate((p_bus, q_bus, p_line_i, p_line_j))])
# print("measurement: ", z)
# NN estimate state
x_hat = model.model.predict(z)
# x_hat = model.model.predict(z)
# print("estimated state: ", x_hat)
z_hat = bus.estimated_np(x_hat)
# print("estimated measurement: ", z_hat)

print("Model performance")
print("x_hat - x, mse", np.mean(np.square(x_hat - x), 1))

print("z_hat - z, mse", np.mean(np.square(z_hat - z), 1))
print("In original scale")

print("x_hat - x, mse", np.mean(np.square(restore_state(x_hat) - restore_state(x)), 1))

print("z_hat - z, mse", np.mean(np.square(restore_meas(z_hat) - restore_meas(z)), 1))




# measurement add random value

z_tilda = data.test_data[:10].copy()
z_tilda[:, 5] += 0.1

print("EU diff of z_tilda and z")
print(np.sqrt(np.mean(np.square(z_tilda - z), 1)))
# NN estimated state
x_tilda_hat = model.model.predict(z_tilda)
print("#2, add 0.1 to measurement, EU dist of real state and model predict state from tampered measurement z - z_tilda_hat\n", np.sqrt(np.sum(np.square(x - x_tilda_hat), 1)))
# model predict tampered state

# meas


# print(z[0])
# print(z_hat[0])
# print("z_hat - z", (z_hat[0] - z[0]))
x1 = model.model.predict(z)
x1 = x1[9:] * 180 / pi
x2 = model.model.predict(z_tilda)
x2 = x2[9:] * 180 / pi
print("#### mean abs diff of x and x_tilda_hat", np.mean(abs(x1 - x2), 1))
print("#3,  EU dist of measurement and measurement from physical law, z- z_hat\n", np.sqrt(np.mean(np.square(z - z_hat), 1)))

# Model test Euclidean distance

print("#4, EU diff of injected measurement and measurement from physical law z_tilda - z_hat\n", np.sqrt(np.mean(np.square(z_tilda - z_hat), 1)))

z_tilda_hat = bus.estimated_np(x_tilda_hat)

print("#5, EU diff of injected measurement and estimated measurement of injected measurement from physical law z_tilda - z_tilda_hats\n", np.sqrt(np.sum(np.square(z_tilda - z_tilda_hat), 1)))

print("z_tilda_hat - z_hat", np.sqrt(np.mean(np.square(z_hat - z_tilda_hat), 1)))
# add random noise
# print("diff of real state and model predict tampered state")
# print(0.5 * np.sum(np.square(raw_state - predict_random_add), 1))














