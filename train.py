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
from config import Bus9
from math import *
from keras import backend as K



def train30(data, file_name, params, num_epoch=100, batch_size=256, init=None):
    model = Sequential()
    # hidden layer
    model.add(Dense(params, input_dim=data.train_data.shape[1], activation='sigmoid'))
    # output layer
    model.add(Dense(data.train_label.shape[1]))

    if init is not None:
        model.load_weights(init)

    adam = Adam(lr=0.001)

    model.compile(loss='mean_squared_error', optimizer=adam, metrics=['mse'])

    history = model.fit(data.train_data, data.train_label,
              batch_size=batch_size,
              epochs=num_epoch,
              validation_data=(data.validation_data, data.validation_label),
              shuffle=True, verbose=2)
        #print(history.history.keys())

    if file_name is not None:
        model.save(file_name)

    return history



def train9(data, bus, file_name, params, num_epoch=100, batch_size=64, init=None):
    # model = Sequential()
    # hidden layer
    # input dim only for first layer
    input = Input(shape=(data.train_data.shape[1], ))
    hidden1 = Dense(params[0], input_dim=data.train_data.shape[1], activation='relu')(input)
    hidden2 = Dense(params[1], activation='relu')(hidden1)

    # output layer
    output = Dense(data.train_label.shape[1])(hidden2)

    model = Model(input, output)

    if init is not None:
        model.load_weights(init)

    adam = Adam(lr=0.001)

    def tian_loss_wrapper(input, batch_size):
        def tian_loss(y_true, y_pred):

            x_pred = bus.estimated(y_pred, batch_size)
            loss = K.mean(K.square(input - x_pred))
            return loss
        return tian_loss



    model.compile(loss=tian_loss_wrapper(input, batch_size), optimizer=adam)

    history = model.fit(data.train_data, data.train_label,
              batch_size=batch_size,
              epochs=num_epoch,
              validation_data=(data.validation_data, data.validation_label),
              shuffle=True, verbose=2)
        #print(history.history.keys())

    if file_name is not None:
        model.save(file_name)

    return history


data = StateEstimation()
# print(data.train_data.shape)
# print(data.train_label.shape)

bus = Bus9()

# history_14 = train14(data, 'models/train_14', 80, num_epoch=100, batch_size=256)
# history_30 = train30(data, 'models/train_30', 80, num_epoch=100, batch_size=256)
history_9 = train9(data, bus, 'models/train_9', [6, 4], num_epoch=50, batch_size=1)
with tf.Session() as sess:
    model = Model9('models/train_9', sess)

# plt.plot(history_14.history['loss'])
# plt.plot(history_14.history['val_loss'])
plt.plot(history_9.history['loss'])
plt.plot(history_9.history['val_loss'])
plt.xlabel('loss')
plt.ylabel('epoch')
plt.title('model loss')
plt.legend(['train', 'val'], loc='upper right')
plt.show()

model = Model9('models/train_9')
V = np.array([1.0000, 1.0000, 1.0000, 0.9870, 0.9755, 1.0034, 0.9856, 0.9962, 0.9576]) * 10
A = np.array([0, 9.6687, 4.7711, -2.4066, -4.0173, 1.9256, 0.6215, 3.7991, -4.3499])
p_bus = np.array([71.9547, 163.0000, 85.0000, 0, -90, 0, -100, 0, -125])
q_bus = np.array([24.07, 14.46, -3.65, 0, -30, 0, -35, 0, -50])
p_line_i = np.array([71.9547, 30.7283, -59.4453, 85.0000, 24.1061, -75.9894, -163.0000, 86.5044, -40.9601])
p_line_j = np.array([-71.9547, -30.5547, 60.8939, -85.0000, -24.0106, 76.4956,163.0000, -84.0399, 41.2264])
q_line_i = np.array([24.0690, -0.5859, -16.3120, -3.6490, 4.5368, -10.5992, 2.2762, -2.5324, -35.7180])
q_line_j = np.array([-20.7530, -13.6880, -12.4275, 7.8907, -24.4008, 0.2562, 14.4601, -14.2820, 21.3389])

bus = Bus9()
# real state
x = data.test_label[:10].copy()
x_ = data.test_label[:10].copy()
z = data.test_data[:10].copy()
x_[:, 4] += 0.1
z_ = bus.estimated_np(x_)
print("###", np.sqrt(np.sum(np.square(z - z_), 1)))
# x = np.array([np.concatenate((V, A))])

# print(x)

# measurement

# z = np.array([np.concatenate((p_bus, q_bus, p_line_i, p_line_j))])
# print(z)
# NN estimate state
x_hat = model.model.predict(z)
# x_hat = model.model.predict(z)
# print(x_hat)

print("#1, diff of real state and model predict state")
print(np.sqrt(np.sum(np.square(x - x_hat), 1)))


# measurement add random value

z_tilda = data.test_data[:10].copy()
z_tilda[:, 12] += 10

print("EU diff of z_tilda and z")
print(np.sqrt(np.sum(np.square(z_tilda - z), 1, )))
# NN estimated state
x_tilda_hat = model.model.predict(z_tilda)
print("#2, EU dist of real state and model predict state from tampered measurement\n", np.sqrt(np.sum(np.square(x - x_tilda_hat), 1)))
# model predict tampered state

# meas

z_hat = bus.estimated_np(x_hat)
# print(z[0])
# print(z_hat[0])
# print("z_hat - z", (z_hat[0] - z[0]))
x1 = model.model.predict(z)
x1 = x1 * 180 / pi
x2 = model.model.predict(z_tilda)
x2 = x2 * 180 / pi
print("####", np.mean((x1 - x2), 1))
print("#3, EU dist of measurement and measurement from physical law\n", np.sqrt(np.sum(np.square(z - z_hat), 1)))

# Model test Euclidean distance

print("#4, diff of injected measurement and measurement from physical law\n", np.sqrt(np.sum(np.square(z_tilda - z_hat), 1)))

z_tilda_hat = bus.estimated_np(x_tilda_hat)

print("#5, diff of injected measurement and estimated measurement of injected measurement from physical law\n", np.sqrt(np.sum(np.square(z_tilda - z_tilda_hat), 1)))

print("z_tilda_hat - z_hat", np.sqrt(np.sum(np.square(z_hat - z_tilda_hat), 1)))
# add random noise
# print("diff of real state and model predict tampered state")
# print(0.5 * np.sum(np.square(raw_state - predict_random_add), 1))








