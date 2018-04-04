from model import StateEstimation
# import tensorflow as tf
import numpy as np
from keras.models import Sequential
from keras.layers import Dense, Activation
from keras.optimizers import SGD, Adam
from keras.wrappers.scikit_learn import KerasRegressor
from sklearn.model_selection import cross_val_score, KFold
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
import matplotlib.pyplot as plt


def train(data, file_name, params, num_epoch=100, batch_size=256, init=None):
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


data = StateEstimation()
print(data.train_data.shape)
print(data.train_label.shape)


history
history_14 = train(data, 'models/train_14', 80, num_epoch=200, batch_size=256)
history_9 = train(data, 'models/train_9', 18, num_epoch=50, batch_size=128)

plt.plot(history14.history['loss'])
plt.plot(history.history['val_loss'])
plt.xlabel('loss')
plt.ylabel('epoch')
plt.title('model loss')
plt.legend(['train', 'test'], loc='upper right')
plt.show()
# loss, mse = model.evaluate(StateEstimation().test_data, StateEstimation().test_label)
# print("Results on test set: %.2f (%.2f) MSE" % (loss, mse))



