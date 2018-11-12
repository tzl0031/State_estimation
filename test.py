from pypower.api import case9, case14
import pypsa
import pandas as pd
import numpy as np
from math import *
import tensorflow as tf
from scipy import sparse
import keras.backend as K

num_bus = 30

# 9
# v = np.array([1, 1, 1, 0.9872, 0.9772, 1.0037, 0.9859, 0.9963, 0.9576]) * 10
# a = np.array([0, 10.212, 5.4522, -2.0686, -3.199, 2.6077, 1.2241, 4.3428, -3.9416])
# p_bus = np.array([61.8641, 163, 85, 0, -80, 0, -100, 0, -125])
# q_bus = np.array([23.3778, 14.3328, -4.2564, 0, -30, 0, -35, 0, -50])

# 14
# v = np.array([1.060, 1.045, 1.010, 1.0177, 1.0195, 1.0700, 1.0615, 1.090, 1.0559, 1.051, 1.0569, 1.0552, 1.0504, 1.0355]) * 10
# a = np.array(
#     [0, -4.9826, -12.7251, -10.3129, -8.7739, -14.2209, -13.3596, -13.3596, -14.9385, -15.0973, -14.7906, -15.0756, -15.1563,
#      -16.0336])
# p_bus = np.array(
#     [232.3933, 40 - 21.7000, -94.2000, -47.8000, -7.6000, -11.2000, 0, 0, -29.5000, -9.0000, -3.5000, -6.1000, -13.5000,
#      -14.9000])
# q_bus = np.array(
#     [-16.5493, 43.5571 - 12.7000, 25.0753 - 19.0000, +3.9000, -1.6000, 12.7309 - 7.5000, 0, 17.6235, -16.6000, -5.8000, -1.8000,
#      -1.6000, -5.8000, -5.0000])

30
v = np.array(
    [1, 1, 0.9831, 0.9801, 0.9824, 0.9732, 0.9674, 0.9606, 0.9805, 0.9844, 0.9805, 0.9855, 1, 0.9767, 0.9802, 0.9774,
     0.9769, 0.9684, 0.9653, 0.9692, 0.9934, 1, 1, 0.9886, 0.9902, 0.9722, 1, 0.9747, 0.9796, 0.9679]) * 10

a = np.array(
    [0, -0.4155, -1.5221, -1.7947, -1.8638, -2.267, -2.6518, -2.7258, -2.9969, -3.3749, -2.9969, -1.5369, 1.4762,
     -2.308, -2.3118, -2.6445, -3.3923, -3.4784, -3.9582, -3.871, -3.4884, -3.3927, -1.5892, -2.6315, -1.69, -2.1393,
     -0.8284, -2.2659, -2.1285, -3.0415])

p_bus = np.array(
    [25.9738, 39.27, -2.4, -7.6, 0, 0, -22.8, -30, 0, -5.8, 0, -11.2, 37, -6.2, -8.2, -3.5, -9, -3.2, -9.5, -2.2, -17.5,
     21.59, 16, -8.7, 0, -3.5, 26.91, 0, -2.4, -10.6])

q_bus = np.array(
    [-0.9985, 19.299, -1.2, -1.6, 0, 0, -10.9, -30, 0, -2, 0, -7.5, 11.3529, -1.6, -2.5, -1.8, -5.8, -0.9, -3.4, -0.7,
     -11.2, 39.57, 6.351, -6.7, 0, -2.3, 10.5405, 0, -0.9, -1.9])

V = np.reshape(v, (-1, num_bus))
A = np.reshape(a, (-1, num_bus))

print(V.shape, A.shape, p_bus.shape, q_bus.shape)

G = pd.read_csv('bus_config/Ybus_' + str(num_bus) + '_real', header=None)._values
# B = pd.read_csv('bus_config/Ybus_14_imag_without_shunt', header=None).values
B = pd.read_csv('bus_config/Ybus_' + str(num_bus) + '_imag', header=None).values

print('real P bus', p_bus)
print("real Q bus", q_bus)

batch_size = 1

# estimate_np
print(A.shape)
A = A.reshape(batch_size, num_bus, 1)
# print(A)
# print(np.tile(A, (1, 1, num_bus)))
# print(np.transpose(np.tile(A, (1, 1, num_bus)), (0, 2, 1)))
#
# A_ = np.transpose(np.tile(A, (1, 1, num_bus)), (0, 2, 1)) - np.tile(A, (1, 1, num_bus))
# cos_ = np.cos(A_ * pi / 180)
# sin_ = np.sin(A_ * pi / 180)
#
# term_1_P = G * cos_ + B * sin_
# term_1_Q = G * sin_ - B * cos_
# P_bus = (V * np.tensordot(V, term_1_P, axes=([1], [1, 2])))
# Q_bus = (V * np.tensordot(V, term_1_Q, axes=([1],[1, 2])))


P_bus = np.zeros((batch_size, num_bus))
Q_bus = np.zeros((batch_size, num_bus))

for k in range(batch_size):
    for i in range(num_bus):
        for j in range(num_bus):
            cos_ = cos((A[k, i] - A[k, j]) * pi / 180)
            sin_ = sin((A[k, i] - A[k, j]) * pi / 180)
            P_bus[k, i] += V[k, i] * V[k, j] * (G[i, j] * cos_ + B[i, j] * sin_)
            Q_bus[k, i] += V[k, i] * V[k, j] * (G[i, j] * sin_ - B[i, j] * cos_)
#
# for i in range(batch_size):
#     P_bus[i] = np.dot(V[i], term_1_P[i])
#     Q_bus[i] = np.dot(V[i], term_1_Q[i])
#
# P_bus = (V * P_bus)[0]
# Q_bus = (V * Q_bus)[0]
print(P_bus, Q_bus)
print("estimate from np\n")
print('P bus diff', P_bus - p_bus)
print('Q bus diff', Q_bus - q_bus)

# P_bus = tf.Variable(tf.zeros([batch_size, num_bus]), dtype=tf.float32)
# Q_bus = tf.Variable(tf.zeros([batch_size, num_bus]), dtype=tf.float32)
# P_line = np.zeros([num_bus, num_bus])
# Q_line = np.zeros([num_bus, num_bus])
# A_ = tf.Variable(tf.zeros((batch_size, num_bus, num_bus)), dtype=tf.float32)

V = np.reshape(v, (-1, num_bus))
A = np.reshape(a, (-1, num_bus))

V = K.variable(V)
A = K.variable(A)

G = K.constant(G)
B = K.constant(B)

A_ = K.permute_dimensions(K.repeat(A, num_bus), [0, 2, 1]) - K.repeat(A, num_bus)
cos_ = K.cos(A_ * pi / 180)
sin_ = K.sin(A_ * pi / 180)
#
# # print(K.eval(cos_))
#
term_1_P = G * cos_ + B * sin_
term_1_Q = G * sin_ - B * cos_
P_bus = V * K.batch_dot(V, term_1_P, axes=[1, 2])
Q_bus = V * K.batch_dot(V, term_1_Q, axes=[1, 2])
#
# V_ = K.permute_dimensions(K.repeat(A, 14), [0, 2, 1]) * K.repeat(A, 14)
# # print(K.eval(K.repeat(A, 14)))
# # print(K.eval(K.permute_dimensions(K.repeat(A, 14), [0, 2, 1])))
# P_line = V_ * term_1_P
#
print(K.eval(P_bus))
print(K.eval(Q_bus))
# # print(K.eval(P_line))

print('P bus diff', abs(K.eval(P_bus) - p_bus) < 0.01)
print('P bus diff', abs(K.eval(P_bus) - p_bus))
print('Q bus diff', abs(K.eval(Q_bus) - q_bus) < 0.01)
print('Q bus diff', abs(K.eval(Q_bus) - q_bus))

