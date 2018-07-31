import pypsa
import numpy as np
from pypower.api import case14, case30
from math import *
import pandas as pd
import tensorflow as tf



class Bus14:
    def __init__(self):
        G = pd.read_csv('bus_config/Ybus_14_real', header=None)
        B = pd.read_csv('bus_config/Ybus_14_imag', header=None)
        self.network = pypsa.Network()
        ppc = case14()
        self.network.import_from_pypower_ppc(ppc)
        self.network.pf()
        self.G = G.values
        self.B = B.values
        self.shunt_b = np.zeros(14)
        self.shunt_b[8] = 0.19
        self.num_bus = 14
        self.num_lines = 17
        self.bus_num = np.array(self.network.lines[['bus0', 'bus1']].values)


    # return power injection at each bus an each line.
    def estimated(self, state):
        # print(state.shape)
        V = state[:, :14]
        A = state[:, 14:]
        # print(V.shape, A.shape)
        P_bus = tf.Variable(np.zeros((A.shape[0], 14)), dtype=tf.float32)
        P_line = tf.Variable(np.zeros((A.shape[0], 14, 14)), dtype=tf.float32)
        # Q_bus = np.zeros(14)
        # P_line = np.zeros([14, 14])
        # Q_line = np.zeros([14, 14])
        G = self.G
        B = self.B
        batch_estimated_measurement = tf.Variable(np.zeros((A.shape[0], 48)), dtype=tf.float32)
        # print(G.shape, B.shape)
        for k in range(A.shape[0]):
            for i in range(self.num_bus):
                for j in range(self.num_bus):
                    # print(G[i, j])
                    # print(B[i, j])
                    cos = tf.cos(tf.subtract(A[k, i], A[k, j]))
                    sin = tf.sin(tf.subtract(A[k, i], A[k, j]))
                    tf.assign(P_bus[k, i], tf.add(P_bus[k, i], tf.multiply(V[k, i], V[k, j]) * tf.add(G[i, j] * cos\
                                       , B[i, j] *  sin)))
                    # Q_bus[i] += V[i] * V[j] * ( \
                    #             G[i, j] * sin(A[i] - A[j]) - B[i, j] * cos(A[i] - A[j]))

                    # print('P_bus', i, P_bus)
                    tf.assign(P_line[k, i, j], tf.subtract(P_line[k, i, j], \
                                                        tf.subtract(tf.square(V[k, i]) * G[i, j], tf.multiply(V[k, i], V[k, j]) \
                                                                    * tf.add(G[i, j]*cos, B[i, j]*sin))))
                    # print('P_line',i,j , P_line)


                # Q_line[i, j] =  V[i]**2 * (B[i, j] + shunt_b[i])  + V[i]*V[j]*(G[i, j]*sin(A[i]-A[j]) - B[i, j]*cos(A[i]-A[j]))

                # print(P_line[i], Q_line[j]
            P0, P1 = self.line_matrix_converter(P_line, A.shape[0])
        # print(P0, P1, P_bus)

            tf.assign(batch_estimated_measurement[k], tf.concat([P_bus, P0, P1], 1))
        # print(batch_estimated_measurement.shape)

        return batch_estimated_measurement

    def line_matrix_converter(self, line, batch_size):

        P0 = tf.Variable(np.zeros((batch_size, 17)), dtype=tf.float32)
        P1 = tf.Variable(np.zeros((batch_size, 17)), dtype=tf.float32)
        for k in range(batch_size):
            for i in range(self.num_lines):
                x = int(self.bus_num[i, 0]) - 1
                y = int(self.bus_num[i, 1]) - 1
                # print(x, y)
                tf.assign(P0[k, i], line[k, x, y])
                tf.assign(P1[k, i], line[k, y, x])
        # print(P0, P1)

        return P0, P1

    def euclidean_distance(self, P_bus_M, P_line_M, estimated_P_bus, estimated_P_line):
        dist1 = np.sum((P_bus_M - estimated_P_bus)**2)
        dist2 = np.sum((P_line_M - estimated_P_line)**2)

        return dist1 + dist2
    #
    # def euclidean_detector(self, real_P_bus, real_P_line, estimated_P_bus,
    #                       estimated_P_line):
    #     dist1 = real_P_bus - estimated_P_busppc = case14()
    #     # dist2 = real_Q_bus - estimated_Q_bus
    #     dist3 = real_P_line - estimated_P_line
    #     # dist4 = real_Q_line - estimated_Q_line
    #
    #     euclidean = sqrt(dist1 ** 2 + dist3 ** 2 )
    #     return euclidean


class Bus30:
    def __init__(self):
        G = pd.read_csv('bus_config/Ybus_30_real', header=None)
        B = pd.read_csv('bus_config/Ybus_30_imag', header=None)
        self.network = pypsa.Network()
        ppc = case30()
        self.network.import_from_pypower_ppc(ppc)
        self.network.pf()
        self.G = G.values
        self.B = B.values
        self.num_bus = 30
        self.num_lines = 41
        self.bus_num = np.array(self.network.lines[['bus0', 'bus1']].values)


    # return power injection at each bus an each line.
    def estimated(self, state):
        # V: voltage, A: angle
        V = state[:, :self.num_bus]
        A = state[:, self.num_bus:]
        # print(V.shape, A.shape)
        P_bus = tf.Variable(np.zeros((A.shape[0], self.num_bus)), dtype=tf.float32)
        P_line = tf.Variable(np.zeros((A.shape[0], self.num_bus, self.num_bus)), dtype=tf.float32)
        # Q_bus = np.zeros(14)
        # P_line = np.zeros([14, 14])
        # Q_line = np.zeros([14, 14])
        G = self.G
        B = self.B
        batch_estimated_measurement = tf.Variable(np.zeros((A.shape[0], 112)), dtype=tf.float32)
        # print(G.shape, B.shape)
        for k in range(A.shape[0]):
            for i in range(self.num_bus):
                for j in range(self.num_bus):
                    # print(G[i, j])
                    # print(B[i, j])
                    cos = tf.cos(tf.subtract(A[k, i], A[k, j]))
                    sin = tf.sin(tf.subtract(A[k, i], A[k, j]))
                    tf.assign(P_bus[k, i], tf.add(P_bus[k, i], tf.multiply(V[k, i], V[k, j]) * tf.add(G[i, j] * cos\
                                       , B[i, j] * sin)))
                    # Q_bus[i] += V[i] * V[j] * ( \
                    #             G[i, j] * sin(A[i] - A[j]) - B[i, j] * cos(A[i] - A[j]))

                    # print('P_bus', i, P_bus)
                    tf.assign(P_line[k, i, j], tf.subtract(P_line[k, i, j], \
                                                        tf.subtract(tf.square(V[k, i]) * G[i, j], tf.multiply(V[k, i], V[k, j]) \
                                                                    * tf.add(G[i, j]*cos, B[i, j]*sin))))
                    # print('P_line',i,j , P_line)


                # Q_line[i, j] =  V[i]**2 * (B[i, j] + shunt_b[i])  + V[i]*V[j]*(G[i, j]*sin(A[i]-A[j]) - B[i, j]*cos(A[i]-A[j]))

                # print(P_line[i], Q_line[j]
            P0, P1 = self.line_matrix_converter(P_line, A.shape[0])
        # print(P0, P1, P_bus)

            tf.assign(batch_estimated_measurement[k], tf.concat([P_bus, P0, P1], 1))
        # print(batch_estimated_measurement.shape)

        return batch_estimated_measurement

    def line_matrix_converter(self, line, batch_size):

        P0 = tf.Variable(np.zeros((batch_size, self.num_lines)), dtype=tf.float32)
        P1 = tf.Variable(np.zeros((batch_size, self.num_lines)), dtype=tf.float32)
        for k in range(batch_size):
            for i in range(self.num_lines):
                x = int(self.bus_num[i, 0]) - 1
                y = int(self.bus_num[i, 1]) - 1
                # print(x, y)
                tf.assign(P0[k, i], line[k, x, y])
                tf.assign(P1[k, i], line[k, y, x])
        # print(P0, P1)

        return P0, P1

    def euclidean_distance(self, P_bus_M, P_line_M, estimated_P_bus, estimated_P_line):
        dist1 = np.sum((P_bus_M - estimated_P_bus)**2)
        dist2 = np.sum((P_line_M - estimated_P_line)**2)

        return


if __name__ == '__main__':
    bus_14 = Bus14()
    v_ang = bus_14.network.buses_t.v_ang
    # print(v_ang)
    v_ang = v_ang.values[0]
    v_mag = bus_14.network.buses_t.v_mag_pu
    v_mag = v_mag.values[0]

    state = [v_mag, v_ang]
    state = np.reshape(state, [1, -1])
    print(state)
    print(state.shape)
    P_bus_real = np.zeros(14)
    Q_bus_real = np.zeros(14)
    # P_line_real = np.zeros([14, 14])
    # Q_line_real = np.zeros([14, 14])

    gen = np.array(bus_14.network.generators['bus'])
    gen_P = np.array(bus_14.network.generators_t.p.values[0])
    gen_Q = np.array(bus_14.network.generators_t.q.values[0])
    load = np.array(bus_14.network.loads['bus'])
    load_P = np.array(bus_14.network.loads_t.p.values[0])
    load_Q = np.array(bus_14.network.loads_t.q.values[0])



    for i in range(len(gen)):
        x = int(gen[i])-1
        P_bus_real[x] = gen_P[i]
        Q_bus_real[x] = gen_Q[i]

    for i in range(len(load)):
        x = int(load[i])-1
        P_bus_real[x] -= load_P[i]
        Q_bus_real[x] -= load_Q[i]

    # print(P_bus_real)
    # print(Q_bus_real)

    bus_num = np.array(bus_14.network.lines[['bus0', 'bus1']].values)
    p0 = bus_14.network.lines_t['p0'].values[0]
    q0 = bus_14.network.lines_t['q0'].values[0]
    p1 = bus_14.network.lines_t['p1'].values[0]
    q1 = bus_14.network.lines_t['q1'].values[0]


    # for i in range(len(p0)):
    #     x = int(bus_num[i, 0])-1
    #     y = int(bus_num[i, 1])-1
    #     P_line_real[x, y] = p0[i]
    #     P_line_real[y, x] = p1[i]
    #     Q_line_real[x, y] = q0[i]
    #     Q_line_real[y, x] = q1[i]


    P_line_real = np.concatenate([p0, p1])
    estimated_measurement = bus_14.estimated(state)
    np.set_printoptions(precision=4)
    # print('P_bus', P_bus)
    # print("Q_bus", Q_bus)
    # print('P_line', P_line)
    # print("Q_line", Q_line)

    P_bus_diff = estimated_measurement[:14] - P_bus_real
    print("P bus diff", P_bus_diff)
    # print('Q bus diff', Q_bus - Q_bus_real)

    P_line_diff = estimated_measurement[14:] - P_line_real
    # Q_line_diff = (Q_line - Q_line_real)[(Q_line - Q_line_real)!=0]
    # P_line_diff[P_line_diff>1] =0
    # P_line_diff[P_line_diff < -1] = 0
    # print("P line real", P_line_real)
    print('P line diff', P_line_diff)
    # print("Q line diff", Q_line_diff)

    print("P bus euclidean", np.sum(P_bus_diff**2))
    print("P line euclidean", np.sum(P_line_diff**2))