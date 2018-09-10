from pypower.api import case9
import pypsa
import pandas as pd
import numpy as np
from math import *
from scipy import sparse

network = pypsa.Network()
ppc = case9()
network.import_from_pypower_ppc(ppc)
bus_num = np.array(network.lines[['bus0', 'bus1']].values)

V = np.array([1.0000, 1.0000, 1.0000, 0.9870, 0.9755, 1.0034, 0.9856, 0.9962, 0.9576]) * 10
A = np.array([0, 9.6687, 4.7711, -2.4066, -4.0173, 1.9256, 0.6215, 3.7991, -4.3499])
p_bus = np.array([71.9547, 163.0000, 85.0000, 0, -90, 0, -100, 0, -125])
q_bus = np.array([24.07, 14.46, -3.65, 0, -30, 0, -35, 0, -50])
p_line_i = np.array([71.9547, 30.7283, -59.4453, 85.0000, 24.1061, -75.9894, -163.0000, 86.5044, -40.9601])
p_line_j = np.array([-71.9547, -30.5547, 60.8939, -85.0000, -24.0106, 76.4956,163.0000, -84.0399, 41.2264])
q_line_i = np.array([24.0690, -0.5859, -16.3120, -3.6490, 4.5368, -10.5992, 2.2762, -2.5324, -35.7180])
q_line_j = np.array([-20.7530, -13.6880, -12.4275, 7.8907, -24.4008, 0.2562, 14.4601, -14.2820, 21.3389])


print(V.shape, A.shape, p_bus.shape, q_bus.shape, p_line_i.shape, p_line_j.shape, q_line_i.shape, q_line_j.shape)


G = pd.read_csv('bus_config/Ybus_9_real', header=None)
B = pd.read_csv('bus_config/Ybus_9_imag', header=None)
g = pd.read_csv('bus_config/Ybranch_9_real', header=None)
b = pd.read_csv('bus_config/Ybranch_9_imag', header=None)
G = G.values
B = B.values
g = g.values
b = b.values
num_bus = 9
num_lines = 9

# print(bus_num)
# print("voltage", V)
# print("angle in rad", A)
# print('real P bus', p_bus)
# print("P line i", p_line_i)
# print("P line j", p_line_j)
# print("real Q bus", q_bus)
# print("real Q line i", q_line_i)
# print("real Q line j", q_line_j)

P_bus = np.zeros(num_bus)
Q_bus = np.zeros(num_bus)
V = np.reshape(V, (num_bus, -1))
P_line = np.zeros([num_bus, num_bus])
Q_line = np.zeros([num_bus, num_bus])
A_ = np.zeros((num_bus, num_bus))

for i in range(num_bus):
    for j in range(num_bus):
        cos_ = cos((A[i] - A[j]) * pi / 180)
        sin_ = sin((A[i] - A[j]) * pi / 180)
        P_bus[i] += V[i] * V[j] * (G[i, j] * cos_ + B[i, j] * sin_)
        Q_bus[i] += V[i] * V[j] * (G[i, j] * sin_ - B[i, j] * cos_)

        P_line[i, j] = V[i] * V[j] * (G[i, j] * cos_ + B[i, j] * sin_) - V[i] ** 2 * G[i, j]
        Q_line[i, j] = V[i] * V[j] * (G[i, j] * sin_ - B[i, j] * cos_) + V[i] ** 2 * B[i, j]

# print('P bus diff',  P_bus - p_bus)
# print('P bus', P_bus)
# print('P line', P_line)
# print('P line', sparse.csr_matrix(P_line))
# print('Q line', sparse.csr_matrix(Q_line))
#
# print("Q bus diff", (Q_bus - q_bus))

P0 = np.zeros(num_lines)
P1 = np.zeros(num_lines)
for i in range(num_lines):
    x = int(bus_num[i, 0]) - 1
    y = int(bus_num[i, 1]) - 1
    # print(x, y）
    P0[i] = P_line[x, y]
    P1[i] = P_line[y, x]



print(P0, P1)

Q0 = np.zeros(num_lines)
Q1 = np.zeros(num_lines)
for i in range(num_lines):
    x = int(bus_num[i, 0]) - 1
    y = int(bus_num[i, 1]) - 1
    # print(x, y）
    Q0[i] = Q_line[x, y]
    Q1[i] = Q_line[y, x]
    # print(P0, P1)

# estimated_measurement = np.concat([P_bus, P0, P1], 1)
print(Q0)
print(Q1)

print("P line i diff", abs(p_line_i - P0) <=0.01)
print("P line j diff", abs(p_line_j - P1) <=0.01)
print("Q line i diff", abs(q_line_i - Q0) )
print("Q line j diff", q_line_j - Q1)


