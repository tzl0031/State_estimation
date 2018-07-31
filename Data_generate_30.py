from pypower.api import case30
import pypsa
import numpy as np


# parameters
num_train = 10000
num_test = 1000
num_buses = 30
num_branches = 41

mpc = case30()
network = pypsa.Network()
network.import_from_pypower_ppc(mpc)

M = np.zeros([num_train, 2*num_branches + 1 * num_buses])
E = np.zeros([num_train, 2*num_buses])


# network.set_snapshots(range(num_train))
temp = network.loads[["p_set", "q_set"]]

for i in range(num_train):
    network.loads[['p_set', 'q_set']] = temp * np.random.normal(1, 0.5)
    network.pf()
    # mag and ang
    v_ang = network.buses_t.v_ang
    v_ang = v_ang.values[0]
    v_mag = network.buses_t.v_mag_pu
    v_mag = v_mag.values[0]
    Y = [v_mag, v_ang]
    Y = np.reshape(Y, [1, -1])[0]
    # print(Y)
    # measurements: p and q at each bus and each lines
    bus_p = network.buses_t['p'].values[0]
    bus_q = network.buses_t['q'].values[0]
    lines_p0 = network.lines_t['p0'].values[0]
    lines_q0 = network.lines_t['q0'].values[0]
    lines_p1 = network.lines_t['p1'].values[0]
    lines_q1 = network.lines_t['q1'].values[0]

    # num measurements = 30*2 + 41 *4
    X = np.concatenate([bus_p, lines_p0, lines_p1])
    X = np.reshape(X, [1, -1])[0]
    print(X)

    M[i, :] = X
    E[i, :] = Y
    # print(X, Y)

print(M.shape, E.shape)

np.savetxt("state_30", E, delimiter=",")
np.savetxt("measurement_30", M, delimiter=",")

