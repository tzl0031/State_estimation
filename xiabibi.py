import pypsa
import numpy as np
network = pypsa.Network()
#add three buses
for i in range(3):
    network.add("Bus", "My bus {}".format(i))

print(network.buses)
#add three lines in a ring
for i in range(3):
    network.add("Line","My line {}".format(i), bus0="My bus {}".format(i), bus1="My bus {}".format((i+1)%3),     x=0.0001)
print(network.lines)

#add a generator at bus 0


network.add("Generator", "My gen", bus="My bus 0", p_set=100)

print(network.generators)
print(network.generators_t.p_set)
#add a load at bus 1
network.add("Load","My load",
bus="My bus 1",
p_set=100)
print(network.loads)
print(network.loads_t.p_set)
#Do a Newton-Raphson power flow
network.pf()
print(network.lines_t.p0)
print(network.buses_t.v_ang*180/np.pi)