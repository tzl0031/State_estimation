import tensorflow as tf
import numpy as np
import time
from config import Bus14, Bus30, Bus9
from model import StateEstimation, Model14, Model30, Model9
from L2_attack import AttackL2
import matplotlib.pyplot as plt
from math import *


# generate a batch of inputs and outputs
def generate_data(data, num_samples=1, start=0):
    """
    Generate the inputs to attack

    :param data: inputs set
    :param num_samples: number of inputs to use
    :param start: offset of the inputs
    :return:
    """
    measurement = []
    state = []
    for i in range(num_samples):
        measurement.append(data.test_data[start+i])
        state.append(data.test_label[start+i])

    state = np.array(state)
    measurement = np.array(measurement)

    return measurement, state


def eval(new_measurement, new_estimated_measurement, c):
    estimated_state = model.model.predict(np.reshape(measurement, (-1, 30)))
    estimated_new_state = model.model.predict(np.reshape(new_measurement, (-1, 30)))
    state_dist = np.sum((estimated_new_state - estimated_state)**2)**.5
    measurement_dist = np.sum((new_estimated_measurement - new_measurement)**2)**.5
    f0 = state_dist
    f1 = state_dist + c * measurement_dist

    return f0, f1


if __name__ == "__main__":
    with tf.Session() as sess:
        data = StateEstimation()
        # print(data.train_data.shape)
        model = Model9("models/train_9", sess)
        bus = Bus9()

        measurement, state = generate_data(data, num_samples=5)
        # print(measurement.shape)

        c_record = []
        f0_record = []
        f1_record = []


        for c in range(-7, 7):
            time_start = time.time()
            attack = AttackL2(sess, model, bus, batch_size=1, initial_const=10**c, max_iterations=1000)
            new_measurement = attack.attack(measurement, state)
            # print(new_measurement)
            time_end = time.time()

            print("Took", time_end-time_start, "seconds to generate", len(measurement), "samples")
            print("Valid:")
            print(measurement)
            print("Adversarial")
            print(new_measurement[0])
            estimated_state = model.model.predict(np.reshape(measurement, (-1, 36)))
            estimated_new_state = model.model.predict(np.reshape(new_measurement[0], (-1, 36)))
            new_estimated_measurement = bus.estimated(estimated_new_state)
            # arb_estimated_new_state = model.model.predict(np.reshape(measurement[i], (-1, 48)))
            print("State Estimation of the valid", estimated_state)
            print("State Estimation of the adversarial", estimated_new_state)
            print("State diff(state)", (np.sum(estimated_new_state - estimated_state)**2))**.5
            print("new estimated", new_estimated_measurement)

            # new_estimated_measurement = new_estimated_measurement_tensor.eval(session=tf.Session())
            # new_estimated_measurement = np.reshape(new_estimated_measurement, (-1, 112))

            # print("Total distortion(measurement):", (np.sum(new_measurement - new_estimated_measurement)**2)**.5)


        # print("random add distortion")
        # # print("original", measurement)
        # estimated_state = model.model.predict(np.reshape(measurement, (-1, 112)))
        # for i in range(112):
        #
        #     new_measurement = np.copy(measurement)
        #     new_measurement[0, i] = 0
        #     # print("random add 1", i)
        #     estimated_new_state = model.model.predict(np.reshape(new_measurement[0], (-1, 112)))
        #     # print("measurement distortion", np.sum((new_measurement[0, i] - measurement[0, i])**2)**.5)
        #     # print("state diff", np.sum((estimated_new_state - estimated_state)**2)**.5)
        #     print(np.sum((new_measurement[0, i] - measurement[0, i])**2)**.5 / np.sum((estimated_new_state - estimated_state)**2)**.5)

        # me = tf.Variable(np.zeros(48), dtype=tf.float32)

        # for i in range(len(new_measurement)):
        #     print("Valid:")
        #     print(measurement[i])
        #     print("Adversarial")
        #     print(new_measurement[i])
        #     tf.Session(tf.global_variables_initializer())
        #     # noise = np.random.normal(0, 0.15, size=measurement.shape)
        #     measurement1 = measurement + noise
        #     f0, f1 = eval(measurement, new_measurement, exp(c))
        #     c_record.append(c)
        #     f0_record.append(f0)
        #     f1_record.append(f1)
        #
        # plt.plot(c_record, f0_record, color='g')
        # plt.plot(c_record, f1_record, color='orange')
        # plt.xlabel('c')
        # plt.ylabel('objective function value')
        # plt.title('f0 and f1 value WRT c')
        # plt.show()
        #
        #
        #
        #     # estimated_state = model.model.predict(np.reshape(measurement[i], (-1, 48)))
        #     # estimated_new_state = model.model.predict(np.reshape(new_measurement[i], (-1, 48)))
        #     # # arb_estimated_new_state = model.model.predict(np.reshape(measurement[i], (-1, 48)))
        #     # print("Total distortion(Euclidean Distance):", np.sum((measurement[i] - measurement)**2)**.5)
        #     #
        #     #
        #     #
        #     #
        #     # print("State Estimation of the valid", estimated_state)
        #     # print("State Estimation of the adversarial", arb_estimated_new_state)
        #     # print("State diff(euclidean distance)", np.sum((arb_estimated_new_state - estimated_state)**2)**.5)
        #     #
