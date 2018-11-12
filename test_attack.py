import tensorflow as tf
import numpy as np
import time
from config9 import Bus9
from model import StateEstimation, Model14, Model30, Model9
from L2_attack import AttackL2
import matplotlib.pyplot as plt
from math import *


# generate a batch of inputs and outputs
def generate_data(data, num_samples=10, start=0):
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



if __name__ == "__main__":
    with tf.Session() as sess:
        data = StateEstimation()
        # print(data.train_data.shape)
        model = Model9("models/train_9", sess)
        bus = Bus9()

        measurement, state = generate_data(data, num_samples=9)
        # print(measurement.shape)

        c_record = []
        f0_record = []
        f1_record = []



        for c1 in range(-3, 3):
            for c2 in range(-3, 3):
                attack = AttackL2(sess, model, bus, batch_size=9, initial_const1=10**c1, initial_const2 = 10**c2,max_iterations=1000)
                time_start = time.time()
                new_measurement = attack.attack(measurement, state)
                # print(new_measurement)
                time_end = time.time()

                print("Took", time_end-time_start, "seconds to generate", len(measurement), "samples")
                # for i in range(len(new_measurement)):
                #     print("Valid:")
                #     print(measurement[i])
                #     print("Adversarial")
                #     print(new_measurement[i])
                #     estimated_state = model.model.predict(np.reshape(measurement[i], (-1, 6)))
                #     estimated_new_state = model.model.predict(np.reshape(new_measurement[i], (-1, 6)))
                #     new_estimated_measurement = bus.estimated_np(estimated_new_state)
                #     print("new estimated", new_estimated_measurement)
                #
                #     # print("State Estimation of the valid", estimated_state)
                #     # print("State Estimation of the adversarial", estimated_new_state)
                #     print("State diff", np.abs(estimated_new_state - estimated_state))
                #
                #     print("State diff", np.mean(np.abs(estimated_new_state - estimated_state)))
                #     print("Measurement diff", np.sqrt(np.sum(np.square(new_estimated_measurement - new_measurement[0]))))
                #     print("injection:", np.sqrt(np.sum(np.square(new_measurement - measurement), 1)))
            # new_estimated_measurement = new_estimated_measurement_tensor.eval(session=tf.Session())
            # new_estimated_measurement = np.reshape(new_estimated_measurement, (-1, 112))

            # print("Total distortion(measurement):", (np.sum(new_measurement - new_estimated_measurement)**2)**.5)


