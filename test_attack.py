import tensorflow as tf
import numpy as np
import time

from model import StateEstimation, Model
from L2_attack import Attack




# generate a batch of inputs and outputs
def generate_data(data, num_samples=1, start=0):
    """
    Generate the inputs to attack

    :param data: inputs set
    :param num_samples: number of inputs to use
    :param start: offset of the inputs
    :return:
    """
    inputs = []
    outputs = []
    for i in range(num_samples):
        inputs.append(data.test_data[start+i])
        outputs.append(data.test_label[start+i])

    inputs = np.array(inputs)
    outputs = np.array(outputs)

    return inputs, outputs


if __name__ == "__main__":
    with tf.Session() as sess:
        data, model = StateEstimation(), Model("models/", sess)
        attack = Attack(sess, model, batch_size=9, max_iterations=1000)
        inputs, outputs = generate_data(data, num_samples=1)

        timestart = time.time()
        adv = attack.attack(inputs, outputs)
        timeend = time.time()

        print("Took", timeend-timestart, "seconds to generate", len(inputs), "samples")

        for i in range(len(adv)):
            print("Valid:")
            print(inputs[i])
            print("Adversarial")
            print(adv[i])

            print("State Estimation of the valid", model.predict(inputs[i]))
            print("State Estimation of the adversarial", model.model)
            print("Total distortion(Euclidean Distance):", np.sum((adv[i] - inputs)**2)**.5)

