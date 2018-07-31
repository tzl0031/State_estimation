from model import Model14
import numpy as np
import pandas as pd
import helper as helper
from differential_evolution import differential_evolution
from model import Model30
from model import StateEstimation
import tensorflow as tf


class Attack_Pixel:
    def __init__(self, model, measurement, thress):
        self.model = model
        self.thress = thress
        self.measurement = measurement

        self.measurement_size, self.state_size = self.model.input_size, self.model.output_size
        # measurement: original measurement
        # delta: injected data, format [index, value]
        # new_measurement = measurement + delta
        # estimated state : state estimated from measurement
        # new_estimated_state: state estimated from new measurement

    def attack_success(self, delta, i, verbose=True):
        # perturb the image ith given pixel and get the state estimation of the model
        attack_meas = helper.perturb_meas(delta, self.measurement[i])
        # must use tensor to predict
        self.tmeasurement = tf.Variable(np.zeros(self.measurement_size, self.state_size), dtype=tf.float32)
        estimated_state = self.model.predict(self.measurement[i])
        new_estimated_state = self.model.predict(attack_meas)

        if verbose:
            print("original measurement", self.measurement[i])
            print("new measurement", attack_meas)
            print("attack vector", delta)
            print("original state", estimated_state)
            print("after attack", new_estimated_state)

        if np.sum(np.square(new_estimated_state - estimated_state)) > self.thres:
            return True

    def attack(self, i, pixel_count=1, maxiter=75, popsize=400, verbose=True):
        # define bounds for a flat vector, delta has format [index, value]
        # ensure the validity of the pixel
        bounds = [(0, self.measurement_size), (-0.5, 0.5)] * pixel_count

        # population multiplier
        popmul = max(1, popsize // len(bounds))

        # predict and callback functions
        # objective function to be minimized
        predict_fn = lambda delta: np.square(delta[1])
        # constraints
        callback_fn = lambda delta, convergence: self.attack_success(delta, i, verbose)

        # call differential evolution
        attack_result = differential_evolution(
            predict_fn, bounds, maxiter=maxiter, popsize=popmul,
            recombination=1, atol=-1, callback=callback_fn, polish=False)

        # evaluation

        # z_tilda
        new_measurement = helper.perturb_meas(attack_result.x, self.measurement[i])
        # x_hat
        estimated_state = self.model.predict(self.measurement[i])
        # x_tilda_hat
        new_estimated_state = self.model.predict(new_measurement[i])

        state_diff = np.sum(np.square(new_estimated_state - estimated_state))

        success = (state_diff > self.thress)

        if verbose:
            print("original measurement", self.measurement)
            print("new measurement", new_measurement)
            print("attack vector", attack_result.x)
            print("original state", estimated_state)
            print("after attack", new_estimated_state)

        return [pixel_count, success, self.measurement[i], new_measurement, attack_result.x, estimated_state, new_estimated_state, state_diff]

    def attack_all(self, samples=500, pixels=(1, 3, 5, 7), maxiter=75, popsize=400, verbose=False):
        results = []
        measurement_samples = self.measurement[:samples]
        for pixel_count in pixels:
            for i, measurement in enumerate(measurement_samples):
                result = self.attack(i, pixel_count, maxiter=maxiter, popsize=popsize, verbose=verbose)

        results.append(result)

        return results


if __name__ == "__main__":
    data = StateEstimation()

    model = Model30("models/train_30")
    thress = 0.5
    attacker = Attack_Pixel(model, data.test_data, thress=0.5)
    print("starting attack")
    results = attacker.attack_all()
