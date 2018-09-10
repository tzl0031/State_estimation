from model import Model14
import numpy as np
import pandas as pd
import helper as helper
from differential_evolution import differential_evolution
from model import Model9
from model import StateEstimation
from config import Bus9
import tensorflow as tf


thress1 = 1
thress2 = 50

class Attack_Pixel:
    def __init__(self, bus, model, measurement, thress1, thress2):
        self.model = model
        self.thress1 = thress1
        self.thress2 = thress2

        self.measurement = measurement
        self.bus = bus

        self.measurement_size, self.state_size = self.model.input_size, self.model.output_size
        # measurement: original measurement
        # delta: injected data, format [index, value]
        # new_measurement = measurement + delta
        # estimated state : state estimated from measurement
        # new_estimated_state: state estimated from new measurement

    def attack_success(self, delta, i, verbose=False):
        # perturb the image ith given pixel and get the state estimation of the model
        attack_meas = helper.perturb_meas(delta, self.measurement[i])
        # must use tensor to predict
        estimated_state = self.model.model.predict(np.reshape(self.measurement[i], (-1, self.measurement_size)))
        new_estimated_state = self.model.model.predict(np.reshape(attack_meas, (-1, self.measurement_size)))
        state_diff = np.sqrt(np.sum(np.square(new_estimated_state - estimated_state)))
        new_estimated_measurement = self.bus.estimated_np(new_estimated_state)
        meas_diff = np.sqrt(np.sum(np.square(new_estimated_measurement - attack_meas), axis=1))

        if verbose:
            print("state diff", state_diff)
        if state_diff > self.thress1 and meas_diff < self.thress2:
            return True

    def objective(self, deltas, i):

        # print(self.measurement[i].shape)
        # z_tilda
        attack_meas = helper.perturb_meas(deltas, self.measurement[i])
        # must use tensor to predict
        # x_hat
        estimated_state = self.model.model.predict(np.reshape(self.measurement[i], (-1, self.measurement_size)))
        # print("diff with real state", )
        # x_tilda_hat
        new_estimated_state = self.model.model.predict(np.reshape(attack_meas, (-1, self.measurement_size)))
        # print(estimated_state.shape)
        # print(estimated_state[:, 6:].shape)
        #
        # print((new_estimated_state[:, 6:] - estimated_state[:, 6:]).shape)
        # In general, we want to maximize the state difference.
        # Consider the range of voltage change being too small, we only target on phase angle manipulation.
        # state_diff = np.sum(np.square(new_estimated_state - estimated_state))
        # x_hat - x_tilda
        state_diff = np.sqrt(np.sum(np.square(new_estimated_state - estimated_state), axis=1))
        # print(state_diff.shape
        # z_tilda_hat
        new_estimated_measurement = self.bus.estimated_np(new_estimated_state)
        meas_diff = np.sqrt(np.sum(np.square(new_estimated_measurement - attack_meas), axis=1))


        # if verbose:
        #     # print("original measurement", self.measurement[i])
        #     # print("new measurement", attack_meas)
        #     print("attack vector", delta)
        #     # print("original state", estimated_state)
        #     # print("after attack", new_estimated_state)
        #     print("state distortion", state_diff)

        # objective functions:
        # min(1/state_diff)
        # add small value to avoid divide by 0
        return state_diff, meas_diff



    def attack(self, i, pixel_count=1, maxiter=75, popsize=100, verbose=False):
        # define bounds for a flat vector, delta has format [index, value]
        # ensure the validity of the pixel
        bounds = [(9, 18), (-1, 1)] * pixel_count

        # population multipliers
        popmul = max(1, popsize // len(bounds))

        # predict and callback functions
        # objective function to be minimized



        predict_fn = lambda deltas: self.objective(deltas, i)[1]

        # early stop
        callback_fn = lambda delta, convergence: self.attack_success(delta, i, verbose)
        # constraint
        cons = {"type": "ineq", "fun": lambda deltas: self.objective(deltas, i)[0] - thress1}

        # call differential evolution
        attack_result = differential_evolution(
           predict_fn, bounds, constraint=cons, maxiter=maxiter, popsize=popmul,
            recombination=1, atol=1, callback=callback_fn, polish=False)

        # evaluation

        # z_tilda
        # print(attack_result.x)
        new_measurement = helper.perturb_meas(attack_result.x, self.measurement[i])
        # print(self.measurement[i] - new_measurement[0])
        # print(new_measurement[0])

        # x_hat
        estimated_state = self.model.model.predict(np.reshape(self.measurement[i], (-1, 36)))
        # x_tilda_hat
        new_estimated_state = self.model.model.predict(np.reshape(new_measurement, (-1, 36)))
        # print(new_estimated_state.shape)
        # z_tilda_hat
        # print(new_estimated_measurement.shape)
        # print(self.measurement[i].shape)
        # print(new_estimated_state)
        # print(estimated_state)
        state_diff = np.sqrt(np.sum(np.square(new_estimated_state - estimated_state), axis=1))
        # print(state_diff.shape)
        new_estimated_measurement = self.bus.estimated_np(new_estimated_state)
        meas_diff = np.sqrt(np.sum(np.square(new_estimated_measurement - new_measurement), axis=1))


        # print(new_measurement * (max_-min_) + min_)
        # print(new_estimated_measurement)
        # print(self.measurement[i] * (max_-min_) + min_)

        success = (state_diff > self.thress1 and meas_diff < self.thress2)

        if verbose:
            # print("original measurement", self.measurement)
            # print("new measurement", new_measurement)
            print("attack vector", attack_result.x)
            # print("original state", estimated_state)
            # print("after attack", new_estimated_state)
            print("state distortion", state_diff)


        return [pixel_count, success, i, state_diff, meas_diff, attack_result.x]

    def attack_all(self, samples=500, pixels=[1, 3, 5, 7], maxiter=75, popsize=200, verbose=True):
        results = []
        measurement_samples = self.measurement[:samples]
        for pixel_count in pixels:
            print("launching %d pixel attack" % pixel_count)
        # print("pixel", pixel_count)
            for i, measurement in enumerate(measurement_samples):
                result = self.attack(i, pixel_count, maxiter=maxiter, popsize=popsize)
                results.append(result)

        return results


if __name__ == "__main__":
    data = StateEstimation()

    model = Model9("models/train_9")
    bus_9 = Bus9()
    attacker = Attack_Pixel(bus_9, model, data.test_data[:10], thress1, thress2)
    print("starting attack")

    results = attacker.attack_all()
    columns = ["pixel_count", "success", "i", "state_diff", "meas_diff", "injection"]
    results_table = pd.DataFrame(results, columns=columns)
    print(results_table[["pixel_count", "success", "i", "state_diff", "meas_diff", "injection"]])
