from model import Model14
import numpy as np
import pandas as pd
import helper as helper
from differential_evolution import differential_evolution
from model import Model9
from model import StateEstimation
from config1 import *
import tensorflow as tf
# from scipy.optimize import differential_evolution
import datetime

# state_diff
thress1 = 0.1
# measurement diff
thress2 = 0.05


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
        if state_diff > self.thress1 and meas_diff < self.thress2 and abs(delta[1]) < 0.2:
            return True

    def objective(self, deltas, i):

        # z_tilda
        attack_meas = helper.perturb_meas(deltas, self.measurement[i])
        # must use tensor to predict
        # x_hat
        estimated_state = self.model.model.predict(np.reshape(self.measurement[i], (-1, self.measurement_size)))
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
        meas_diff_L2 = np.sqrt(np.sum(np.square(new_estimated_measurement - attack_meas), axis=1))
        meas_diff_L_inf = np.max(np.abs(new_estimated_measurement - attack_meas), axis=1)


        # meas_diff = np.sum(new_estimated_measurement - attack_meas, axis=1)


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
        return state_diff, meas_diff_L2



    def attack(self, i, pixel_count=1, maxiter=75, popsize=100, verbose=False):
        # define bounds for a flat vector, delta has format [index, value]
        # ensure the validity of the pixel
        bounds = [(0, 6), (-0.01, 0.01)] * pixel_count

        # population multipliers
        popmul = max(1, popsize // len(bounds))

        # objective function to be minimized
        predict_fn = lambda deltas: - self.objective(deltas, i)[0]

        # early stop
        callback_fn = lambda delta, convergence: self.attack_success(delta, i, verbose)

        con = lambda deltas: thress1 - self.objective(deltas, i)[0]

        # call differential evolution
        cons = {'type': 'eq', "fun": con}

        # evaluation

        attack_result = differential_evolution(predict_fn, bounds, strategy='best2bin', maxiter=maxiter, popsize=popmul,
                                               recombination=0.5, callback=callback_fn, polish=False, constraint=cons)

        new_measurement = helper.perturb_meas(attack_result.x, self.measurement[i])

        # x_hat
        estimated_state = self.model.model.predict(np.reshape(self.measurement[i], (-1, 6)))
        # x_tilda_hat
        new_estimated_state = self.model.model.predict(np.reshape(new_measurement, (-1, 6)))
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
        restore_meas_diff = np.sum(np.abs(restore_meas(new_estimated_measurement) - restore_meas(new_measurement)), axis=1)
        restore_state_diff_mag = np.sum(np.abs(restore_state(new_estimated_state)[:, :9] - restore_state(estimated_state)[:, :9]), axis=1)
        restore_state_diff_ang = np.sum(np.abs(restore_state(new_estimated_state)[:, 9:] - restore_state(estimated_state)[:, 9:])) * 180 / pi
        i = int(attack_result.x[0])
        min_ = min_meas_np[i]
        max_ = max_meas_np[i]
        restore_injection = (attack_result.x[1])/2 * (max_ - min_)


        # print(new_measurement * (max_-min_) + min_)
        # print(new_estimated_measurement)
        # print(self.measurement[i] * (max_-min_) + min_)

        success = (restore_state_diff_ang > 5)

        if verbose:
            # print("original measurement", self.measurement)
            # print("new measurement", new_measurement)
            print("attack vector", attack_result.x)
            # print("original state", estimated_state)
            # print("after attack", new_estimated_state)
            print("state distortion", state_diff)


        # return [int(attack_result.x[0]), attack_result.x[1]]


        return [pixel_count, success, i, state_diff, restore_state_diff_mag, restore_state_diff_ang, meas_diff,restore_meas_diff, int(attack_result.x[0]), attack_result.x[1], restore_injection, np.abs(new_estimated_measurement-self.measurement[i])]

    def attack_all(self, samples=500, pixels=[1], maxiter=75, popsize=200, verbose=True):
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

    attacker = Attack_Pixel(bus_9, model, data.test_data[:1000], thress1, thress2)
    print("starting attack")
    start_time = datetime.datetime.now()
    results = attacker.attack_all()
    end_time = datetime.datetime.now()
    columns = ["pixel_count", "success", "i", "state_diff", "state_diff_restore_mag","state_diff_restore_ang", "meas_diff", "meas_diff_restore", "index", "injection", "restore_injection", "estimated - real"]
    results_table = pd.DataFrame(results, columns=columns)
    results_table.to_csv("one_pixel_0.01_L2_clip_opt_state_diff_1031.csv", sep='\t')
    print(results_table[["pixel_count", "success", "i", "state_diff", "state_diff_restore_mag","state_diff_restore_ang", "meas_diff", "meas_diff_restore", "index", "injection","restore_injection", "estimated - real"]])
    print("seconds", (end_time-start_time).seconds)