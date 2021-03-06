from model import Model14
import numpy as np
import pandas as pd
import helper as helper
from differential_evolution import differential_evolution
from model import Model9, Model14
from model import StateEstimation
from config9 import *
# from config_14 import *
# from config_30 import *
# import tensorflow as tf
# from scipy.optimize import differential_evolution
import datetime

# state_diff
thress1 = 0.1
# measurement diff
thress2 = 0.1
num_bus = 9
num_input = 6


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

    def attack_success(self, delta, i, verbose=True):
        # perturb the image ith given pixel and get the state estimation of the model
        attack_meas = helper.perturb_meas(delta, self.measurement[i])
        # must use tensor to predict
        estimated_state = self.model.model.predict(np.reshape(self.measurement[i], (-1, self.measurement_size)))
        new_estimated_state = self.model.model.predict(np.reshape(attack_meas, (-1, self.measurement_size)))
        state_diff = np.sqrt(np.sum(np.square(new_estimated_state - estimated_state)))
        new_estimated_measurement = self.bus.estimated_np(new_estimated_state)
        meas_diff = np.sqrt(np.sum(np.square(new_estimated_measurement - attack_meas), axis=1))
        restore_meas_diff = np.sqrt(np.sum(np.square(restore_meas(new_estimated_measurement) - restore_meas(attack_meas)), axis=1))
        restore_state_diff_ang = np.sum(np.abs(restore_state(new_estimated_state)[:, num_bus:] - restore_state(estimated_state)[:, num_bus:]), axis=1) * 180 / pi


        if verbose:
            print("state diff", state_diff)
        if restore_state_diff_ang > 5 and restore_meas_diff < 5:
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
        # x_hat - x_tilda
        state_diff = np.sqrt(np.sum(np.square(new_estimated_state - estimated_state), axis=1))

        # print(state_diff.shape
        # z_tilda_hat
        new_estimated_measurement = self.bus.estimated_np(new_estimated_state)
        # meas_diff_L2 = np.sqrt(np.sum(np.square(new_estimated_measurement - attack_meas), axis=1))
        meas_diff_L_inf = np.max(np.abs(new_estimated_measurement - attack_meas), axis=1)
        restore_meas_diff = np.sqrt(np.sum(np.square(restore_meas(new_estimated_measurement) - restore_meas(attack_meas)), axis=1))
        restore_state_diff_ang = np.sum(np.abs(restore_state(new_estimated_state)[:, num_bus:] - restore_state(estimated_state)[:, num_bus:]), axis=1) * 180 / pi



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
        return restore_state_diff_ang, restore_meas_diff



    def attack(self, i, pixel_count=1, maxiter=75, popsize=400, verbose=False):
        # define bounds for a flat vector, delta has format [index, value]
        # ensure the validity of the pixel
        bounds = [(0, num_input), (-0.1, 0.1)] * pixel_count

        # population multipliers
        popmul = max(1, popsize // len(bounds))

        # objective function to be minimized
        predict_fn = lambda deltas: -self.objective(deltas, i)[0]

        # early stop
        callback_fn = lambda delta, convergence: self.attack_success(delta, i, verbose)

        # con = lambda deltas: 0.1 - self.objective(deltas, i)[1]

        # call differential evolution
        # cons = {'type': 'ineq', "fun": con}

        # evaluation

        attack_result = differential_evolution(predict_fn, bounds, strategy='best1bin', maxiter=maxiter, popsize=popmul,
                                               recombination=1, callback=callback_fn, polish=False, atol=1)

        new_measurement = helper.perturb_meas(attack_result.x, self.measurement[i])

        # x_hat
        estimated_state = self.model.model.predict(np.reshape(self.measurement[i], (-1, num_input)))
        # x_tilda_hat
        new_estimated_state = self.model.model.predict(np.reshape(new_measurement, (-1, num_input)))
        # print(new_estimated_state.shape)
        # z_tilda_hat
        # print(new_estimated_measurement.shape)
        # print(self.measurement[i].shape)
        # print(new_estimated_state)
        # print(estimated_state)
        # state_diff = np.sqrt(np.sum(np.square(new_estimated_state - estimated_state), axis=1))
        # print(state_diff.shape)
        new_estimated_measurement = self.bus.estimated_np(new_estimated_state)
        meas_diff = np.sqrt(np.sum(np.square(new_estimated_measurement - new_measurement), axis=1))
        restore_meas_diff = np.sqrt(np.sum(np.square(restore_meas(new_estimated_measurement) - restore_meas(new_measurement)), axis=1))
        restore_state_diff_mag = np.sum(np.abs(restore_state(new_estimated_state)[:, :num_bus] - restore_state(estimated_state)[:, :num_bus]))
        restore_state_diff_ang = np.sum(np.abs(restore_state(new_estimated_state)[:, num_bus:] - restore_state(estimated_state)[:, num_bus:])) * 180 / pi
        i = int(attack_result.x[0])
        min_ = min_meas_np[i]
        max_ = max_meas_np[i]
        restore_injection = (attack_result.x[1])/2 * (max_ - min_)


        # print(new_measurement * (max_-min_) + min_)
        # print(new_estimated_measurement)
        # print(self.measurement[i] * (max_-min_) + min_)

        # success = (restore_state_diff_ang > 5)

        # if verbose:
        #     # print("original measurement", self.measurement)
        #     # print("new measurement", new_measurement)
        #     print("attack vector", attack_result.x)
        #     # print("original state", estimated_state)
        #     # print("after attack", new_estimated_state)
        #     print("state distortion", state_diff)


        # return [int(attack_result.x[0]), attack_result.x[1]]


        return [pixel_count, i, restore_state_diff_mag, restore_state_diff_ang, meas_diff[0], restore_meas_diff[0], int(attack_result.x[0]), attack_result.x[1], restore_injection]

    def attack_all(self, samples=100, pixels=[1], maxiter=75, popsize=400, verbose=True):
        results = []
        measurement_samples = self.measurement[:samples]
        for pixel_count in pixels:
            print("launching %d pixel attack" % pixel_count)
        # print("pixel", pixel_count)
            for i, measurement in enumerate(measurement_samples):
                print("attack ith measurement", i)
                start_time = datetime.datetime.now()

                result = self.attack(i, pixel_count, maxiter=maxiter, popsize=popsize)
                end_time = datetime.datetime.now()

                print("attack time for ith measurement", (end_time - start_time).seconds)
                print(result)
                results.append(result)

        return results


if __name__ == "__main__":
    data = StateEstimation()

    model = Model9("models/train_9")
    bus = Bus9()

    attacker = Attack_Pixel(bus, model, data.test_data[:100], thress1, thress2)
    print("starting attack")
    start_time = datetime.datetime.now()
    results = attacker.attack_all()
    end_time = datetime.datetime.now()
    columns = ["pixel_count", "i","state_diff_restore_mag","state_diff_restore_ang", "meas_diff", "meas_diff_restore", "index", "injection", "restore_injection"]
    results_table = pd.DataFrame(results, columns=columns)
    results_table.to_csv("9_0.csv", sep='\t')
    print(results_table[["pixel_count","i", "state_diff_restore_mag","state_diff_restore_ang", "meas_diff", "meas_diff_restore", "index", "injection","restore_injection"]])
    print("seconds", (end_time-start_time).seconds)