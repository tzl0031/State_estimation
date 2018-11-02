import sys
import tensorflow as tf
import numpy as np
import itertools


LEARNING_RATE = 1e-3
MAX_ITERATIONS = 10000
INITIAL_CONST = 1e-3
BINARY_SEARCH_STEPS = 20
ABORT_EARLY = True
e2 = 1e-3
c1 = 0.5

class AttackL2:
    def __init__(self, sess, model, bus, initial_const, batch_size=1, learning_rate=LEARNING_RATE, binary_search_steps=BINARY_SEARCH_STEPS, max_iterations=MAX_ITERATIONS, abort_early=ABORT_EARLY):
        """

        :type batch_size: int
        """
        measurement_size, state_size = model.input_size, model.output_size

        self.sess = sess
        self.LEARNING_RATE = learning_rate
        self.BINARY_SEARCH_STEPS = binary_search_steps
        self.MAX_ITERATIONS = max_iterations
        self.ABORT_EARLY = abort_early
        self.initial_const = initial_const
        self.batch_size = batch_size
        self.model = model
        self.bus = bus

        self.repeat = binary_search_steps >= 10

        # shape of batch input
        meas_shape = (self.batch_size, self.model.input_size)
        # variable to optimize over
        delta = tf.Variable(np.ones(meas_shape, dtype=np.float32))

        # shape of input, output and c
        self.tmeasurement = tf.Variable(np.ones(meas_shape), dtype=tf.float32)
        self.tstate = tf.Variable(np.ones((self.batch_size, self.model.output_size)), dtype=tf.float32)
        self.const = tf.Variable(np.ones(self.batch_size), dtype=tf.float32)

        # placeholder
        self.assign_tmeasurement = tf.placeholder(tf.float32, meas_shape)
        self.assign_tstate = tf.placeholder(tf.float32, (self.batch_size, self.model.output_size))
        self.assign_const = tf.placeholder(tf.float32, [self.batch_size])

        self.boxmul = 1
        self.boxplus = 0
        # new measurement
        # z_tilde

        # self.new_measurement = self.tmeasurement + delta
        self.new_measurement = tf.tanh(delta + self.tmeasurement) * self.boxmul + self.boxplus

        # compute the estimated state
        # x_hat
        self.estimated_state = self.model.predict(tf.reshape(self.tmeasurement, (-1, self.model.input_size)))
        # x_tilde_hat
        self.new_estimated_state = self.model.predict(tf.reshape(self.new_measurement, (-1, self.model.input_size)))
        # l1 and l2 distance of state diff
        # (x_tilde_hat - x_hat)
        # loss1 distance of states
        self.loss1_ = tf.reduce_sum(tf.square(self.new_estimated_state - self.estimated_state), 1)

        # (tf.tanh(self.estimated_state) * self.boxmul + self.boxplus)))

        self.estimated_new_measurement = self.bus.estimated(self.new_estimated_state, batch_size)
        # loss function
        # loss2 distance of measurements
        # epsilon - (z_tilda_hat - z_tilda)
        self.loss2_ = tf.reduce_sum(tf.square(self.estimated_new_measurement - self.new_measurement), 1)
        self.loss2_ = tf.reduce_sum(tf.square(self.new_measurement - tf.tanh(self.estimated_new_measurement) * self.boxmul + self.boxplus), 1)
        # self.loss1 = tf.reduce_sum(self.const * loss1)

        # with a given c
        # loss = loss1 + loss2
        # self.loss = self.loss1 - self.loss2
        self.loss = 1. / self.loss1_ + tf.reduce_sum(self.initial_const * self.loss2_)


        start_vars = set(x.name for x in tf.global_variables())
        optimizer = tf.train.AdamOptimizer(self.LEARNING_RATE)
        self.train = optimizer.minimize(self.loss, var_list=[delta])
        end_vars = tf.global_variables()
        new_vars = [x for x in end_vars if x.name not in start_vars]

        # initialize variables
        self.setup = []
        self.setup.append(self.tmeasurement.assign(self.assign_tmeasurement))
        self.setup.append(self.tstate.assign(self.assign_tstate))
        # self.setup.append(self.const.assign(self.assign_const))

        self.init = tf.variables_initializer(var_list=[delta]+new_vars)



# measurement corresponding to new_estimated states
        # z_tilde_hat
        # convert to np and calculated measurement

        # set up adam optimizer


    def attack(self, measurement, state):
        """
        Perform the L_2 attack on the given images for the given targets.
        """
        r = []
        print('go up to', len(measurement))
        for i in range(0, len(measurement), self.batch_size):
            print("tick", i)
            r.extend(self.attack_batch(measurement[i:i+self.batch_size], state[i:i+self.batch_size]))
        return np.array(r)


    def attack_batch(self, measurement, state):
        """
        Perform attack on a batch of measurements
        :param self:
        :param measurement:
        :param state:
        :return:
        """

        batch_size = self.batch_size
        # find the best c
        lower_bound = np.zeros(batch_size)
        upper_bound = np.ones(batch_size)*1e10

        o_bestl1 = [-1e10]*batch_size
        o_bestl2 = [-1e10]*batch_size
        o_bestattack = [np.zeros(measurement[0].shape)]*batch_size


        # without binary search
        self.sess.run(self.init)

        batch_measurement = measurement[:batch_size]
        batch_state = state[:batch_size]
        bestl1 = [-1e10] * batch_size
        bestl2 = [-1e10] * batch_size



        self.sess.run(self.setup, {self.assign_tmeasurement: batch_measurement,
                                   self.assign_tstate: batch_state})
        for iteration in range(self.MAX_ITERATIONS):
            # perform attack
            # l: total loss, l1s: state dist, l2s: measurement dist
            _, l, l1s, new_measurement = self.sess.run(
                [self.train, self.loss, self.loss1_, self.new_measurement])

            # # print total loss, state dist, measurement dist
            if iteration == 0:
                print('num of iteration', "total loss", "mea_dist", "state_dist")
            if iteration % (self.MAX_ITERATIONS // 10) == 0:
                print('#', iteration, self.sess.run([self.loss, self.loss2_, self.loss1_]))
            #
            # # if self.ABORT_EARLY and iteration % (self.MAX_ITERATIONS // 10) == 0:
            # #     if l > prev * .09999:
            # #         break
            # # prev = l
            #
            for e, (l1s, new_measurement) in enumerate(zip(l1s, new_measurement)):
                if l1s > bestl1[e]:
                    bestl1[e] = l1s
                    o_bestattack[e] = new_measurement


        # for outer_step in range(self.BINARY_SEARCH_STEPS):
        #     # print(o_bestl2)
        #     # reset adam's internal state
        #     self.sess.run(self.init)
        #     batch_measurement = measurement[:batch_size]
        #     batch_state = state[:batch_size]
        #
        #     # bestl1 = [-1e10]*batch_size
        #     bestl2 = [-1e10]*batch_size
        #
        #     # The last iteration repeat the search once
        #     if self.repeat is True and outer_step == self.BINARY_SEARCH_STEPS-1:
        #         CONST = upper_bound
        #
        #     # set the variables that we don't have to send them over again
        #     self.sess.run(self.setup, {self.assign_tmeasurement: batch_measurement,
        #                                self.assign_tstate: batch_state,
        #                                self.assign_const: CONST})
        #
        #     prev = 1e6
        #     for iteration in range(self.MAX_ITERATIONS):
        #         # perform attack
        #         _, l, l1s, l2s, new_measurement = self.sess.run([self.train, self.loss, self.loss1, self.l2dist, self.new_measurement])
        #
        #         if iteration % (self.MAX_ITERATIONS//10) == 0:
        #             print('#', iteration, self.sess.run([self.loss, self.loss1, self.loss2]))
        #
        #         if self.ABORT_EARLY and iteration % (self.MAX_ITERATIONS//10) == 0:
        #             if l > prev*.09999:
        #                 break
        #         prev = l
        #
        #
        #         for e, (l2, ii) in enumerate(zip(l2s, new_measurement)):
        #             if l2 > bestl2[e]:
        #                 bestl2[e] = l2
        #
        #             # if l1 > bestl1[e]:
        #             #     bestl1[e] = l1
        #
        #             if l2 > o_bestl2[e]:
        #                 o_bestl2[e] = l2
        #                 o_bestattack[e] = ii
        #         for e in range(self.batch_size):
        #             if l2s > bestl2[e]:
        #                 bestl2[e] = l2s
        #
        #             if l2s > o_bestl2[e]:
        #                 o_bestl2[e] = l2s
        #                 o_bestattack[e] = new_measurement[e]


            # # binary search for const
            # for e in range(batch_size):
            #     if bestl2[e] > e2:
            #         # success, dive const by 2
            #         upper_bound[e] = min(upper_bound[e], CONST[e])
            #         if upper_bound[e] < 1e9:
            #             CONST[e] = (lower_bound[e] + upper_bound[e]) / 2
            #     else:
            #         # failure, multiply by 10 or search with the known upper bound
            #         lower_bound[e] = max(lower_bound, CONST[e])
            #         if upper_bound[e] < 1e9:
            #             CONST[e] = (lower_bound[e] + upper_bound[e]) / 2
            #         else:
            #             CONST[e] *= 10

        o_bestl1 = np.array(o_bestl1)
        print(self.initial_const)
        # print(o_bestattack)

        return o_bestattack
