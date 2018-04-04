import sys
import tensorflow as tf
import numpy as np


LEARNING_RATE = 1e-3
MAX_ITERATIONS = 10000
INITIAL_CONST = 1e-3
BINARY_SEARCH_STEPS = 9
ABORT_EARLY = True
e1 = 1e-3
e2 = 1e-3

class Attack:
    def __init__(self, sess, model, batch_size=1, initial_const=INITIAL_CONST, learning_rate=LEARNING_RATE, binary_search_steps=BINARY_SEARCH_STEPS, max_iterations=MAX_ITERATIONS, abort_early = ABORT_EARLY, box_min=-0.5, box_max=0.5):

        input_size, output_size = model.input_size, model.output_size

        self.sess = sess
        self.LEARNING_RATE = learning_rate
        self.batch_size = batch_size
        self.initial_const = INITIAL_CONST
        self.BINARY_SEARCH_STEPS = binary_search_steps
        self.MAX_ITERATIONS = max_iterations
        self.ABORT_EARLY = abort_early

        self.repeat = binary_search_steps >= 10

        # shape of batch input
        shape = (batch_size, input_size)
        # variable to optimize over
        delta = tf.Variable(np.zeros(shape, dtype=np.float32))

        # shape of input, output and const
        self.tinput  = tf.Variable(np.zeros(shape), dtype=tf.float32)
        self.toutput = tf.Variable(np.zeros(batch_size, output_size))
        self.const = tf.Variable(np.zeros(batch_size), dtype=tf.float32)
        # assign
        self.assign_tinput = tf.placeholder(tf.float32, shape)
        self.assign_toutput = tf.placeholder(tf.float32, (batch_size, output_size))
        self.assign_const = tf.placeholder(tf.float32, [batch_size])

        # tanh transform
        self.boxmul = (box_max - box_min) / 2.
        self.boxplus = (box_min + box_max) / 2.
        self.newinput = 0.5 * tf.tanh(delta + self.tinput) + 1

        # compute the output
        self.output = model.predict(self.newinput)

        # L2 distance
        # self.l2dist = tf.reduce_sum(tf.square(self.newinput-(self.boxmul * tf.tanh(self.tinput) + self.boxplus)))

        # new loss function
        # loss1 distance of states
        self.loss1 = tf.reduce_mean(-tf.square(self.output - model.predict(self.tinput)))


        # loss1 distance of measurements
        self.loss2 = tf.reduce_mean(-self.const * tf.squared_difference(self.newinput, self.boxmul * tf.tanh(self.tinput) + self.boxplus))
        self.loss = self.loss1 + self.loss2

        # set up adam optimizer
        start_vars = set(x.name for x in tf.global_variables())
        optimizer = tf.train.AdamOptimizer(self.LEARNING_RATE)
        self.train = optimizer.minimize(-self.loss, var_list=[delta])
        end_vars = tf.global_variables()
        new_vars = [x for x in end_vars if x.name not in start_vars]


        # initialize variables
        self.setup = []
        self.setup.append(self.tinput.assign(self.assign_tinput))
        self.setup.append(self.toutput.assign(self.assign_toutput))
        self.setup.append(self.const.assign(self.assign_const))

        self.init = tf.variables_initializer(var_list=[delta] + self.newinput)

    def attack_batch(self, inputs, outputs):
        """
        Run the attack on a batch of measurements and states.
        :param inputs: a batch of inputs
        :param outputs: corresponding outputs
        :return:
        """

        # compare the 2 outputs
        def success(input, input_adv, model):
            l1 = tf.reduce_sum(tf.square(input, input_adv)) >= e1
            l2 = tf.reduce_sum(tf.square(model.predict(input) - model.predict(input_adv))) >= e2

            return tf.reduce_sum(tf.square(input, input_adv)) >= e1 and tf.reduce_sum(tf.square(model.predict(input) - model.predict(input_adv))) >= e2

        batch_size = self.batch_size
        # convert to tanh
        inputs = np.arctanh((inputs - self.boxplus) / self.boxmul * 0.999999)

        lower_bound = np.zeros(batch_size)
        CONST = np.ones(batch_size) * self.initial_const
        upper_bound = np.ones(batch_size) * 1e10

        o_bestl2 = [1e10]*batch_size
        o_bestattack = [np.zeros(inputs[0].shape)]

        for outer_step in range(self.BINARY_SEARCH_STEPS):
            print(o_bestl2)
            # reset adam's internal state
            self.sess.run(self.init)
            batch = inputs[:batch_size]
            batch_output = outputs[: batch_size]

            bestl2 = [1e10]*batch_size
            bestscore = [-1]*batch_size

            #
            if self.repeat is True and outer_step == self.BINARY_SEARCH_STEPS-1:
                CONST = upper_bound

            # set the variables
            self.sess.run(self.setup, {self.assign_tinput: batch,
                                       self.assign_toutput: batch_output,
                                       self.assign_const: CONST})

            prev = 1e6
            for iteration in range(self.MAX_ITERATIONS):
                # perform attack
                _, l, l1, l2, toutput, ninput = self.sess.run([self.train, self.loss, self.loss1, self.loss2, self.toutput, self.newinput])

                if iteration % (self.MAX_ITERATIONS//10) == 0:
                    print(iteration, self.sess.run(self.loss, self.loss1, self.loss2))

                if self.ABORT_EARLY and iteration%(self.MAX_ITERATIONS//10) == 0:
                    if l > prev*.09999:
                        break
                    prev = l

                for e, (l2, sc, ii) in enumerate(zip(self.loss1, toutput, ninput)):
                    if l2 < bestl2[e] and success(toutput, ninput, model):
                        bestl2[e] = l2

                    if l2 < o_bestl2[e]:
                        o_bestl2[e] = l2
                        o_bestattack[e] = ii

            # binary search for const
            for e in range(batch_size):
                if bestl2[e] > e2 and bestscore != -1:
                    # success, dive const by 2
                    upper_bound[e] = min(upper_bound[e], CONST[e])
                    if upper_bound[e] < 1e9:
                        CONST[e] = (lower_bound[e] + upper_bound[e]) / 2
                else:
                    # failure, multiply by 10 or search with the known upper bound
                    lower_bound[e] = max(lower_bound, CONST[e])
                    if upper_bound[e] < 1e9:
                        CONST[e] = (lower_bound[e] + upper_bound[e]) / 2
                    else:
                        CONST[e] *= 10

        o_bestl2 = np.array(o_bestl2)

        return o_bestattack
