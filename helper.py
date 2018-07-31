import numpy as np
import pandas as pd


def perturb_meas(delta, meas):
    # delta has format [x, value]
    """

    :param delta:
    :param meas:
    :return:
    """
    delta[0] = int(delta[0])
    pos, value = delta
    pos = int(pos)
    meas[pos] = value

    return meas





