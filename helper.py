import numpy as np
import pandas as pd


def perturb_meas(xs, meas):
    # xs multiple pixels for 1 image
    # can have multiple xs
    # return popsize * meas
    """

    :param delta:
    :param meas:
    :return:
    """
    if xs.ndim < 2:
        xs = np.array([xs])

    tile = [len(xs)] + [1]
    meass = np.tile(meas, tile)


    for x, meas in zip(xs, meass):
        # print(len(x))
        # print(x)
        pixels = np.split(x, len(x) // 2)

        for pixel in pixels:
    # print(pixel.shape)
            pixel[0] = int(pixel[0])
    #         print(pixel)
            pos, value = pixel
            # pos = int(pos)
            # print(int(pos))
            # print(meas.shape)
            # print(meas[int(pos)])
            # print(value)
            meas[int(pos)] += value
            np.clip(meas, -1, 1)
        # print(x)
        # print(meas)

    return meass





