# -*- coding: UTF-8 -*-

import os

import pickle

import numpy as np
import matplotlib.pyplot as plt

file_path = os.path.dirname(os.path.realpath(__file__))

def gen_fc():
    whole_fc = np.zeros((144, 144), dtype=np.float32)
    for i in range(0, 28):
        for j in range(28, 56):
            whole_fc[i][j] = 1
        for j in range(88, 116):
            whole_fc[i][j] = 1

    for i in range(28, 56):
        for j in range(0, 144):
            whole_fc[i][j] = 1

    for i in range(56, 88):
        for j in range(28, 56):
            whole_fc[i][j] = 1
        for j in range(88, 116):
            whole_fc[i][j] = 1

    for i in range(88, 116):
        for j in range(0, 144):
            whole_fc[i][j] = 1

    for i in range(116, 144):
        for j in range(28, 56):
            whole_fc[i][j] = 1
        for j in range(88, 116):
            whole_fc[i][j] = 1

    return whole_fc


def draw_groundtruth():
    fc = gen_fc()
    plt.figure()
    plt.imshow(fc)

    plt.show()


def save_groundtruth():
    fc = gen_fc()
    fc_dict = dict(
        fc=fc
    )
    groundtruth_folder = os.path.join(file_path, "groundtruth")
    if not os.path.exists(groundtruth_folder):
        os.makedirs(groundtruth_folder)
    with open(os.path.join(groundtruth_folder, "groundtruth.pkl"), "wb") as f:
        pickle.dump(fc_dict, f)


if __name__ == '__main__':
    save_groundtruth()
    # draw_groundtruth()
