#!/usr/bin/env python3

import random
import numpy as np
from copy import deepcopy

import qml
from qml.kernels import get_local_symmetric_kernel_mbdf, get_local_kernel_mbdf
from qml.math import svd_solve

import MBDF


def predictions(X, Q, Xt, Qt, Y, n, sigma):
    # calculate training kernel
    K  = get_local_symmetric_kernel_mbdf(X, Q, sigma)
    # calculate test kernel
    Kt = get_local_kernel_mbdf(X, Xt, Q, Qt, sigma)

    # train model
    C = deepcopy(K)
    alpha = svd_solve(C, Y)

    # make prediction
    Yss = np.dot(Kt, alpha)

    return Yss


def grid_scan(X, Q, Y_CCSD, Y_HF, Y_HF_MP2, Y_MP2_CCSD, idx_train, test_idx):

    # number of cross validations
    nModels = 5

    # test energies
    Y_test = Y_CCSD[test_idx]

    # training set seizes
    N = [175, 375, 750, 1500, 3000, 6000 ]

    # pre optimized hyperparameters (sigmas)
    sigmas_HF   = [ 10, 100, 1000, 10, 1000, 10 ]
    sigmas_MP2  = [ 1, 1000, 100, 1000, 10, 1000 ]
    sigmas_CCSD = [ 1000, 1, 10, 10, 10, 100 ]

    for i, n_HF in enumerate(N):
        for j, n_MP2 in enumerate(N):
            for k, n_CCSD in enumerate(N):
                maes = np.array([])
                times = np.array([])

                for nModel in range(nModels):
                    random.shuffle(idx_train)
                    train_HF   = idx_train[:n_HF]
                    train_MP2  = idx_train[:n_MP2]
                    train_CCSD = idx_train[:n_CCSD]

                    # train a model for each multi level step and get predictions
                    Yp_HF      = predictions(X[train_HF], Q[train_HF], X[test_idx], Q[test_idx], Y_HF[train_HF], n_HF, sigmas_HF[i])
                    Yp_MP2     = predictions(X[train_MP2], Q[train_MP2], X[test_idx], Q[test_idx], Y_HF_MP2[train_MP2], n_MP2, sigmas_MP2[j])
                    Yp_CCSD    = predictions(X[train_CCSD], Q[train_CCSD], X[test_idx], Q[test_idx], Y_MP2_CCSD[train_CCSD], n_CCSD, sigmas_CCSD[k])

                    # get total CCSD(T) energy from multi level predictions
                    Y_multi = Yp_HF + Yp_MP2 + Yp_CCSD

                    # calculate Mean Absolute Error (MAE)
                    MAE = np.mean(np.abs(Y_test - Y_multi))
                    maes = np.append(maes, MAE)

                # print mean results from CV
                print("{}-{}-{}:\t\t\t{:.3f} kcal/mol".format(n_HF, n_MP2, n_CCSD, np.mean(maes)))


def main():
    # read data from pre computed representations and proerties
    data       = np.load("qm7b.npz", allow_pickle=True)
    X          = data['X'] # MBDF representation
    Q          = data['Q'] # nuclear charges
    Y_CCSD     = data['Y_CCSD'] # CCSD(T) energies
    Y_HF       = data['Y_HF'] # Hartree-Fock energies
    Y_HF_MP2   = data['Y_HF_MP2'] # difference between HF and MP2 energies
    Y_MP2_CCSD = data['Y_MP2_CCSD'] # difference between MP2 and CCSD(T) energies
    idx_train  = np.concatenate((data['idx_train'], data['idx_val'])) # training indeces
    idx_test   = data['idx_test'] # test indeces

    # line scan over training set sizes
    grid_scan(X, Q, Y_CCSD, Y_HF, Y_HF_MP2, Y_MP2_CCSD, idx_train, idx_test)


if __name__ == '__main__':
    main()
