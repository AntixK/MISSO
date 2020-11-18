"""
Copyright (c) 2020, Anand K Subramanian.
All rights reserved.
"""

import numpy as np
from typing import Optional, Tuple

def lsmi1D(X:np.ndarray,
           Y:Optional[np.ndarray] = None,
           num_centers:Optional[int] = 200,
           rbf_sigma:Optional[float] = None,
           alpha: Optional[float] = None,
           random_seed:int = 42) -> Tuple:
    """
    Compute the LSMI estimate of the Mutualk information between two 1Dcrandom variables
    X and Y, each with M samples.
    :param X: [M, 1] Vector of samples
    :param Y: [M, 1] Vector of samples
    :param num_centers: Number of centers to use when computing RBF kernel (**default:200*)
    :param rbf_sigma: Length-scale parameters for the RBF kernel
    :param alpha: L2 regularizer for the LSMI estimator
    :param random_seed: Random seed (*default:42*)
    :return: A tuple of LSMI estimate and the cross validation score

    References:
    [1] http://www.ms.k.u-tokyo.ac.jp/software.html#LSMI

    """

    np.random.seed(random_seed)
    if Y is None:
        Y = X

    M, D = X.shape
    My, Dy = Y.shape

    assert M == My, "Both X & Y must have the same number of samples (dim 1)"
    assert D == Dy, "Both X & Y must have the same dimension (dim 2)"

    num_centers = min(num_centers, M)

    # Randomly choose the centres for Gaussian Kernels
    centers = np.random.choice(M, size=num_centers, replace=False)
    X_norm = np.sum(X**2, axis=-1) # [C,]
    Y_norm = np.sum(Y**2, axis=-1) # [C,]

    # Compute the norm with respect to the centers

    X_dist = np.tile(X_norm, (num_centers, 1)) + \
             np.tile(X_norm[None, centers].T, (1, M)) - \
             2 * X[centers, :] @ X.T       # [C x M]

    Y_dist = np.tile(Y_norm, (num_centers, 1)) + \
             np.tile(Y_norm[None, centers].T, (1, M)) - \
             2 * Y[centers, :] @ Y.T       # [C x M]

    score_cv = -np.inf

    if rbf_sigma is None or alpha is None:
        # perform cross-validation
        sigma_pool = np.logspace(-2, 2, 9)
        alpha_pool = np.logspace(-3, 1, 9)

        scores_cv = np.zeros((len(sigma_pool), len(alpha_pool)))

        num_fold = 5
        cv_indx = np.floor(np.arange(M)*num_fold/M)
        np.random.shuffle(cv_indx)
        fold_inds = np.arange(num_fold)

        n_cv = []
        for i in fold_inds:
            n_cv.append(np.sum(cv_indx == i))
        n_cv = np.array(n_cv)
        for n, sigma in enumerate(sigma_pool):
            X_phi_sigma = np.exp(- X_dist/ (2 * sigma ** 2)) # [C x M]
            Y_phi_sigma = np.exp(- Y_dist/ (2 * sigma ** 2)) # [C x M]
            K_sigma = X_phi_sigma * Y_phi_sigma

            H_cv_xphi = []
            H_cv_yphi = []
            h_cv = []
            for i in fold_inds:
                H_cv_xphi.append(X_phi_sigma[:, cv_indx == i] @ X_phi_sigma[:, cv_indx == i].T)
                H_cv_yphi.append(Y_phi_sigma[:, cv_indx == i] @ Y_phi_sigma[:, cv_indx == i].T)
                h_cv.append(np.sum(K_sigma[:, cv_indx == i], axis = -1))

            for i in fold_inds:
                H_cv_tr = np.sum(np.array(H_cv_xphi)[fold_inds!=i], axis=0) *\
                          np.sum(np.array(H_cv_yphi)[fold_inds!=i], axis=0) \
                          / np.sum(n_cv[fold_inds!=i]**2)
                H_cv_te = H_cv_xphi[i]* H_cv_yphi[i] / (n_cv[i]**2)
                h_cv_tr = np.mean(np.array(h_cv)[fold_inds != i], axis = 0).reshape(-1,1) / np.sum(n_cv[fold_inds != i])
                h_cv_te = h_cv[i].reshape(-1,1) / n_cv[i]

                for j, alpha in enumerate(alpha_pool):
                    H_cv_tr += alpha * np.eye(num_centers)
                    theta_cv = np.linalg.solve(H_cv_tr, h_cv_tr)
                    wh_cv = 0.5*(theta_cv.T @ H_cv_te @ theta_cv) - h_cv_te.T @ theta_cv
                    scores_cv[n,j] = scores_cv[n,j] + wh_cv/num_fold

        sigma_ind, alpha_ind = np.unravel_index(np.argmin(scores_cv, axis=None), scores_cv.shape)
        score_cv = scores_cv[sigma_ind, alpha_ind]
        rbf_sigma = sigma_pool[sigma_ind]
        alpha = alpha_pool[alpha_ind]

    # print(rbf_sigma, alpha)
    X_phi = np.exp(- X_dist / (2 * rbf_sigma**2))
    Y_phi = np.exp(- Y_dist / (2 * rbf_sigma**2))
    K = X_phi * Y_phi # [C x M]
    H = (X_phi @ X_phi.T) * (Y_phi @ Y_phi.T) / (M ** 2) # [C x C]
    H = H + alpha * np.eye(num_centers)
    h = np.mean(K, axis=-1).reshape(-1, 1) # [C, 1]

    # Solve the objective equation
    theta = np.linalg.solve(H, h) # [C x 1]

    # TODO: Use coordinate descent or any gradient-based solver to solve the above equation

    SMI = 0.5 * (h.T @ theta) - 0.5
    return SMI, score_cv
