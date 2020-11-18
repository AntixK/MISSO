import torch
import numpy as np
from typing import Optional, Tuple
from torch.nn.utils.rnn import pad_sequence

def lsmi1D(X:torch.tensor,
           Y:Optional[torch.tensor] = None,
           num_centers:Optional[int] = 200,
           rbf_sigma:Optional[float] = None,
           alpha: Optional[float] =  None,
           random_seed:int = 42,
           verbose:bool=False) -> Tuple:
    """
    Compute the LSMI estimate of the Mutualk information between two 1Dcrandom variables
    X and Y, each with M samples.
    :param X: [M, 1] Vector of samples
    :param Y: [M, 1] Vector of samples
    :param num_centers: Number of centers to use when computing RBF kernel
    :param rbf_sigma: Length-scale parameters for the RBF kernel
    :param alpha: L2 regularizer for the LSMI estimator
    :param random_seed:
    :param verbose:
    :return: A tuple of LSMI estimate and the cross validation score

    References:
    [1] http://www.ms.k.u-tokyo.ac.jp/software.html#LSMI

    """

    np.random.seed(random_seed)
    torch.manual_seed(random_seed)
    torch.cuda.manual_seed(random_seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    if Y is None:
        Y = X

    M, D = X.shape
    My, Dy = Y.shape

    assert M == My, "Both X & Y must have the same number of samples (dim 1)"
    assert D == Dy, "Both X & Y must have the same dimension (dim 2)"

    num_centers = min(num_centers, M)

    # Randomly choose the centres for Gaussian Kernels
    centers = np.random.choice(M, size=num_centers, replace=False)
    X_norm = torch.sum(X**2, dim=-1) # [C,]
    Y_norm = torch.sum(Y**2, dim=-1) # [C,]

    # Compute the norm with respect to the centers
    X_dist = X_norm.repeat(num_centers, 1) + \
             X_norm[None, centers].t().repeat(1, M) -\
             2 * X[centers, :] @ X.t() # [C x M]
    # X_dist = np.tile(X_norm, (num_centers, 1)) + \
    #          np.tile(X_norm[None, centers].T, (1, M)) - \
    #          2 * X[centers, :] @ X.T       # [C x M]
    Y_dist = Y_norm.repeat(num_centers, 1) + \
             Y_norm[None, centers].t().repeat(1, M) -\
             2 * Y[centers, :] @ Y.t() # [C x M]
    # Y_dist = np.tile(Y_norm, (num_centers, 1)) + \
    #          np.tile(Y_norm[None, centers].T, (1, M)) - \
    #          2 * Y[centers, :] @ Y.T       # [C x M]

    score_cv = torch.tensor(float('-inf'))

    if rbf_sigma is None or alpha is None:
        if verbose:
            print("Cross validating...")
        # perform cross-validation
        sigma_pool = torch.logspace(-2, 2, 9)
        alpha_pool = torch.logspace(-3, 1, 9)

        scores_cv = torch.zeros((len(sigma_pool), len(alpha_pool)))

        num_fold = 5
        cv_indx = torch.floor(torch.arange(M)*(num_fold/M))
        cv_indx = cv_indx[torch.randperm(M)]
        # np.random.shuffle(cv_indx)
        fold_inds = np.arange(num_fold)

        n_cv = []
        for i in fold_inds:
            n_cv.append(torch.sum(cv_indx == i))

        n_cv = torch.tensor(n_cv)
        for n, sigma in enumerate(sigma_pool):
            X_phi_sigma = torch.exp(- X_dist/ (2 * sigma ** 2)) # [C x M]
            Y_phi_sigma = torch.exp(- Y_dist/ (2 * sigma ** 2)) # [C x M]
            K_sigma = X_phi_sigma * Y_phi_sigma

            H_cv_xphi = []
            H_cv_yphi = []
            h_cv = []
            for k in fold_inds:
                H_cv_xphi.append(X_phi_sigma[:, cv_indx == k] @ X_phi_sigma[:, cv_indx == k].T)
                H_cv_yphi.append(Y_phi_sigma[:, cv_indx == k] @ Y_phi_sigma[:, cv_indx == k].T)
                h_cv.append(torch.sum(K_sigma[:, cv_indx == k], dim = -1))

            for i in fold_inds:
                # print(i)
                H_cv_xphi= pad_sequence(H_cv_xphi).transpose(0,1)
                H_cv_yphi= pad_sequence(H_cv_yphi).transpose(0,1)
                h_cv = pad_sequence(h_cv).transpose(0,1)

                H_cv_tr = torch.sum(H_cv_xphi[fold_inds!=i], dim=0) *\
                          torch.sum(H_cv_yphi[fold_inds!=i], dim=0) \
                          / torch.sum(n_cv[fold_inds!=i]**2)
                H_cv_te = H_cv_xphi[i]* H_cv_yphi[i] / (n_cv[i]**2)
                h_cv_tr = torch.mean(h_cv[fold_inds != i], dim = 0).view(-1,1) \
                          / torch.sum(n_cv[fold_inds != i])
                h_cv_te = h_cv[i].view(-1,1) / n_cv[i]

                for j, alpha in enumerate(alpha_pool):
                    H_cv_tr += alpha * torch.eye(num_centers)
                    theta_cv, _ = torch.solve(h_cv_tr, H_cv_tr)
                    wh_cv = 0.5*(theta_cv.t() @ H_cv_te @ theta_cv) - h_cv_te.t() @ theta_cv
                    # print((wh_cv/num_fold).shape)
                    scores_cv[n,j] = scores_cv[n,j] + (wh_cv/num_fold).item()


        sigma_ind, alpha_ind = np.unravel_index(torch.argmin(scores_cv).item(), list(scores_cv.size()))
        # sigma_ind, alpha_ind = np.unravel_index(torch.argmin(scores_cv, dim=None, keepdim=True), scores_cv.shape)
        score_cv = scores_cv[sigma_ind, alpha_ind]
        rbf_sigma = sigma_pool[sigma_ind]
        alpha = alpha_pool[alpha_ind]

    # print(rbf_sigma, alpha)
    X_phi = torch.exp(- X_dist / (2 * rbf_sigma**2))
    Y_phi = torch.exp(- Y_dist / (2 * rbf_sigma**2))
    K = X_phi * Y_phi # [C x M]
    H = (X_phi @ X_phi.t()) * (Y_phi @ Y_phi.t()) / (M ** 2) # [C x C]
    H = H + alpha * torch.eye(num_centers)
    h = torch.mean(K, dim=-1).view(-1, 1) # [C, 1]

    # Solve the objective equation

    theta = torch.solve(h, H)[0].numpy()
    # TODO: Use coordinate descent or any gradient-based solver to solve the above equation

    SMI = 0.5 * (h.t() @ theta) - 0.5
    return SMI, score_cv

if __name__ == '__main__':
    # np.random.seed(2)
    # np.set_printoptions(precision=2)
    x = torch.randint(-10, 10, (100, 1))
    y = torch.randint(-1, 1, (100, 1))
    # y = np.sin(x/10 * np.pi)

    print(lsmi1D(x))
