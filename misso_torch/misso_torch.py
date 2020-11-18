import os
import shutil
import torch
import numpy as np
from graphviz import Graph
import matplotlib.pyplot as plt
from typing import Optional, List

import logging

from tqdm import tqdm
from joblib import Parallel, delayed

from .lsmi import lsmi1D

class MISSO:
    def __init__(self,
                 num_centers:Optional[int] = 200,
                 rbf_sigma: Optional[float] = None,
                 alpha: Optional[float] =  None,
                 verbose:bool = False,
                 random_seed:int = 42,
                 mp:bool = None,
                 device:str=None) -> None:
        """

        :param num_centers: Number of centers to use when computing the RBF kernel
        :param rbf_sigma: Length-scale for the RBF kernel
        :param alpha: L2 regularizer weight for the LSMI
        :param verbose: Boolean to display computation progress
        :param random_seed: Integer seed for reproducibility (default: 42)
        """
        self.num_centers = num_centers
        self.rbf_sigma = rbf_sigma
        self.alpha = alpha

        self.device = device

        if mp is None:
            self.use_mp = os.cpu_count() >= 4
        else:
            self.use_mp = mp

        if self.use_mp:
            self.folder = "./joblib_memmap"

        self.verbose = verbose

        if self.verbose:
            self.logger = logging.getLogger()
            self.logger.setLevel(logging.INFO)
            self.info = logging.info

        np.random.seed(random_seed)
        torch.manual_seed(random_seed)
        torch.cuda.manual_seed(random_seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

        self.random_seed = random_seed

    def compute_smi(self, *args):
        mim, x,y, i, j = args

        if self.verbose:
            self.info(f"Computing SMI for [{i}, {j}]")

        smi, _ = lsmi1D(x, y,
                        num_centers=self.num_centers,
                        rbf_sigma=self.rbf_sigma,
                        alpha=self.alpha,
                        random_seed = self.random_seed,
                        verbose=self.verbose)
        mim[i, j] = smi.item()
        mim[j, i] = smi.item()


        if self.verbose:
            self.info(f"Finished SMI for [{i}, {j}]")


    def fit(self,
            X:np.ndarray,
            Y:Optional[np.ndarray] = None) -> np.ndarray:
        """
        Computes the sparse mutual information matrix using the LSMI-LASSO method.

        :param X: [M x N] Set of N random variables with M samples each
        :param Y: [N x M] Set of N random variables with M samples each. Default: None (Will use X)
        :return:  [N x N] Sparse Mutual Information Matrix
        """

        if Y is None:
            Y = X

        M, N = X.shape
        My, Ny = Y.shape

        assert M == My, "Both X & Y must have the same number of samples (dim 1)"
        assert N == Ny, "Both X & Y must have the same # of random variables (dim 2)"

        self.N = N

        X, Y = torch.from_numpy(X), torch.from_numpy(Y)

        if self.device is not None:
            X.to(self.device)
            Y.to(self.device)
        process_args = [(X[:, i].reshape(-1, 1), Y[:, j].reshape(-1, 1), i, j)
                        for i in range(N) for j in range(i + 1)]

        if self.use_mp:  # Multiprocessing Code
            os.makedirs(self.folder, exist_ok=True)
            shared_mimfile = os.path.join(self.folder, 'mim_memmap')
            shared_mim = np.memmap(shared_mimfile, dtype=np.float, shape=(N, N), mode='w+')

            if not self.verbose:
                Parallel(n_jobs=os.cpu_count())(
                    delayed(self.compute_smi)(shared_mim, *p_arg) for p_arg in tqdm(process_args, desc='Computing MIM'))
            else:
                Parallel(n_jobs=os.cpu_count())(
                    delayed(self.compute_smi)(shared_mim, *p_arg) for p_arg in process_args)

            self.MIM = np.array(shared_mim)
            shutil.rmtree(self.folder)

        else:  # Sequential Processing
            self.MIM = np.zeros((N, N))

            if self.verbose:
                pbar = process_args
            else:
                pbar = tqdm(process_args, desc='Computing MIM')

            for args in pbar:
                self.compute_smi(self.MIM, *args)

        return self.MIM

    def show_graph(self, M:np.ndarray, threshold:float, node_labels:List, title:str) -> Graph:

        g = Graph('G', filename=title+'.gv', engine='dot')
        M = np.round(M, 3)
        for i in range(M.shape[0]):
            for j in range(i + 1):
                if(M[i, j] >= threshold and i!= j):
                    g.edge(node_labels[i], node_labels[j], label=str(M[i,j]))

        return g

    def show_matrix(self, M:np.ndarray, xlabels: List, ylabels: List = None):

        if ylabels is None:
            ylabels = xlabels

        fig, ax = plt.subplots()
        im = ax.matshow(M, cmap=plt.cm.summer)
        plt.xticks(np.arange(0, M.shape[0]), xlabels)
        plt.yticks(np.arange(0, M.shape[1]), ylabels)

        plt.colorbar(im)
        for i in range(M.shape[0]):
            for j in range(M.shape[1]):
                c = np.round(M[i, j], 3)
                ax.text(i, j, str(c), va='center', ha='center')
        plt.grid(False)
        plt.show()

def init_shared_data(shared_array):
    """
    Multiprocessing requires the shared memory to be
    inherited and not passed. This is the simplest and most
    elegant way I could come up with.

    :param shared_array: The array to be shared across the processes
    :return: A global instance of the array
    """
    global shared_MIM
    shared_MIM = shared_array

if __name__ == '__main__':
    # import matplotlib.pyplot as plt
    from time import time
    # plt.style.use('ggplot')

    # np.set_printoptions(precision=2)
    import numpy as n
    t = n.random.uniform(-10, 10, (800, 1))
    #
    y = n.sin(t/10 * np.pi)
    X = y

    y = n.cos(t / 10 * np.pi)
    X = n.hstack([X, y])
    y = n.random.uniform(-1, 1, (800, 1))
    X = n.hstack([X, y])
    y = n.sin(t/10 * np.pi + 2 * np.pi / 3.)
    X = n.hstack([X, y])
    y = n.cos(t / 10 * np.pi - 2 * np.pi / 3.)
    X = n.hstack([X, y])
    # X = np.hstack([X, y])
    # X = np.hstack([X, y])
    # X = np.hstack([X, y])
    # X = np.hstack([X, y])
    # X = np.hstack([X, y])
    # X = np.hstack([X, y])
    # X = np.hstack([X, y])
    # X = np.hstack([X, y])
    # X = np.hstack([X, y])
    # X = np.hstack([X, y])
    # X = np.hstack([X, y])
    # X = np.hstack([X, y])
    # X = np.hstack([X, y])
    # X = np.hstack([X, y])
    # X = np.hstack([X, y])
    # X = np.hstack([X, y])
    # X = np.hstack([X, y])
    # X = np.hstack([X, y])
    # X = np.hstack([X, y])
    # X = np.hstack([X, y])

    # y = np.vstack([-np.ones((50, 1)), np.ones((50, 1))])
    # X = np.hstack([X, y])

    print(X.shape)

    # g = MISSO(verbose=False, mp=False)
    # s = time()
    # m_s = g.fit(X)
    # print(f"Elapsed time: {time() - s}")


    # g = MISSO(verbose=True, mp=False)
    g = MISSO(verbose=False, mp=False)
    s = time()
    m_p = g.fit(X)
    print(f"Elapsed time: {time() - s}")

#
#     g.show_matrix(m, xlabels = [r'$sin(\pi t/10)$',
#                                  r'$Cos(\pi t/10)$',
#                                  r'$U(0,1)$',
#                                  r'$sin(\frac{\pi t}{10} + \frac{2\pi}{3})$',
#                                  r'$cos(\frac{\pi t}{10} - \frac{2\pi}{3})$',
#                                  r'Sign(t)'])

