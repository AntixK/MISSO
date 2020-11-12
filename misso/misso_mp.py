import os
import numpy as np
from graphviz import Graph
import matplotlib.pyplot as plt
from typing import Optional, List

import logging
from contextlib import closing

from tqdm import tqdm
import multiprocessing
from multiprocessing import Pool
from multiprocessing import sharedctypes

from lsmi import lsmi1D

class MISSO:
    def __init__(self,
                 num_centers:Optional[int] = 200,
                 rbf_sigma: Optional[float] = None,
                 alpha: Optional[float] =  None,
                 verbose:bool = False,
                 random_seed:int = 42,
                 mp:bool = None) -> None:
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

        if mp is None:
            self.use_mp = os.cpu_count() >= 4
        else:
            self.use_mp = mp

        self.verbose = verbose

        if self.verbose:
            if self.use_mp:
                self.logger = multiprocessing.log_to_stderr()
                self.logger.setLevel(logging.INFO)
                self.info = multiprocessing.get_logger().info
            else:
                self.logger = logging.getLogger()
                self.logger.setLevel(logging.INFO)
                self.info = logging.info

        np.random.seed(random_seed)
        self.random_seed = random_seed

    def compute_smi(self, args):
        x,y, i, j = args

        if self.verbose:
            self.info(f"Computing SMI for [{i}, {i}]")

        if self.use_mp:
            # Get the local instance of the shared memory
            # Note that we don't need lock here as the
            # parts accessed by each process are independent

            local_mim = np.ctypeslib.as_array(shared_MIM)

            smi, _ = lsmi1D(x, y,
                            num_centers=self.num_centers,
                            rbf_sigma=self.rbf_sigma,
                            alpha=self.alpha,
                            random_seed = self.random_seed)
            local_mim[i, j] = smi
            local_mim[j, i] = smi
        else:
            smi, _ = lsmi1D(x, y,
                            num_centers=self.num_centers,
                            rbf_sigma=self.rbf_sigma,
                            alpha=self.alpha,
                            random_seed = self.random_seed)
            self.MIM[i, j] = smi
            self.MIM[j, i] = smi  # MI is symmetric

        if self.verbose:
            self.info(f"Finished SMI for [{i}, {i}]")

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
        process_args = [(X[:, i].reshape(-1, 1), Y[:, j].reshape(-1, 1), i, j)
                        for i in range(N) for j in range(i + 1)]

        if self.use_mp: # Multiprocessing Code
            MIM = np.ctypeslib.as_ctypes(np.zeros((N, N)))
            shared_MIM = sharedctypes.RawArray(MIM._type_, MIM)

            with closing(Pool(initializer=init_shared_data, initargs=(shared_MIM,))) as p:
                if self.verbose:
                    p.map_async(self.compute_smi, process_args)
                else:
                    with tqdm(total=len(process_args), desc="Computing MIM") as pbar:
                        for i, _ in enumerate(p.imap_unordered(self.compute_smi, process_args)):
                            pbar.update()
            p.join()
            MIM = np.ctypeslib.as_array(shared_MIM)
            self.MIM = MIM

        else: # Sequential Processing
            self.MIM = np.zeros((N, N))

            if self.verbose:
                pbar = process_args
            else:
                pbar = tqdm(process_args, desc = 'Computing MIM')

            for args in pbar:
                self.compute_smi(args)

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

    np.set_printoptions(precision=2)
    t = np.random.uniform(-10, 10, (100, 1))
    #
    y = np.sin(t/10 * np.pi)
    X = y

    y = np.cos(t / 10 * np.pi)
    X = np.hstack([X, y])
    y = np.random.uniform(-1, 1, (100, 1))
    X = np.hstack([X, y])
    y = np.sin(t/10 * np.pi + 2 * np.pi / 3.)
    X = np.hstack([X, y])
    y = np.cos(t / 10 * np.pi - 2 * np.pi / 3.)
    X = np.hstack([X, y])
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

    g = MISSO(verbose=False, mp=False)
    s = time()
    m_s = g.fit(X)
    print(f"Elapsed time: {time() - s}")


    g = MISSO(verbose=False, mp=True)
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

