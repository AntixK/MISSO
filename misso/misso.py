"""
Copyright (c) 2020, Anand K Subramanian.
All rights reserved.
"""
import os
import shutil
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
                 mp:bool = None) -> None:
        """

        :param num_centers: Number of centers to use when computing the RBF kernel
        :param rbf_sigma: Length-scale for the RBF kernel
        :param alpha: L2 regularizer weight for the LSMI
        :param verbose: Boolean to display computation progress
        :param random_seed: Integer seed for reproducibility (default: 42)
        :param mp: Boolean to use multiprocessing. If `None`, will use multprocessing is the
                   current device has multiple cores.
        """
        self.num_centers = num_centers
        self.rbf_sigma = rbf_sigma
        self.alpha = alpha

        if mp is None:
            self.use_mp = os.cpu_count() >= 4
        else:
            self.use_mp = mp

        if self.use_mp:
            self.folder = "./joblib_memmap"

        self.verbose = verbose

        if self.verbose:
            if self.use_mp:
                self.logger = logging.getLogger()
                self.logger.setLevel(logging.INFO)
                self.info = logging.info

            else:
                self.logger = logging.getLogger()
                self.logger.setLevel(logging.INFO)
                self.info = logging.info

        np.random.seed(random_seed)
        self.random_seed = random_seed

    def compute_smi(self, *args):
        mim, x, y, i, j = args
        if self.verbose:
            self.info(f"Computing SMI for [{i}, {j}]")

        smi, _ = lsmi1D(x, y,
                        num_centers=self.num_centers,
                        rbf_sigma=self.rbf_sigma,
                        alpha=self.alpha,
                        random_seed = self.random_seed)
        mim[i, j] = smi
        mim[j, i] = smi

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
            os.makedirs(self.folder, exist_ok=True)
            shared_mimfile = os.path.join(self.folder, 'mim_memmap')
            shared_mim = np.memmap(shared_mimfile, dtype=np.float, shape=(N,N), mode='w+')

            if not self.verbose:
                Parallel(n_jobs = os.cpu_count())(
                                delayed(self.compute_smi)(shared_mim, *p_arg) for p_arg in tqdm(process_args, desc='Computing MIM'))
            else:
                Parallel(n_jobs=os.cpu_count())(
                    delayed(self.compute_smi)(shared_mim, *p_arg) for p_arg in process_args)

            self.MIM = np.array(shared_mim)
            shutil.rmtree(self.folder)

        else: # Sequential Processing
            self.MIM = np.zeros((N, N))

            if self.verbose:
                pbar = process_args
            else:
                pbar = tqdm(process_args, desc = 'Computing MIM')

            for args in pbar:
                self.compute_smi(self.MIM, *args)

        return self.MIM

    def show_graph(self,
                   M:np.ndarray,
                   threshold:float,
                   node_labels:List,
                   title:str) -> Graph:
        """

        :param M:
        :param threshold:
        :param node_labels:
        :param title:
        :return:
        """

        g = Graph('G', filename=title+'.gv', engine='dot')
        M = np.round(M, 3)
        for i in range(M.shape[0]):
            for j in range(i + 1):
                if(M[i, j] >= threshold and i!= j):
                    g.edge(node_labels[i], node_labels[j], label=str(M[i,j]))

        return g

    def show_matrix(self,
                    M:np.ndarray,
                    xlabels: List,
                    ylabels: List = None):
        """

        :param M:
        :param xlabels:
        :param ylabels:
        :return:
        """

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
