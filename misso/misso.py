import numpy as np
import threading
from graphviz import Graph
from typing import Optional, List
import matplotlib.pyplot as plt
from .lsmi import lsmi1D

class MISSO:
    def __init__(self,
                 num_centers:Optional[int] = 200,
                 rbf_sigma: Optional[float] = None,
                 alpha: Optional[float] =  None) -> None:
        """

        :param num_centers: Number of centers to use when computing the RBF kernel
        :param rbf_sigma: Length-scale for the RBF kernel
        :param alpha: L2 regularizer weight for the LSMI
        """
        self.num_centers = num_centers
        self.rbf_sigma = rbf_sigma
        self.alpha = alpha

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

        MIM = np.zeros((N, N))
        for i in range(N):
            for j in range(i + 1):

                x, y = X[:, i].reshape(-1,1), Y[:, j].reshape(-1,1)
                smi,  _ = lsmi1D(x, y,
                             num_centers=self.num_centers,
                             rbf_sigma=self.rbf_sigma,
                             alpha=self.alpha)
                MIM[i,j] = smi
                MIM[j,i] = smi # MI is symmetric

        return MIM
    def show_graph(self, M:np.ndarray, threshold:float, node_labels:List, title:str) -> None:

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


# if __name__ == '__main__':
#     import matplotlib.pyplot as plt
#     # plt.style.use('ggplot')
#     np.set_printoptions(precision=2)
#     t = np.random.uniform(-10, 10, (100, 1))
#     #
#     y = np.sin(t/10 * np.pi)
#     X = y
#
#     y = np.cos(t / 10 * np.pi)
#     X = np.hstack([X, y])
#     y = np.random.uniform(-1, 1, (100, 1))
#     X = np.hstack([X, y])
#     y = np.sin(t/10 * np.pi + 2 * np.pi / 3.)
#     X = np.hstack([X, y])
#     y = np.cos(t / 10 * np.pi - 2 * np.pi / 3.)
#     X = np.hstack([X, y])
#
#     y = np.vstack([-np.ones((50, 1)), np.ones((50, 1))])
#     X = np.hstack([X, y])
#
#     print(X.shape)
#
#     g = MISSO()
#     m = g.fit(X)
#     print(m)
#
#     g.show_matrix(m, xlabels = [r'$sin(\pi t/10)$',
#                                  r'$Cos(\pi t/10)$',
#                                  r'$U(0,1)$',
#                                  r'$sin(\frac{\pi t}{10} + \frac{2\pi}{3})$',
#                                  r'$cos(\frac{\pi t}{10} - \frac{2\pi}{3})$',
#                                  r'Sign(t)'])

