import sys
sys.path.append('../')
from time import time
import numpy as np
from misso import MISSO as misso_cpu
# from misso_gpu import MISSO as misso_gpu
from misso_torch import MISSO as misso_t
import matplotlib.pyplot as plt
plt.style.use('seaborn')
import cProfile


np.set_printoptions(precision=4)

def get_data(N, M:int = 25):
    t = np.random.uniform(-10, 10, (N, 1))
    y = np.sin(t/10 * np.pi)
    X = y

    y = np.cos(t / 10 * np.pi)
    X = np.hstack([X, y])
    y = np.random.uniform(-1, 1, (N, 1))
    X = np.hstack([X, y])
    y = np.sin(t/10 * np.pi + 2 * np.pi / 3.)
    X = np.hstack([X, y])
    y = np.cos(t / 10 * np.pi - 2 * np.pi / 3.)

    for _ in range(M-4):
        X = np.hstack([X, y])
    return X

def run():
    X = get_data(1500)
    print(X.shape)
    g = misso_t(verbose=False, mp=True, device ='cuda')
    m_c = g.fit(X)

    g = misso_t(verbose=False, mp=True, device = None)
    m_t = g.fit(X)

    # g = misso_cpu(verbose=False, mp=True)
    # m_p = g.fit(X)
    #
    # g = misso_gpu(verbose=False, mp=False)
    # m_g = g.fit(X)

    assert np.allclose(m_c, m_t), f"Error: {(m_c - m_t).max()}"
    # assert np.allclose(m_c, m_p), f"Error: {m_c - m_p}"
    # assert np.allclose(m_c, m_g), f"Error: {m_c - m_g}"

run()
