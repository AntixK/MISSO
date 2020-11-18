import sys
sys.path.append('../')
from time import time
import numpy as np
# from misso import MISSO as misso_cpu
from misso_gpu import MISSO as misso_gpu
import matplotlib.pyplot as plt
plt.style.use('seaborn')
import cProfile

np.set_printoptions(precision=2)

def get_data(N, M:int = 5):
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
    g = misso_gpu(verbose=False, mp=False)
    g.fit(X)

cProfile.run("run()")



