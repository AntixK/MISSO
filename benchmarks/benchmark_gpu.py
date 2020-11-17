from time import time
import numpy as np
from misso import MISSO as misso_cpu
from misso_gpu import MISSO as misso_gpu
import matplotlib.pyplot as plt
plt.style.use('seaborn')

# np.set_printoptions(precision=2)

K = 15
def get_data(N, M:int = K):
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

cpu_time, gpu_time = [], []
sizes = [100, 500, 1000, 1500, 2000, 3000, 4000, 6000, 10000, 15000]

print("Running warmup iterations...")
# Perform warmup iterations
for _ in range(3):
    X = get_data(100)
    # print(X.shape)
    g = misso_cpu(verbose=False, mp=False)
    # s = time()
    m_c = g.fit(X)
    # cpu_time.append(time() - s)

    g = misso_gpu(verbose=False, mp=False)
    # s = time()
    m_g = g.fit(X)
    # gpu_time.append(time() - s)
    # assert np.allclose(m_g, m_c), f"Error! {m_g - m_c}"

print("Starting Benchmark...")
cpu_log, gpu_log = [], []
for n in sizes:
    print(f"Running size: {n}")
    X = get_data(n)
    # print(X.shape)
    cpu_time, gpu_time = [], []
    for _ in range(5):
        g = misso_cpu(verbose=False, mp=False)
        s = time()
        m_c = g.fit(X)
        cpu_time.append(time() - s)

        g = misso_gpu(verbose=False, mp=False)
        s = time()
        m_g = g.fit(X)
        gpu_time.append(time() - s)
        # assert np.allclose(m_g, m_c), f"Error! {m_g - m_c}"
    cpu_log.append([np.mean(cpu_time), np.std(cpu_time)])
    gpu_log.append([np.mean(gpu_time), np.std(gpu_time)])

cpu_log, gpu_log = np.array(cpu_log), np.array(gpu_log)
x = np.arange(len(sizes))
width = 0.35
fig, ax = plt.subplots(figsize=(12, 5))
rects1 = ax.bar(x - width/2, cpu_log[:, 0], width, yerr = cpu_log[:, 1], label='CPU')
rects2 = ax.bar(x + width/2, gpu_log[:, 0], width, yerr = gpu_log[:, 1], label='GPU')

# Add some text for labels, title and custom x-axis tick labels, etc.
ax.set_ylabel('Time (s)')
ax.set_xlabel('# of Samples')
ax.set_title(fr'Toy GPU Benchmark [Single Core] (# samples $\times {K}$)')
ax.set_xticks(x)
ax.set_xticklabels(sizes)
ax.legend()
fig.tight_layout()

plt.savefig("GPU Benchmark_SingleCore.png", dpi=300)
plt.show()

