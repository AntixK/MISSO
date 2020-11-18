from time import time
import numpy as np
from misso import MISSO as misso_cpu
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

    g = misso_cpu(verbose=False, mp=True)
    # s = time()
    m_g = g.fit(X)
    # gpu_time.append(time() - s)
    assert np.allclose(m_g, m_c), f"Error! {m_g - m_c}"

print("Starting Benchmark...")
cpu_log, mp_log = [], []
for n in sizes:
    print(f"Running size: {n}")
    X = get_data(n)
    # print(X.shape)
    cpu_time, mp_time = [], []
    for _ in range(5):
        g = misso_cpu(verbose=False, mp=False)
        s = time()
        m_c = g.fit(X)
        cpu_time.append(time() - s)

        g = misso_cpu(verbose=False, mp=True)
        s = time()
        m_g = g.fit(X)
        mp_time.append(time() - s)
        # assert np.allclose(m_g, m_c), f"Error! {m_g - m_c}"
    cpu_log.append([np.mean(cpu_time), np.std(cpu_time)])
    mp_log.append([np.mean(mp_time), np.std(mp_time)])

cpu_log, mp_log = np.array(cpu_log), np.array(mp_log)
x = np.arange(len(sizes))
width = 0.35
fig, ax = plt.subplots(figsize=(12, 5))
rects1 = ax.bar(x - width/2, cpu_log[:, 0], width, yerr = cpu_log[:, 1], label='Single Core')
rects2 = ax.bar(x + width/2, mp_log[:, 0], width, yerr = mp_log[:, 1], label='Multi Core')

# Add some text for labels, title and custom x-axis tick labels, etc.
ax.set_ylabel('Time (s)')
ax.set_xlabel('# of Samples')
ax.set_title(fr'Toy Multiprocessing Benchmark (# samples $\times {K}$)')
ax.set_xticks(x)
ax.set_xticklabels(sizes)
ax.legend()
fig.tight_layout()

plt.savefig("Multiprocessing Benchmark.png", dpi=300)
plt.show()

