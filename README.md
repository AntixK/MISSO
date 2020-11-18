MISSO


### Installation
 - Using `pip`
```

```
**Note:** In our benchmarks, multi-core version was always faster than the GPU accelerated version. So, we
highly recommend installing just the CPU version and using multi-core computation.

- Installing from source
```

```
### Usage

### Benchmarks

** Benchmarks were run on a machine with the following configuration
```
CPU:       6 core Intel Core i7-8750H (-MT-MCP-) [12 core with Hyperthreading]
           arch: Skylake rev.10 cache: 9216 KB
           flags: (lm nx sse sse2 sse3 sse4_1 sse4_2 ssse3 vmx) bmips: 26399
           clock speeds: max: 4100 MHz 1: 2479 MHz 2: 3013 MHz 3: 3211 MHz
           4: 3098 MHz 5: 3362 MHz 6: 3769 MHz 7: 3082 MHz 8: 3290 MHz
           9: 3090 MHz 10: 3141 MHz 11: 3055 MHz 12: 3650 MHz
Graphics:  Card-1: Intel Device 3e9b bus-ID: 00:02.0
           Card-2: NVIDIA Device 1f10 bus-ID: 01:00.0
           Display Server: x11 (X.Org 1.19.6 )
           drivers: modesetting,nvidia (unloaded: fbdev,vesa,nouveau)
           Resolution: 3840x1600@59.99hz
           OpenGL: renderer: GeForce RTX 2070 with Max-Q Design/PCIe/SSE2
           version: 4.6.0 NVIDIA 440.100 Direct Render: Yes
```
### TODO
- [ ] Try gradient-based solvers 
    - [ ] Conjugate-gradient descent
- [x] Multi-processing for `lsmi` computation
    - [x] Reduce interprocess overhead
    - [x] Try other methods to parallelize the code
- [ ] Benchmarks
    - [x] Multiprocessing
    - [x] GPU benchmarks
    - [ ] Solver Benchmarks
    - [ ] Run benchmarks on multiple machines and put in benchmark reports
- [ ] Detailed comparison with graphical Lasso (Tutorials)
    - [ ] Toy Example
    - [ ] Time Series: Stationary & Dynamic [link](https://watermark.silverchair.com/bhs352.pdf)
    - [ ] Comparison of MISSO and GLASSO on indirect coupling [link](https://watermark.silverchair.com/btr638.pdf)
- [ ] GPU Acceleration    
    - [x] Use Cupy for solving
    - [ ] Reduce GPU overhead 
    - [ ] Verify correctness
    - [x] Check if pytorch is faster
- [x] tqdm for Notebook and Script
- [ ] Pandas DataFrame support
- [ ] Packaging
    - [ ] pip package
    - [ ] Travis CI
    - [ ] Automatically detect and install GPU version
- [ ] Readme

