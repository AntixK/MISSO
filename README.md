MISSO


### Installation
 - Using `pip`
```
pip install misso
```
**Note:** In our benchmarks, multi-core version was always faster than the GPU accelerated version. So, we
highly recommend installing just the CPU version and using multi-core computation.

- Installing from source
```

```

### Usage
```
from misso import MISSO


```
For a more detailed usage, check out the Tutorials folder.

### Benchmarks

|       **Benchmark Results**            |              **CPU**               |      **OS**                |
|----------------------------------------|------------------------------------|----------------------------|
|                                        |   `6 core Intel Core i7-8750H`     |   `Ubuntu 18.04`           |
|                                        |   `4 core Intel Core i7-8569U`     |   `macOS Catalina 10.15.7` |
|                                        |                                    |                            |


### License




### TODO
- [ ] Try gradient-based solvers 
    - [ ] Conjugate-gradient descent
- [ ] Make it compatible with Sklearn
- [x] Multi-processing for `lsmi` computation
    - [x] Reduce interprocess overhead
    - [x] Try other methods to parallelize the code
- [ ] Benchmarks
    - [x] Multiprocessing
    - [x] GPU benchmarks
    - [ ] Solver Benchmarks
    - [ ] Run benchmarks on multiple machines and put in benchmark reports
- [ ] Tutorials
    - [ ] Toy Example comaparing with GLASSO
    - [ ] Comaprison with Partial Mutual Information planar graphs
    - [ ] Iter-operability with sklearn and pandas in a notebook
    - [ ] Time Series: Stationary & Dynamic [link](https://academic.oup.com/cercor/article-pdf/24/3/663/14099596/bhs352.pdf)
    - [ ] Comparison of MISSO and GLASSO on indirect coupling [link](https://academic.oup.com/bioinformatics/article-pdf/28/2/184/16908913/btr638.pdf)
- [ ] GPU Acceleration    
    - [x] Use Cupy for solving
    - [x] Reduce GPU overhead 
    - [ ] Verify correctness (Still an issue)
    - [x] Try torch for GPU acceleration
- [x] tqdm for Notebook and Script
- [ ] Pandas DataFrame support
- [ ] Packaging
    - [x] pip package
    - [ ] Travis CI
- [ ] Readme

### Extending to a full library
- [ ] HSIC
- [ ] PMI
- [ ] CMI
- [ ] Rename MISSO to MIM
- [ ] l1-LSMI
- [ ] LSIR