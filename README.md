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
    - [x] Reduce inter-process overhead
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
- [ ] Study the MST-based construction for the graph [link](https://arxiv.org/pdf/1703.00485.pdf)
- [ ] Pandas DataFrame support
- [ ] Packaging
    - [x] pip package
    - [ ] Travis CI
- [ ] Readme

### Extending to a full library
- [ ] HSIC
- [ ] Partial Correlation
- [ ] [PMI](https://www.pnas.org/content/113/18/5130.short)
- [ ] CMI 
- [ ] Partial Mutual Information
- [ ] Rename MISSO to MIM
- [ ] l1-LSMI
- [ ] LSIR
- [ ] [PIDC](https://www.cell.com/cell-systems/pdfExtended/S2405-4712(17)30386-1) (Optional)


### References
- [https://github.com/dit/dit](https://github.com/dit/dit)
