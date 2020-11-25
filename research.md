# Research Topic

## Abstract
Part-Mutual Information (PMI) has been an effective alternative for finding *direct* non-linear associations between random variables, in place of conditional mutual information (CMI). Another alternate would be the causal strength (CS) as defined in [link](http://proceedings.mlr.press/v108/wang20i/wang20i.pdf). But PMI is more widely used owing to its symmetric nature and the non-requirement of the full knowledge of the causal DAG.

However, a major issue with using PMI in practice it is computability. Usual usage often assumes a aussian case, which goes against the fundamental aspect of mutual information being independent of the underlying distribution. So, this research focuses on whether there is a method (parametric or otherwise) to compute PMI without underlying assumption.

Clearly, direct computation is NP-Hard and approximation has been the go-to method. Some employ the "bins"-based approximation, while others employ a gaussian approximation. Now, there are two alternate methods - density ratio approximation (usually parametric, but recently non-parametric methods have been developed), and contrastive/discriminative classifier-based solution (parametric).

## Introduction
*Part Mutual Information* (PMI) [not to be confused with partial mutual information] has become popular owing to its usefulness in detecting and quantifying direct associations between random variables and infer the underlying network topology. PMI differs from CMI in subtle but crucial ways. CMI is defined as follows
$$
\text{CMI}(X; Y|Z) := \int_{x,y,z} p(x,y,z) \log \frac{p(x,y|z)}{p(x|z) p(y|z)} 

$$
While PMI is defined as 
$$
\text{PMI}(X; Y|Z) := \int_{x,y,z} p(x,y,z) \log \frac{p(x,y|z)}{p^*(x|z) p^*(y|z)} 
$$
Where $p^*(x|z) := \int_y p(x|y,z)p(y)$ and $p^*(y|z) := \int_x p(y|x,z)p(x)$. This partial marginalization is what makesthe computation particularly difficult as it does not allow for usual bayesian computation. This partial marginalization can be understood as similar to the notion of *intervention* in causal inference but with important differences.

So, the task is now to use some parametric methods to compute the above PMI given three random variables $X, Y, Z$.

## Method
Like other approaches, we shall focus on approximating the ratio of the densities in the log term in the PMI equation. But, we cannot do it in a strightforward manner like CMI, owing to the fact that $p^*(x|z)$ and $p^*(y|z)$ cannot be easily sampled. Therefore, we require a generative model to learn the partial marginals and to be able to sample from them. Then, we can use a classifier-based model to learn the log density ratio to finally estimate the PMI.

Now, note that $\log p^*(x|z)$ can be written as

$$
\begin{aligned}
\log p^*(x|z) &= \log \int_y p(x|z, y)p(y) \\
         &= \log \int_y p(x|z, y) \frac{p(z|y)}{p(z|y)}p(y)\\
         &= \log \int_y p(x|z, y) \frac{p(z,y)}{p(z|y)}\\
         &= \log \int_y \frac{p(x, z,y)}{p(z|y)} \frac{p(y)}{p(y)}\\
         &= \log \int_y \frac{p(x, z,y)}{p(z, y)} p(y)\\
         &= \log \mathbb{E}_{p(y)} \frac{p(x, z,y)}{p(z, y)} \\
         &\geq \mathbb{E}_{p(y)} \big [\log p(x, z,y) \big ] - \mathbb{E}_{p(y)} \big [p(z, y) \big ] \\
\end{aligned}
$$
Similarly,
$$
\log p^*(y|z) \geq \mathbb{E}_{p(x)} \big [\log p(x, z,y) \big ] - \mathbb{E}_{p(x)} \big [p(z, x) \big ]
$$

Therefore, learning the densities can be done by simply maximizing the above lower bounds. Now, we can write the log density ratio as

$$
\log \frac{p(x,y|z)}{p^*(x|z) p^*(y|z)}  = \log \big [p(x,y|z) \big ] - \log \big [p^*(x|z) \big ] - \log \big [p^*(y|z) \big ]
$$

Now, all we need is to find an appropriate generative model (do we need a gradient estimator?) to learn the above partial marginals and a discriminative network to learn the density ratio and we are done!

### References
#### Neural CMI estimation
- https://arxiv.org/pdf/2005.08226v2.pdf
- https://arxiv.org/pdf/1911.02277v2.pdf
- http://proceedings.mlr.press/v115/mukherjee20a/mukherjee20a.pdf
- https://arxiv.org/pdf/2010.01766v1.pdf


