# PDDA: Position-Dependent Diffusivity Analysis

[![made-with-python](https://img.shields.io/badge/Made%20with-Python-green.svg)](https://www.python.org/)
[![MIT license](https://img.shields.io/badge/License-MIT-green.svg)](https://lbesson.mit-license.org/)

**PDDA** is a Python library for estimating diffusivities from one-dimensional molecular dynamics (MD) trajectories.

The current implementation focuses on the **residence-time approach (RTA)** for estimating the **diffusivity** in a spatial interval under the assumptions of **constant PMF** and **constant diffusivity** within that interval.

## What PDDA provides

### Residence-time approach (RTA) for bulk diffusion in an interval

For a spatial interval $\Omega = [x_\mathsf{min}, x_\mathsf{max})$, PDDA implements the bulk residence-time identity

$$
D = \frac{(x_\mathsf{max} - x_\mathsf{min})^2}{12\,\tau_\Omega},
$$

where $\tau_\Omega$ is the mean first-exit time, averaged over all trajectory frames with $x[k]\in\Omega$.

### Uncertainty estimates for correlated data

PDDA includes an implementation of Jonsson’s automated blocking method to estimate the standard error of the mean (SEM) from correlated time series.

- Jonsson, *Phys. Rev. E* **98**, 043304 (2018). https://doi.org/10.1103/PhysRevE.98.043304

In code, this is exposed via `helpers.sem(x, corr=True)`.

## Quickstart

```python
import numpy as np
from pdda.rta import diffusivity

# x: 1D trajectory sampled uniformly every dt
# x = np.loadtxt("trajectory_x.dat")

dt = 0.01
xmin, xmax = 4.0, 6.0

D_est, D_sem = diffusivity(x, xmin, xmax, dt)

print("D =", D_est, "+/-", D_sem)
```

## Citation

If you use this code, please cite:

Thomas, R.; Prabhakar, P. R.; von Domaros, M.
A Residence-Time Approach for Determining Position-Dependent Diffusivities from Biased Molecular Simulations.
[arXiv:2604.01940](https://arxiv.org/abs/2604.01940) (2026).
