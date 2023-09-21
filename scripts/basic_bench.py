import timeit

import _xts_linalg as xtsl
import numpy as np
from scipy.sparse.linalg import cg

# Setting up your variables
A = np.array([[4, 1], [1, 3]])
b = np.array([1, 2])
x0 = np.array([2, 1])  # Initial guess
maxIters = 1000
tol = 1e-5
cginp = xtsl.linalg.iterative.ConjugateGradientParams()
cginp.max_iter = maxIters


# Function for xtsl's Conjugate Gradient
def xtsl_cg():
    xtsl.iterative.conjugate_gradient(A, b, x0, cginp)


# Function for scipy's Conjugate Gradient
def scipy_cg():
    cg(A, b, x0=x0, maxiter=maxIters, tol=tol)


# Benchmarking
print("Benchmarking xtsl's Conjugate Gradient:")
xtsl_time = timeit.timeit(xtsl_cg, number=1000)
print(f"Average time per iteration: {xtsl_time / 1000:.6f} seconds")

print("\nBenchmarking scipy's Conjugate Gradient:")
scipy_time = timeit.timeit(scipy_cg, number=1000)
print(f"Average time per iteration: {scipy_time / 1000:.6f} seconds")
