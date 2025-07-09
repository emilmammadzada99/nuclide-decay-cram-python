"""
CRAM-16 Decay Solver
---------------------
Uses the 16th-order Chebyshev Rational Approximation Method (CRAM-16)
to compute the radioactive decay of a single nuclide and compares it
to the analytical solution.

"""

import numpy as np
from scipy.sparse import csc_matrix, eye
from scipy.sparse.linalg import spsolve

# CRAM-16 coefficients (from OpenMC)
alpha = np.array([
    5.464930576870210e+3 - 3.797983575308356e+4j,
    9.045112476907548e+1 - 1.115537522430261e+3j,
    2.344818070467641e+2 - 4.228020157070496e+2j,
    9.453304067358312e+1 - 2.951294291446048e+2j,
    7.283792954673409e+2 - 1.205646080220011e+5j,
    3.648229059594851e+1 - 1.155509621409682e+2j,
    2.547321630156819e+1 - 2.639500283021502e+1j,
    2.394538338734709e+1 - 5.650522971778156e+0j
], dtype=np.complex128)

theta = np.array([
    3.509103608414918 + 8.436198985884374j,
    5.948152268951177 + 3.587457362018322j,
    -5.264971343442647 + 16.22022147316793j,
    1.419375897185666 + 10.92536348449672j,
    6.416177699099435 + 1.194122393370139j,
    4.993174737717997 + 5.996881713603942j,
    -1.413928462488886 + 13.49772569889275j,
    -10.84391707869699 + 19.27744616718165j
], dtype=np.complex128)

alpha0 = 2.124853710495224e-16

# Decay parameters
decay_const = 3.124e-17   # U-235 decay constant (1/s)
t = 100.0                 # Time in seconds
N0 = 1.0                  # Initial number density

# Matrix setup
A = csc_matrix([[-decay_const]])
I = eye(1, format='csc')
y = np.array([N0], dtype=np.complex128)

# CRAM-16 evaluation
for a, th in zip(alpha, theta):
    y += 2.0 * np.real(a * spsolve(t * A - th * I, y))

y *= alpha0
cram_result = np.real(y[0])
analytic_result = N0 * np.exp(-decay_const * t)

# Output
print(f"[CRAM-16] At t = {t:.1f} s -> Result: {cram_result:.5e}, Analytic: {analytic_result:.5e}, Difference: {abs(cram_result - analytic_result):.2e}")
