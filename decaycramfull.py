import numpy as np
import matplotlib.pyplot as plt
from scipy.sparse import csc_matrix, eye
from scipy.sparse.linalg import spsolve

# -------------------------------
# CRAM-48 coefficients
# -------------------------------
c48_theta = np.array([
    -4.465731934165702e+1 + 6.233225190695437e+1j,
    -5.284616241568964e+0 + 4.057499381311059e+1j,
    -8.867715667624458e+0 + 4.325515754166724e+1j,
    +3.493013124279215e+0 + 3.281615453173585e+1j,
    +1.564102508858634e+1 + 1.558061616372237e+1j,
    +1.742097597385893e+1 + 1.076629305714420e+1j,
    -2.834466755180654e+1 + 5.492841024648724e+1j,
    +1.661569367939544e+1 + 1.316994930024688e+1j,
    +8.011836167974721e+0 + 2.780232111309410e+1j,
    -2.056267541998229e+0 + 3.794824788914354e+1j,
    +1.449208170441839e+1 + 1.799988210051809e+1j,
    +1.853807176907916e+1 + 5.974332563100539e+0j,
    +9.932562704505182e+0 + 2.532823409972962e+1j,
    -2.244223871767187e+1 + 5.179633600312162e+1j,
    +8.590014121680897e-1 + 3.536456194294350e+1j,
    -1.286192925744479e+1 + 4.600304902833652e+1j,
    +1.164596909542055e+1 + 2.287153304140217e+1j,
    +1.806076684783089e+1 + 8.368200580099821e+00j,
    +5.870672154659249e+00 + 3.029700159040121e+01j,
    -3.542938819659747e+01 + 5.834381701800013e+01j,
    +1.901323489060250e+01 + 1.194282058271408e+00j,
    +1.885508331552577e+01 + 3.583428564427879e+00j,
    -1.734689708174982e+01 + 4.883941101108207e+01j,
    +1.316284237125190e+01 + 2.042951874827759e+01j
], dtype=np.complex128)

c48_alpha = np.array([
    +6.387380733878774e+2 - 6.743912502859256e+2j,
    +1.909896179065730e+2 - 3.973203432721332e+2j,
    +4.236195226571914e+2 - 2.041233768918671e+3j,
    +4.645770595258726e+2 - 1.652917287299683e+3j,
    +7.765163276752433e+2 - 1.783617639907328e+4j,
    +1.907115136768522e+3 - 5.887068595142284e+4j,
    +2.909892685603256e+3 - 9.953255345514560e+3j,
    +1.944772206620450e+2 - 1.427131226068449e+3j,
    +1.382799786972332e+5 - 3.256885197214938e+6j,
    +5.628442079602433e+3 - 2.924284515884309e+4j,
    +2.151681283794220e+2 - 1.121774011188224e+3j,
    +1.324720240514420e+3 - 6.370088443140973e+4j,
    +1.617548476343347e+4 - 1.008798413156542e+6j,
    +1.112729040439685e+2 - 8.837109731680418e+1j,
    +1.074624783191125e+2 - 1.457246116408180e+2j,
    +8.835727765158191e+1 - 6.388286188419360e+1j,
    +9.354078136054179e+1 - 2.195424319460237e+2j,
    +9.418142823531573e+1 - 6.719055740098035e+2j,
    +1.040012390717851e+2 - 1.693747595553868e+2j,
    +6.861882624343235e+1 - 1.177598523430493e+1j,
    +8.766654491283722e+1 - 4.596464999363902e+3j,
    +1.056007619389650e+2 - 1.738294585524067e+3j,
    +7.738987569039419e+1 - 4.311715386228984e+1j,
    +1.041366366475571e+2 - 2.777743732451969e+2j
], dtype=np.complex128)

c48_alpha0 = 2.258038182743983e-47

# -------------------------------
# Pu-241 Decay Chain
# -------------------------------
isotopes = [
    "Pu-241", "Am-241", "Np-237", "Pa-233", "U-233", "Th-229",
    "Ra-225", "Ac-225", "Fr-221", "At-217", "Bi-213", "Po-213",
    "Pb-209", "Bi-209"
]

# Half-lives in years (approx)
half_lives_years = np.array([
    14.3,         # Pu-241
    432.2,        # Am-241
    2.14e6,       # Np-237
    2.73e4,       # Pa-233
    1.592e5,      # U-233
    7340,         # Th-229
    14.9,         # Ra-225
    0.01,         # Ac-225 (10 days)
    0.00029,      # Fr-221 (2.6 hours)
    0.000000014,  # At-217 (~0.03 s)
    0.000000046,  # Bi-213 (~46 s)
    0.000000004,  # Po-213 (~4 µs)
    2.3e7,        # Pb-209
    np.inf        # Bi-209 stable
])

# Convert to decay constants (1/s)
seconds_per_year = 365.25 * 24 * 3600
lambdas = np.log(2) / (half_lives_years * seconds_per_year)
lambdas[-1] = 0.0  # Bi-209 stable

# -------------------------------
# Build decay matrix A
# -------------------------------
n = len(isotopes)
A = np.zeros((n, n))

for i in range(n):
    A[i, i] = -lambdas[i]
    if i < n - 1:
        A[i + 1, i] = lambdas[i]  # parent → daughter

A = csc_matrix(A)

# -------------------------------
# CRAM-48 solver
# -------------------------------
def cram48_solve(A, N0, t):
    I = eye(A.shape[0], format='csc')
    y = np.array(N0, dtype=np.complex128)
    for a, th in zip(c48_alpha, c48_theta):
        y += 2.0 * np.real(a * spsolve(t * A - th * I, y))
    y *= c48_alpha0
    return np.real(y)

# -------------------------------
# Time evolution (in years)
# -------------------------------
times_years = np.logspace(-6, 7, 200)  # from microyear to 10 million years
results = []

N0 = np.zeros(n)
N0[0] = 1.0  # start with Pu-241 = 1.0

for t_years in times_years:
    t_sec = t_years * seconds_per_year
    y = cram48_solve(A, N0, t_sec)
    results.append(y)

results = np.array(results)

# -------------------------------
# Plot results
# -------------------------------
plt.figure(figsize=(10,6))
# Tüm izotopları çizelim
for i, iso in enumerate(isotopes):
    plt.plot(times_years, results[:, i], label=iso)
# for i, iso in enumerate(isotopes):
#     if i in [0, 1, 2, 4, 13]:  # only major isotopes for clarity
#         plt.plot(times_years, results[:, i], label=iso)

plt.xscale("log")
plt.xlim(1e-5, 1e+9)
plt.yscale("log")
plt.ylim(1e-29, 1e+7)  # <-- y-axis starts from 10^-29 up to 1
plt.xlabel("Time (years)")
plt.ylabel("Quantity (N / N₀)")
plt.title("Pu-241 Decay Chain (CRAM-48 Simulation)")
plt.legend()
plt.grid(True, which="both", ls="--", lw=0.5)
plt.tight_layout()
plt.show()
