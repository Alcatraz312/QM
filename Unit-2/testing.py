import numpy as np
import matplotlib.pyplot as plt
from scipy.linalg import eigh  # For solving eigenvalue problems

# Constants
hbar = 1.0  # Planck's constant (normalized for simplicity)
m = 1.0     # Mass of particle (normalized)

# Define the grid
L = 1.0  # Length of the domain (for particle in a box)
N = 1000  # Number of grid points
x = np.linspace(0, L, N)  # Grid points

dx = x[1] - x[0]  # Spatial step size

# Potential definitions
def potential_infinite_well(x):
    """ Infinite potential well, 0 inside and very large outside."""
    V = np.zeros_like(x)
    V[0] = V[-1] = 1e10  # Large value at boundaries
    return V

def potential_finite_well(x, V0=50):
    """ Finite potential well with depth V0."""
    V = np.zeros_like(x)
    V[0] = V[-1] = V0  # Finite potential well walls
    return V

def harmonic_potential(x, k=1):
    """ Harmonic potential: V(x) = (1/2) * k * x^2 """
    return 0.5 * k * (x - L/2) ** 2

# Choose a potential
V = potential_infinite_well(x)  # Choose a potential function here

# Setup Hamiltonian matrix (H = T + V)
H = np.zeros((N, N))

# Second derivative using finite difference approximation
for i in range(1, N-1):
    H[i, i] = -2.0
    H[i, i-1] = H[i, i+1] = 1.0

# Scale by constants
H = - (hbar*2 / (2 * m * dx*2)) * H

# Add potential to Hamiltonian
for i in range(N):
    H[i, i] += V[i]

# Solve eigenvalue problem for H
energies, wavefunctions = eigh(H)

# Plot results
def plot_wavefunction(n):
    plt.plot(x, wavefunctions[:, n], label=f"n={n+1}, E={energies[n]:.2f}")
    plt.xlabel("Position x")
    plt.ylabel(r"$|\psi(x)|^2$")
    plt.legend()

plt.figure(figsize=(8,6))
plot_wavefunction(0)  # Ground state
plot_wavefunction(1)  # First excited state
plot_wavefunction(2)  # Second excited state
plt.title("Probability Densities for a 1D Potential")
plt.show()

