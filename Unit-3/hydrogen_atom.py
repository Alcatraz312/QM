import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp

# Constants
m = 0.511e6  # Reduced mass of the electron (eV/c^2)
hbar = 1973 / (2 * np.pi)  # hbar in (eV*Å)
e2 = 3.795 

# Finite Difference Method Implementation

# Parameters
N = 1000  # Number of grid points
r_min, r_max = 1e-5, 20  # Range for r (Å)
r = np.linspace(r_min, r_max, N)  # Radial grid
dr = r[1] - r[0]  # Step size

# Potential energy function
V = -e2 / r  # Coulomb potential

# Constants for the Hamiltonian matrix
const = -hbar**2 / (2 * m * dr**2)  # Discretization constant

# Hamiltonian matrix construction (tridiagonal)
H = np.zeros((N, N))
for i in range(1, N - 1):
    H[i, i - 1] = const
    H[i, i] = -2 * const + V[i]
    H[i, i + 1] = const

# Boundary conditions
H[0, 0] = H[-1, -1] = 1e10  # Large values to enforce u(0) = u(r_max) = 0

# Solve eigenvalue problem
eigenvalues, eigenvectors = np.linalg.eigh(H)

# Select the ground state (n=1) and first excited state (n=2)
ground_state_index = 0
first_excited_state_index = 1

# Normalize wavefunctions
u_ground = eigenvectors[:, ground_state_index]
u_ground /= np.sqrt(np.trapz(u_ground**2, x=r))

u_excited = eigenvectors[:, first_excited_state_index]
u_excited /= np.sqrt(np.trapz(u_excited**2, x=r))

# Plotting the results
plt.figure(figsize=(10, 6))
plt.plot(r, u_ground, label="Ground State (n=1)")
plt.plot(r, u_excited, label="First Excited State (n=2)")
plt.title("Radial Wavefunctions of Hydrogen Atom (FDM)")
plt.xlabel("r (Å)")
plt.ylabel("u(r) (normalized)")
plt.legend()
plt.grid()
plt.show()

# Displaying energy levels
ground_energy = eigenvalues[ground_state_index]
excited_energy = eigenvalues[first_excited_state_index]

ground_energy, excited_energy
