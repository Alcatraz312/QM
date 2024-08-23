import numpy as np
import matplotlib.pyplot as plt
import scipy.optimize as opt

def finite_potential_well(V0, a):
    """
    Calculates the energy eigenvalues for a particle in a finite potential well.

    Args:
        V0: Depth of the potential well.
        a: Width of the potential well.

    Returns:
        A list of energy eigenvalues.
    """

    def transcendental_equation(k):
        return np.tan(k * a) + np.sqrt((V0 - k**2) / k**2)

    # Find the roots of the transcendental equation
    roots = opt.fsolve(transcendental_equation, np.linspace(0, np.sqrt(V0), 100))

    # Calculate the energy eigenvalues
    energies = roots**2 / 2

    return energies

def normalize_wavefunction(psi, x):
    """
    Normalizes a wavefunction.

    Args:
        psi: The wavefunction.
        x: The spatial coordinate.

    Returns:
        The normalized wavefunction.
    """

    normalization_factor = np.sqrt(np.trapz(np.abs(psi)**2, x))
    return psi / normalization_factor

def plot_wavefunctions(V0, a, energies):
    """
    Plots the normalized eigenfunctions for a particle in a finite potential well.

    Args:
        V0: Depth of the potential well.
        a: Width of the potential well.
        energies: A list of energy eigenvalues.
    """

    x = np.linspace(-a, a, 1000)

    for energy in energies:
        # Calculate the wavefunction
        k = np.sqrt(2 * energy)
        alpha = np.sqrt(2 * (V0 - energy))

        psi_inside = np.sin(k * x)
        psi_outside_left = np.exp(-alpha * (x + a))
        psi_outside_right = np.exp(-alpha * (x - a))

        psi = np.zeros_like(x)  # Create an array of zeros with the same shape as x
        psi[x < -a] = psi_outside_left
        psi[np.logical_and(-a <= x, x <= a)] = psi_inside
        psi[x > a] = psi_outside_right

        # Normalize the wavefunction
        psi_normalized = normalize_wavefunction(psi, x)

        # Plot the wavefunction
        plt.plot(x, psi_normalized, label=f"E = {energy:.2f}")

    plt.xlabel("x")
    plt.ylabel("ψ(x)")
    plt.title(f"Finite Potential Well (V0 = {V0}, a = {a})")
    plt.legend()
    plt.show()

# Example usage
V0_values = [5, 10, 15]
a_values = [1, 2, 3]

for V0 in V0_values:
    for a in a_values:
        energies = finite_potential_well(V0, a)
        plot_wavefunctions(V0, a, energies)