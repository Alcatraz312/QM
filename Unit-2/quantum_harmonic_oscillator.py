import numpy as np
from scipy import integrate
import matplotlib.pyplot as plt

def U(x):
    return (x**2)/2

a = -6
b = 6
n = 100

x = np.linspace(a,b, num = n)
step = x[1] - x[0]
x

def kinetic_energy(n):
    kinetic_energy_matrix = np.zeros((n-2)**2).reshape(n-2,n-2)

    for i in range(n-2):
        for j in range(n-2):
            if i == j:
                kinetic_energy_matrix[i][j] = -2
            elif np.abs(i-j) == 1:
                kinetic_energy_matrix[i][j] = 1
            else:
                kinetic_energy_matrix[i][j] = 0

    return kinetic_energy_matrix

kinetic_energy(n)

def potential_energy(n):
    potential_energy_matrix = np.zeros((n-2)**2).reshape(n-2,n-2)

    for i in range(n-2):
        for j in range(n-2):
            if i == j:
                potential_energy_matrix[i][j] = U(x[i + 1])

            else:
                potential_energy_matrix[i][j] = 0

    return potential_energy_matrix

potential_energy(n)

hamiltonian = kinetic_energy(n)/(2*step**2) + potential_energy(n)

E_value, E_func = np.linalg.eig(hamiltonian)

Eigen_value = np.argsort(E_value, axis = 0)
Eigen_value = Eigen_value[0:2]
E_func = E_func.T

energies = (E_value[Eigen_value]/E_value[Eigen_value][0])
print(energies)

for i in range(len(Eigen_value)):
    y = np.concatenate(([0], E_func[i], [0]))
    plt.plot(x,y)

plt.show()

