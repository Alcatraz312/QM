# import numpy as np
# import matplotlib.pyplot as plt
# x = np.linspace(-1,1,num = 100)

# y = 2 * x 
# z = x ** 2


# psi = np.piecewise(x, [x < 0, x >= 0], funclist = [lambda x : y, lambda x : z])

# print(psi)
# plt.plot(x,psi)
# plt.show()

import numpy as np
import scipy.optimize as opt
import matplotlib.pyplot as plt 

def potential_well(a, U):

    def transcendental_equation(k):
        return np.tan(k*a) + (np.sqrt(k**2 - U))/k
    
    roots = opt.fsolve(transcendental_equation, x0= np.linspace(0,np.sqrt(U),100))

    eigen_values = roots ** 2

    return eigen_values

def normalize_wavefunction(psi, x):

    n_factor = np.sqrt(np.trapz(np.abs(psi**2), x))

    return psi/n_factor

def plot_wavefunction(U, a, energies):

    x = np.linspace(-8*a , 8*a , 1000)
    alpha_arr = []
    eigen_arr = []
    k_arr = []

    for eigen_value in energies:
        # print(f'Eigen Value: {eigen_value}')
        eigen_arr.append(eigen_value)
        k = np.sqrt(eigen_value)
        k_arr.append(k)
        # print(f'k:{k}')
        alpha = np.sqrt(U - eigen_value)
        # print(f'Alpha:{alpha}')
        alpha_arr.append(alpha)

        psi_inside = np.cos(k * x)
        psi_outside_left = np.exp(- alpha * (x + a))
        psi_outside_right = np.exp(- alpha * (x - a))
        const_func = 1

        psi = np.piecewise(x, [x < -a, -a <= x,  x <= a, x > a], funclist = [lambda x: np.exp( - alpha * (x + a)) ,lambda x: np.cos(k * x) ,lambda x: np.cos(k * x), lambda x: np.exp( - alpha * (x - a))])

        normalize_psi = normalize_wavefunction(psi, x)

        plt.plot(x, normalize_psi)


    plt.xlabel("x")
    plt.ylabel("ψ(x)")
    plt.title(f"Finite potential well U = {U}, a = {a}")
    plt.legend()
    plt.show()

    print(alpha_arr)
    print(eigen_arr)
    print(k_arr)

U_values = [5]
a_values = [3]

for U in U_values:
    for a in a_values:
        eigen_values = potential_well(U,a)
        plot_wavefunction(U, a, energies = eigen_values)

