import numpy as np
import scipy.optimize as opt
import matplotlib.pyplot as plt 

def potential_well(a, U):      #parameters: well width --> -a to a, and depth of the potential well

    def transcendental_equation(k):
        return np.tan(k*2*a) + (np.sqrt(k**2 - U))/k    # derived the transcendental equation
    
    roots = opt.fsolve(transcendental_equation, x0= np.linspace(0,np.sqrt(U),100))   # solving the transcendental equation for k

    eigen_values = roots ** 2    # because k = root(Energy)

    return eigen_values   # this function returns different eigen values


def normalize_wavefunction(psi, x):

    n_factor = np.sqrt(np.trapz(np.abs(psi**2), x))    # using trapz for all space integral 

    return psi/n_factor    # n-factor - normalization factor

def plot_wavefunction(U, a, energies):

    x = np.linspace(-2*a , 2*a , 1000)   # defining the x - axis
    alpha_arr = []    # array of all the alpha values
    eigen_arr = []     # array of all the eigen values
    k_arr = []          # array of all the k values

    for eigen_value in energies:     # for a energy value in all the energy values 
        # print(f'Eigen Value: {eigen_value}')
        eigen_arr.append(eigen_value)
        k = np.sqrt(eigen_value)      # k = root(energy) inside the well
        k_arr.append(k)
        # print(f'k:{k}')
        alpha = np.sqrt(U - eigen_value)  # alpha = root(potential energy - energy) outside of the well
        # print(f'Alpha:{alpha}')
        alpha_arr.append(alpha) 

        # psi_inside = np.cos(k * x)
        # psi_outside_left = np.exp(- alpha * (x + a))
        # psi_outside_right = np.exp(- alpha * (x - a))
        # const_func = 1

        # breaking the function into 3 parts: outside left of well, inside the well, outside right of well, and putting their corresponding functions
        psi = np.piecewise(x, [x < -a, -a <= x, x > a], funclist = [lambda x: np.exp(alpha * (x + a)) \
                                                                        ,lambda x: np.cos(k * x) \
                                                                        ,lambda x: np.exp(- alpha * (x - a))])
        

        normalize_psi = normalize_wavefunction(psi, x)    # normalizing the psi function

        plt.plot(x, normalize_psi)  # plotting 


    plt.xlabel("x")
    plt.ylabel("ψ(x)")
    plt.title(f"Finite potential well U = {U}, a = {a}")
    plt.legend()
    plt.show()

    print(alpha_arr)
    print(eigen_arr)
    print(k_arr)

U_values = [5, 10, 15]     # creating a list of values for the potential well depth
a_values = [1, 2, 3]        # creating a list of values for the width of the potential well

# U_values = [5]
# a_values = [3]

for U in U_values:
    for a in a_values:
        eigen_values = potential_well(U,a)
        plot_wavefunction(U, a, energies = eigen_values)

