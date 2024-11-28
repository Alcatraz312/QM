import matplotlib.pyplot as plt 
import numpy as np

N = 1000    #number of grid points
r_min, r_max = 1e-5, 20    # setting up maximum r and minimum r
r = np.linspace(r_min, r_max, num= N)   # radial distance
dr = r[1] - r[0]    # step value
e2 = 3.795   # equivalent to e^2 
m = 0.511e6    # reduced mass of the electron
vpot = -e2/r    # potential energy in the hydrogen atom
h_bar = 1973/(2*np.pi)   # h bar 

constant = -h_bar**2/(2*m*(dr**2))  

T = np.zeros((N,N))
for i in range(N):
    for j in range(N):
        if i == j:
            T[i,j] = -2 * constant
        elif np.abs(i - j) == 1:
            T[i,j] = constant

V = np.zeros((N,N))
for i in range(N):
    for j in range(N):
        if i == j:
            V[i,j] = vpot[i]


H = T + V

H[0,0] = H[-1,-1] = 1e10

def normalized(u):
    return np.sqrt(np.trapz(u**2, x = r))

eigen_values, eigen_vectors = np.linalg.eigh(H)

groundstate_wavefunction = eigen_vectors[:,0]  
normalized_groundstate_wavefunction = normalized(groundstate_wavefunction)

first_excited_wavefunction = eigen_vectors[:,1]
normalized_first_excited_wavefunction = normalized(first_excited_wavefunction)

print(normalized_first_excited_wavefunction)
plt.figure(figsize=(10,6))
plt.plot(r, groundstate_wavefunction, label = "Ground state wavefunction")
plt.plot(r, first_excited_wavefunction, label = "First excited state wavefunction")
plt.legend()
plt.grid()
plt.show()

