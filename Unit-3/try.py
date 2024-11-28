import numpy as np
import matplotlib.pyplot as plt 

n = 1000
r_min, r_max = 1e-5, 20
r = np.linspace(r_min, r_max, n)

dr = r[1] - r[0]
e2 = 3.795
m = 0.511e6

vpot = e2/r
h_bar = 1973/(2*np.pi)

const = -h_bar**2/(2*m*(dr**2))


T = np.zeros((n,n))
for i in range(n):
    for j in range(n):
        if i == j:
            T[i][j] = -2 * const
        elif abs(i-j) == 1:
            T[i][j] = 1 * const



V = np.zeros((n,n))
for i in range(n):
    for j in range(n):
        if i == j:
            V[i][j] = vpot[i]



H = V + T

H[0,0] = H[-1,-1] = 1e10

eigen_values, eigen_vectors = np.linalg.eigh(H)

ground_energy = eigen_values[0]
first_excited_state = eigen_values[1]

def normalized(u):
    return np.sqrt(np.trapz(u**2, x = r))

ground_state_wavefunction = eigen_vectors[:, 0]
first_excited_state_wavefucntion = eigen_vectors[:, 1]

plt.figure(figsize=(10,6))
plt.plot(r, ground_state_wavefunction)
plt.plot(r, first_excited_state_wavefucntion)

plt.show()

