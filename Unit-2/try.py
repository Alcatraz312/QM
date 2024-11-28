import numpy as np 
import matplotlib.pyplot as plt 

N = int(input())
a = -6
b = 6

x = np.linspace(a,b,num = N)
h = x[1] - x[0]

def vpot(x):
    return x
    

def kinetic_energy(n):
    K = np.zeros((n-2,n-2))

    for i in range(n-2):
        for j in range(n-2): 
            if i == j:
                K[i][j] = -2
            elif abs(i - j) == 1:
                K[i][j] = 1
    
    return K

def potential_energy(n):
    V = np.zeros((n-2, n-2))

    for i in range(n-2):
        for j in range(n-2): 
            if i == j:
                V[i][j] == vpot(x[i + 1])
            else:
                V[i][j] == 0

    return V

hamiltonian = -kinetic_energy(N)/2*h**2 + potential_energy(N)

val, vec = np.linalg.eig(hamiltonian)

z = np.argsort(val)
z = z[0:3]

energies = val[z]/val[z][0]

for i in range(len(z)):
    y = []
    y = np.append(y, vec[:,z[i]])
    y = np.append(y,0)
    y = np.insert(y,0,0)
    plt.plot(x,y)

plt.xlabel("X")
plt.ylabel("$\psi$(x)")
plt.show()



