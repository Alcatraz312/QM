import numpy as np 
import matplotlib.pyplot as plt
from scipy.special import sph_harm

def spherical_harm(l, m):

    theta, phi = np.mgrid[0:np.pi:50j, 0:2 * np.pi:50j]
    psi = sph_harm(l,m, theta, phi)

    p_den = np.abs(psi)**2

    x = p_den * np.sin(theta) * np.cos(phi)
    y = p_den * np.sin(theta) * np.sin(phi)
    z = p_den * np.cos(theta)

    fig, ax = plt.subplots(subplot_kw={"projection" : "3d"})
    ax.plot_surface(x,y,z, facecolors = plt.cm.viridis(p_den))
    ax.set_title("Spherical harmonics")
    plt.show()

spherical_harm(1,1)