import numpy as np
from scipy.special import sph_harm
import matplotlib.pyplot as plt


def plot_spharm(l : int,m : int):

    theta, phi = np.mgrid[0:np.pi:50j, 0:2 * np.pi:50j]
    psi = sph_harm(l,m,theta,phi)

    p_den = np.abs(psi) ** 2

    x = p_den * np.sin(theta) * np.cos(phi)
    y = p_den * np.sin(theta) * np.sin(phi)
    z = p_den * np.cos(theta)

    fig, ax = plt.subplots(subplot_kw={"projection" : "3d"})   

    ax.plot_surface(x, y, z, rstride=3, cstride=3, facecolors=plt.cm.viridis(p_den), linewidth = 0, antialiased = False)
    ax.set_title(f"Spherical Harmonics l = {l}, m = {m}")
    plt.show()

plot_spharm(0,0)



