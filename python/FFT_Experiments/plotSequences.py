import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import numpy as np

plt.ion()
M = 12
n = np.arange(M)
fig = plt.figure()
for k in range(9):
    seq = np.exp(1j*2.0*np.pi/M*k*n)

    x = n
    y = seq.imag
    z = seq.real

    ax = fig.add_subplot(3, 3, k+1, projection='3d')

    ax.set_xlabel('n')
    ax.set_ylabel('imag')
    ax.set_zlabel('real')
    ax.set_axis_off()

    #plot sequence points
    ax.scatter(x,y,z, s=40)
    ax.legend(["k=%d"%k])

    #plot n axis
    ax.plot(xs=[0.0, M], ys=[0.0, 0.0], zs=[0.0,0.0])

    for i,p in enumerate(seq):
        ax.plot([i, i], [0, p.imag], [0, p.real], ls='--', c='y')


plt.draw()
