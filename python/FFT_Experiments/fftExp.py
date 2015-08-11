import numpy as np
import matplotlib.pyplot as plt

a = np.linspace(-1.0, 1.0, 100)*0.5
X,Y = np.meshgrid(a,a)

ZZ = X + 1j*Y

ZZ = ZZ.flatten()

real = ZZ.real
imag = ZZ.imag

def evaluateZtransform(z, seq):
    seq = seq.reshape(-1,1)
    zz = np.power(z, np.arange(seq.shape[0])).reshape(-1,1)
    return np.sum(zz*seq)

filter = np.hstack((np.ones((1,10)), np.zeros((1,40))))
seq = np.fft.ifft(filter).reshape(-1,1)
Y = np.array([evaluateZtransform(z, seq) for z in ZZ]).reshape(-1,1)
mag = np.abs(Y)

n = np.sqrt(mag.shape[0])

mag = mag.reshape((n,n))
real = real.reshape((n,n))
imag = imag.reshape((n,n))


from mpl_toolkits.mplot3d import Axes3D
from matplotlib import cm
from matplotlib.ticker import LinearLocator, FormatStrFormatter
import matplotlib.pyplot as plt
import numpy as np

fig = plt.figure()
ax = fig.gca(projection='3d')
surf = ax.plot_surface(real, imag, mag)
ax.set_zlim(0, 10)


plt.show()


