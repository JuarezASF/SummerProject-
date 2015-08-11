import numpy as np
from numpy.fft import *
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

n = np.arange(0,10)
seq = 1./(n + 1)

plt.figure()
ax = plt.subplot()
plt.title('Length 10 sequence')
plt.xlabel('n')
ax.set_ylim([0.0, 1.2])
ax.plot(n, seq, marker = 'o', ls = '--')


N = n.shape[0]
freqResponse = fft(seq, n = N)
freq = fftfreq(n = N)

freqResponse = fftshift(freqResponse)
freq = fftshift(freq)

plot_FFT = plt.figure()
ax = plt.subplot()
plt.title('Length 10 Discrete Fourrier Transform')
ax.plot(freq, np.abs(freqResponse), marker = 'o', ls = '--')
plt.xlabel('freq(2pi rad/sample)')
plt.ylabel('Magnitude of DFT')


n = np.arange(0, 20)
rec = np.zeros_like(n)
for c,w in zip(freqResponse, freq):
    component = c*np.exp(1j*2*np.pi*w*n)/N
    rec = rec + component

plt.figure()
plt.title('Reconstructed signal')
plt.xlabel('n')
plt.plot(n, rec, marker = 'o', ls = '--')

plt.show()
