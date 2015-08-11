import numpy as np
import matplotlib.pyplot as plt

n = np.arange(20).reshape((1,20))
seq = np.zeros((1,20))
seq[0, 0:5] = 1.0

N = 501
fft = np.fft.fftshift(np.fft.fft(seq, n=N))
freq = np.fft.fftshift(np.fft.fftfreq(N)).reshape(1,-1)


fig = plt.figure(1)
ax = fig.add_subplot(121)

ax.plot(n, seq, ls='--', marker='o', c='b')
ax.set_xlabel('n')
ax.set_ylabel('a(n)')
ax.set_xlim([0, 10])
ax.set_ylim([-0.5, 1.5])

ax = fig.add_subplot(122)
ax.plot(np.pi*freq.transpose(), np.abs(fft).transpose())
ax.set_xlabel('freq(rad/sample)')
ax.set_ylabel('DFT')

plt.show()
