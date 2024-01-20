import numpy as np
import matplotlib.pyplot as plt

# Sampling frequency [Hz]
fs = 1000

# Creating a timeline
t = np.arange(0, 1, 1/fs)

# Setting the frequency and amplitude of two signals
f1 = 10
f2 = 100
A1 = 1
A2 = 0.5

# Generate a wave by overlapping two sine waves
x = A1 * np.sin(2 * np.pi * f1 * t) + A2 * np.sin(2 * np.pi * f2 * t)

# FFT (Fourier transform computation)
X = np.fft.fft(x)

# Calculating frequency components
freq = np.fft.fftfreq(len(x), d=1/fs)
idx = np.argsort(freq)

# Setting the plot placement and size
fig, axs = plt.subplots(2, 1, figsize=(8, 6))

# Original sine wave
axs[0].plot(t, x)
axs[0].set_xlabel('Time [s]')
axs[0].set_ylabel('Amplitude')

# Display the signal after converting to frequency axis
#axs[1].stem(freq[idx], np.abs(X[idx]) * 2 / len(x))
axs[1].plot(freq[idx], np.abs(X[idx]) * 2 / len(x))
axs[1].set_xlabel('Frequency [Hz]')
axs[1].set_ylabel('Magnitude')
plt.show()
