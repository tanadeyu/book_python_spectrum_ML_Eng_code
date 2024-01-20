import numpy as np
import matplotlib.pyplot as plt

# Sampling frequency [Hz]
fs = 1000

# Creating a timeline
t = np.arange(0, 1, 1/fs)

# Setting frequency and amplitude
f1 = 10
f2 = 20
A1 = 1
A2 = 0.5

# Superposition wave from two sine waves
x = A1 * np.sin(2 * np.pi * f1 * t) + A2 * np.sin(2 * np.pi * f2 * t)

# Fourier transform calculation
X = np.fft.fft(x)

# Calculating frequency components
freq = np.fft.fftfreq(len(x), d=1/fs)
idx = np.argsort(freq)

# Plot placement and size settings
fig, axs = plt.subplots(4, 1, figsize=(8, 8))

# Original sine wave
axs[0].set_title('Original')
axs[0].plot(t, x)
axs[0].set_xlabel('Time [s]')
axs[0].set_ylabel('Amplitude')

# Displayed on frequency axis
axs[1].set_title('FFT')
axs[1].plot(freq[idx], np.abs(X[idx]) * 2 / len(x))
axs[1].set_xlabel('Frequency [Hz]')
axs[1].set_ylabel('Magnitude')

# Inverse Fourier transform
y = np.fft.ifft(X)
axs[2].set_title('iFFT')
axs[2].plot(t, y.real)
axs[2].set_xlabel('Time [s]')
axs[2].set_ylabel('Amplitude')

# Displaying phase information
phase = np.angle(X)
axs[3].set_title('Phase')
axs[3].plot(freq[idx], phase[idx]/np.pi)
axs[3].set_xlabel('Frequency [Hz]')
axs[3].set_ylabel('Phase [rad]')

# Display graph
plt.tight_layout()
plt.show()
