import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import savgol_filter

# Generate a pseudo spectral signal based on Gaussian
x = np.linspace(400, 800, 1000)  # wavelength range
gaussian_signal \
= 32000 * np.exp(-(x - 520)**2 / (2 * 3.0**2))  # Gaussian spectrum signal
noise = np.random.normal(0, 1000, gaussian_signal.shape)
gaussian_signal= gaussian_signal+noise

# Moving average window size (odd number)
window_size = 21

# Smoothing using Savitzky-Golay filter
smoothed_signal = savgol_filter(gaussian_signal, window_size, 2)  

# Create a graph
plt.figure(figsize=(9, 6))
plt.plot(x, gaussian_signal, label='Original Signal', \
         color='blue', alpha = 0.3)
plt.plot(x, smoothed_signal, label='Smoothed Signal', \
         color='red',linewidth=3,linestyle='--')
plt.xlabel('Wavelength')
plt.ylabel('Intensity')
plt.title('Spectrum Signal and Moving Average')
plt.legend()
plt.grid(True)
plt.show()
