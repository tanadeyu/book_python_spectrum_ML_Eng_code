import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import savgol_filter

# Generation of Gaussian signal
np.random.seed(0)
x = np.linspace(0, 10, 100)
gaussian_signal = np.exp(-(x - 5)**2 / (2 * 1.5**2)) + \
                  np.random.normal(0, 0.01, len(x))

# Differentiation using Savitzky-Golay filter
window_size = 9  # Window size (specify an odd number)
order = 1  # Degree of polynomial
derivative = \
savgol_filter(gaussian_signal, window_size, polyorder=order, deriv=1)

# Drawing a graph
plt.figure(figsize=(9, 5))

# Plotting the original data
plt.subplot(2, 1, 1)
plt.plot(x, gaussian_signal, color='b', label='Original Signal')
plt.title('Original Gaussian Signal')
plt.xlabel('X')
plt.ylabel('Amplitude')
plt.legend()

# Plotting data after Savitzky-Golay differentiation
plt.subplot(2, 1, 2)
plt.plot(x, derivative, color='r', label='Derivative (Savitzky-Golay)')
plt.title('Differentiation using Savitzky-Golay Filter')
plt.xlabel('X')
plt.ylabel('Amplitude')
plt.legend()
plt.tight_layout()
plt.show()
