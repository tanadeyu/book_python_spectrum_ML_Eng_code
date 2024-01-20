import numpy as np
import matplotlib.pyplot as plt

# Generation of Gaussian signal
np.random.seed(0)
x = np.linspace(0, 10, 101)
gaussian_signal = np.exp(-(x - 5) ** 2 / (2 * 1.5 ** 2)) + \
                  np.random.normal(0, 0.01, len(x))

# Differentiation of Gaussian signal (using central difference)
dx = x[1] - x[0]
derivative = np.gradient(gaussian_signal, dx)

# Specifying graph size
plt.figure(figsize=(9, 5))

# Plotting the original data
plt.subplot(2, 1, 1)
plt.plot(x, gaussian_signal, color='b', label='Original Signal')
plt.title('Original Gaussian Signal')
plt.xlabel('X')
plt.ylabel('Amplitude')
plt.legend()

# Plot the data after differentiation
plt.subplot(2, 1, 2)
plt.plot(x, derivative, color='r', label='Derivative (Numerical)')
plt.title('Numerical Derivative')
plt.xlabel('X')
plt.ylabel('Amplitude')
plt.legend()
plt.tight_layout()
plt.show()
