import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import windows

# Sine wave and white noise generation
t = np.arange(0, 1, 0.001)
sin_wave = np.sin(2 * np.pi * 5 * t)  # 5Hz sine wave
white_noise = np.random.normal(0, 1, len(t))  #mean 0, standard deviation 1

# Composite wave
composite_wave = sin_wave + white_noise

# Window function
windows_names = ['rectangular', 'hamming', 'hann', 'gaussian']
windows_functions = [windows.boxcar(len(t)), \
                     windows.hamming(len(t)), \
                     windows.hann(len(t)), \
                     windows.gaussian(len(t), std=150)]

# graph display
fig, axs = plt.subplots(4, 1, figsize=(7, 7))
for i in range(4):
    axs[i].plot(t, composite_wave * windows_functions[i])
    axs[i].set_title(windows_names[i])
plt.tight_layout()
plt.show()