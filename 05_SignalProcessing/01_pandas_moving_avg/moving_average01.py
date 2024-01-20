import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# Preparation of spectral data (example)
# Wavelength data in 10 intervals from 400 to 1000 [nm]
wavelengths = np.arange(400, 1000, 10)        
intensity = np.random.rand(len(wavelengths)) # Dummy strength data

# Define moving average window size
window_size = 5

# Store data in DataFrame type
data = pd.DataFrame({'Wavelength': wavelengths, 'Intensity': intensity})

# Calculate moving average
data['Moving Average'] = \
  data['Intensity'].rolling(window=window_size, min_periods=1).mean()

# Draw graph
plt.figure(figsize=(7, 5))
plt.plot(data['Wavelength'], data['Intensity'], \
         label='Original Spectrum', color='b')
plt.plot(data['Wavelength'], data['Moving Average'], \
         label='Moving Average', color='r', linewidth=3, linestyle='--')
plt.xlabel('Wavelength [nm]')
plt.ylabel('Intensity')
plt.title('Spectrum with Moving Average')
plt.legend()
plt.show()
