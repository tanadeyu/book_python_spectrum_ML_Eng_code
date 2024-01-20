import numpy as np
import matplotlib.pyplot as plt

# Generating two Gaussian data
x = np.linspace(0, 10, 100)
y1 = np.exp(-(x - 6)**2 / 2)
y2 = np.exp(-(x - 7)**2 / 2)

# Drawing a graph
fig, ax = plt.subplots()
ax.plot(x, y1, label='Gaussian 1',lw=3)
ax.plot(x, y2, label='Gaussian 2',lw=3,ls="dashed")
ax.set_xlabel('Wavenumber')
ax.set_ylabel('Intensity')
ax.legend()

# Calculating the correlation coefficient
corr_coef = np.corrcoef(y1, y2)[0, 1]
print('Correlation coefficient:', corr_coef)
