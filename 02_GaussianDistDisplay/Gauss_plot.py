import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import norm

# Set Gaussian distribution parameters
mu, sigma = 0, 0.1

# Generate x-axis value
x = np.linspace(-1, 1, 100)

# Calculate the formula for Gaussian distribution
y = norm.pdf(x, mu, sigma)

# Display the graph
plt.plot(x, y)
plt.show()
