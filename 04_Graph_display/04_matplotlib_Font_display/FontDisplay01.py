import matplotlib.pyplot as plt
import numpy as np

# Fonts setting
#plt.rcParams['font.family'] = 'serif'
plt.rcParams['font.serif'] = 'PT Serif'

#plt.rcParams['font.family'] = 'PT Serif'
#plt.rcParams['font.family'] = 'Times New Roman'

# Japanese Fonts
#plt.rcParams['font.family'] = 'Noto Sans JP'
#plt.rcParams['font.family'] = 'Noto Serif JP'
#plt.rcParams['font.family'] = 'Yu Gothic'
#plt.rcParams['font.family'] = 'MS Gothic'

# Creating a graph
x = np.linspace(-1, 1, 100)
y = np.exp(x)
plt.plot(x, y, label="Exponential function", ls='--',lw=3)

# tangent line at x=0
plt.plot(0, 1, 'ro')  # dot at x=0
plt.plot(x, 1 + x, label="tangent line at x=0",lw=3)

# Title ans Label set
plt.title("Graph Title")
plt.xlabel("X-Label")
plt.ylabel("Y-Label")
plt.legend()

# Display graph
plt.tight_layout()
plt.show()