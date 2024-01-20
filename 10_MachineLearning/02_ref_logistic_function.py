import numpy as np
import matplotlib.pyplot as plt

def logistic_function(x):
    return 1 / (1 + np.exp(-x))

# Generate 100 points in the range -6 to 6
x = np.linspace(-6, 6, 100)  
y = logistic_function(x)

plt.plot(x, y)
plt.title('Logistic Function')
plt.xlabel('x')
plt.ylabel('y')
plt.grid(True)
plt.show()