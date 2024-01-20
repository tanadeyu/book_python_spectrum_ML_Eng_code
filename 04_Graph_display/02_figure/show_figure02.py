import matplotlib.pyplot as plt

# Preparing the data
x = [1, 2, 3, 4, 5]
y1 = [2, 3, 4, 1, 5]
y2 = [2, 4, 6, 8, 10]

# Create graph, share the X axis
fig, axs = plt.subplots(2, sharex=True) 

# Plot in subplot1
axs[0].plot(x, y1, color='b', marker='o', linestyle='-', linewidth=2, markersize=8)
axs[0].set_ylabel('Y-axis 1')

# Plot in subplot2
axs[1].plot(x, y2, color='g', marker='s', linestyle='--', linewidth=2, markersize=8)
axs[1].set_xlabel('X-axis')
axs[1].set_ylabel('Y-axis 2')

# display graph
plt.tight_layout()
plt.show()
