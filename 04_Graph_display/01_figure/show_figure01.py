import matplotlib.pyplot as plt

# Data preparation
x = [1, 2, 3, 4, 5]
y = [2, 3, 7, 1, 5]

# Create a graph
fig, ax = plt.subplots()
ax.plot(x, y, color='b', marker='o', linestyle='-', linewidth=2, markersize=8)

# Set title and label
ax.set_title('Single Graph')
ax.set_xlabel('X-axis')
ax.set_ylabel('Y-axis')

# display graph
plt.show()
