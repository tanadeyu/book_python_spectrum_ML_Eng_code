import matplotlib.pyplot as plt
import networkx as nx

# Defining functions for creating neural network diagrams
def create_neural_network_figure(num_input, num_hidden_nodes_list, num_output):
    G = nx.DiGraph()

    # Create node
    # Input layer
    for i in range(1, num_input + 1):
        G.add_node("Input{}".format(i), pos=(0, -i -1))
    # Hidden layer (middle layer)
    max_hidden_nodes = max(num_hidden_nodes_list)
    num_hidden_layers = len(num_hidden_nodes_list)
    for j in range(num_hidden_layers):
        for k in range(1, num_hidden_nodes_list[j] + 1):
            G.add_node("H{}_{}".format(j + 1, k), pos=(j + 1, (max_hidden_nodes - num_hidden_nodes_list[j]) / 2 - k))
    # Output layer
    for l in range(1, num_output + 1):
        G.add_node("Output{}".format(l), pos=(num_hidden_layers + 1, -l -1))

    # Creating edges
    for i in range(1, num_input + 1):
        for j in range(1, num_hidden_nodes_list[0] + 1):
            G.add_edge("Input{}".format(i), "H1_{}".format(j))
    for idx in range(len(num_hidden_nodes_list) - 1):
        for j in range(1, num_hidden_nodes_list[idx] + 1):
            for k in range(1, num_hidden_nodes_list[idx + 1] + 1):
                G.add_edge("H{}_{}".format(idx + 1, j), "H{}_{}".format(idx + 2, k))
    for l in range(1, num_output + 1):
        for j in range(1, num_hidden_nodes_list[-1] + 1):
            G.add_edge("H{}_{}".format(len(num_hidden_nodes_list), j), "Output{}".format(l))

    pos = nx.get_node_attributes(G, 'pos')
    nx.draw(G, pos, with_labels=True, node_size=3500, edgecolors="black", \
            node_color="white", font_weight='bold', arrowsize=25)
    plt.show()

# Creating a diagram (calling a function)
create_neural_network_figure(2, [4, 4], 2)

