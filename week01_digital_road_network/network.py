import networkx as nx
import matplotlib.pyplot as plt

# Creates an empty graph
G = nx.Graph()

# Add points (nodes)
G.add_nodes_from(["A", "B", "N", "S", "L"])

# Add roads (edges)
G.add_edges_from([
    ("A", "B"),
    ("N", "S"),
    ("N", "L"),
    ("S", "L")
])

# Draw the digital network
pos = {
    "A": (0, 1),
    "B": (4, 1),
    "N": (2, 3),
    "S": (2, 0),
    "L": (3.5, 1)
}

nx.draw(
    G,
    pos,
    with_labels=True,
    node_size=1500,
    font_size=12
)

plt.title("FloodPath Digital Road Network")
plt.savefig("network.png")
plt.show()