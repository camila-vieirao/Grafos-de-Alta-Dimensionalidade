from graph import Graph

# Load a Pajek graph
graph = Graph.load_pajek("grafo_500n_2000e_undir.net", directed=False)

# Save the graph back to disk
graph.save_pajek("grafo_500n_2000e_undir_output.net")

# Connectivity and components
print("Connected?", graph.is_connected())
components = graph.components()
print("Component count:", len(components))

# Eulerian
print("Eulerian?", graph.is_eulerian())

# Cyclic?
print("Cyclic?", graph.is_cyclic())

# Centralities
closeness = graph.closeness_centrality()
betweenness = graph.betweenness_centrality()
print("Closeness centrality:", closeness)
print("Betweenness centrality:", betweenness)

# Large random graph generator
large_graph = Graph.random_graph(5000, 20000, directed=False, connected=True, seed=42)
large_graph.save_pajek("big_graph.net")
