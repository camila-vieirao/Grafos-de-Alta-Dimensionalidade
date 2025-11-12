from graph import Graph

graph = Graph.load_pajek("grafo_50n_200e_undir.net", directed=False)
graph.save_pajek("grafo_50n_200e_undir_output.net")

print("Connected?", graph.is_connected())
components = graph.components()
print("Component count:", len(components))
print("Eulerian?", graph.is_eulerian())
print("Cyclic?", graph.is_cyclic())

# Centralities
closeness = graph.closeness_centrality()
betweenness = graph.betweenness_centrality()
print("Closeness centrality (top 10):", sorted(closeness.items(), key=lambda x: x[1], reverse=True)[:10])
print("Betweenness centrality (top 10):", sorted(betweenness.items(), key=lambda x: x[1], reverse=True)[:10])
