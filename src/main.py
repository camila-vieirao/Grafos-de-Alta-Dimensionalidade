from graph import Graph
import networkx as nx
import matplotlib.pyplot as plt

print("=== TESTE DAS FUNCIONALIDADES ===\n")

print("1. Gerando grafo aleatório...")
n_nodes = 8
n_edges = 15
connected = True
grafo = Graph.random_graph(n_nodes=n_nodes, n_edges=n_edges, connected=connected)
print(f"nos: {n_nodes}")
print(f"arestas: {n_edges}")
print(f"conexo: {'Sim' if connected else 'Não'}")

print("\n2. Salvando em formato Pajek...")
grafo.save_pajek("teste.net")

print("\n3. Carregando grafo do arquivo...")
grafo_carregado = Graph.load_pajek("teste.net")

print("\n4. Verificando propriedades:")
print(f"   - Conexo: {'Sim' if grafo_carregado.is_connected() else 'Não'}")
print(f"   - Euleriano: {'Sim' if grafo_carregado.is_eulerian() else 'Não'}")
print(f"   - Cíclico: {'Sim' if grafo_carregado.is_cyclic() else 'Não'}")


componentes = grafo_carregado.components()
print(f"   - Componentes: {len(componentes)}")

print("\n5. Calculando centralidades...")
closeness = grafo_carregado.closeness_centrality()
betweenness = grafo_carregado.betweenness_centrality()

print("\n   Top 8 Centralidade de Proximidade:")
for node, score in sorted(closeness.items(), key=lambda x: x[1], reverse=True)[:8]:
    print(f"   - {node}: {score:.4f}")

print("\n   Top 8 Centralidade de Intermediação:")
for node, score in sorted(betweenness.items(), key=lambda x: x[1], reverse=True)[:8]:
    print(f"   - {node}: {score:.4f}")


# ------------- PLOTAGEM DO GRAFO -------------
nx_graph = nx.DiGraph() if grafo_carregado.directed else nx.Graph()
nx_graph.add_nodes_from(grafo_carregado.nodes())
for u, v, w in grafo_carregado.edges():
    nx_graph.add_edge(u, v, weight=w)

pos = nx.shell_layout(nx_graph)
nx.draw(nx_graph, pos=pos, with_labels=True)
edge_labels = nx.get_edge_attributes(nx_graph, "weight")
nx.draw_networkx_edge_labels(nx_graph, pos, edge_labels=edge_labels)
plt.show()

print("\n=== TESTE CONCLUÍDO ===")
