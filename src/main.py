from graph import Graph

print("=== TESTE DAS FUNCIONALIDADES ===\n")

print("1. Gerando grafo aleatório...")
n_nodes = 50
n_edges = 200
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

if not grafo_carregado.is_connected():
    componentes = grafo_carregado.components()
    print(f"   - Componentes: {len(componentes)}")

print("\n5. Calculando centralidades...")
closeness = grafo_carregado.closeness_centrality()
betweenness = grafo_carregado.betweenness_centrality()

print("\n   Top 5 Centralidade de Proximidade:")
for node, score in sorted(closeness.items(), key=lambda x: x[1], reverse=True)[:5]:
    print(f"   - {node}: {score:.4f}")

print("\n   Top 5 Centralidade de Intermediação:")
for node, score in sorted(betweenness.items(), key=lambda x: x[1], reverse=True)[:5]:
    print(f"   - {node}: {score:.4f}")

print("\n=== TESTE CONCLUÍDO ===")
