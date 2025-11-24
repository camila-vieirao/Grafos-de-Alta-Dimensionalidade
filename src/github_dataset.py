import csv
from collections import Counter
import matplotlib.pyplot as plt
import networkx as nx
from graph import Graph

# TRATAMENTO DO DATASET -----------------------------------------------------------------------------------

def carregar_dataset_github():
    print("=== CARREGANDO DATASET GITHUB ===\n")
    
    usuarios = {}
    print("Carregando usuários...")
    with open("musae_git_target.csv", 'r', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        for row in reader:
            id_usuario = row['id']
            nome = row.get('name', id_usuario)
            usuarios[id_usuario] = nome
    
    print(f"Carregados {len(usuarios)} usuários")
    
    print("\nCarregando conexões...")
    conexoes = []
    with open("musae_git_edges.csv", 'r', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        for row in reader:
            id1 = row['id_1']
            id2 = row['id_2']
            conexoes.append((id1, id2))
    
    print(f"Carregadas {len(conexoes)} conexões")
    

    print("\nSelecionando os 5000 usuários mais conectados...")
    contador = Counter() # o counter eh uma forma simples de contar quantas vezes cada no aparece nas conexoes
    for id1, id2 in conexoes:
        contador[id1] += 1
        contador[id2] += 1
    
    top_usuarios = [id_u for id_u, _ in contador.most_common(5000)]
    usuarios_selecionados = set(top_usuarios)
    
    print("\nFiltrando 20000 conexões válidas...")
    conexoes_validas = []
    for id1, id2 in conexoes: # pra cada conexao, verifica se os dois nos estao na lista de usuarios selecionados
        if id1 in usuarios_selecionados and id2 in usuarios_selecionados: # se os dois nos estao na lista de usuarios selecionados, adiciona a conexao na lista de conexoes validas
            conexoes_validas.append((id1, id2))
            if len(conexoes_validas) >= 20000:
                break
    
    print(f"Selecionadas {len(conexoes_validas)} conexões")
    
    # FIM DO TRATAMENTO DO DATASET -----------------------------------------------------------------------------------
    
    print("\nConstruindo grafo...")
    grafo = Graph(directed=True)
    
    for id_usuario in usuarios_selecionados: # para cada usuario selecionado, adiciona um no no grafo
        nome = usuarios.get(id_usuario, f"User_{id_usuario}") # se o usuario nao tiver nome, usa o id como nome
        grafo.add_node(nome)
    
    for id1, id2 in conexoes_validas: # para cada conexao valida, adiciona uma aresta no grafo
        nome1 = usuarios.get(id1, f"User_{id1}")
        nome2 = usuarios.get(id2, f"User_{id2}")
        grafo.add_edge(nome1, nome2, weight=1)
    
    print("\nGrafo criado:")
    print(f"- {len(grafo.nodes())} nós")
    print(f"- {len(list(grafo.edges()))} arestas")
    
    return grafo

def analisar_grafo(grafo, arquivo_saida="github_graph.net"):
    print("\n=== ANÁLISE DO GRAFO ===\n")
    
    print(f"Salvando em formato Pajek: {arquivo_saida}")
    grafo.save_pajek(arquivo_saida)
    
    print("\nPropriedades básicas:")
    print(f"- Nós: {len(grafo.nodes())}")
    print(f"5 primeiros nós: {grafo.nodes()[:5]}")
    print(f"- Arestas: {len(list(grafo.edges()))}")
    print(f"5 primeiras arestas: {list(grafo.edges())[:5]}")
    print(f"- Direcionado: {'Sim' if grafo.directed else 'Não'}")
    
    print("\nConectividade:")
    if grafo.is_connected():
        print("- O grafo é conexo")
    else:
        print("- O grafo NÃO é conexo")
        componentes = grafo.components()
        print(f"- Número de componentes: {len(componentes)}")
        print(f"5 primeiros nós da maior componente: {list(max(componentes, key=len))[:5]}")
    
    print("\nPropriedades:")
    print(f"- Euleriano: {'Sim' if grafo.is_eulerian() else 'Não'}")
    print(f"- Cíclico: {'Sim' if grafo.is_cyclic() else 'Não'}")
    
    print("\nCalculando centralidades...")
    closeness = grafo.closeness_centrality()
    betweenness = grafo.betweenness_centrality()
    
    print("\nTop 10 usuários por Centralidade de Proximidade:")
    top_closeness = sorted(closeness.items(), key=lambda x: x[1], reverse=True)[:10]
    for i, (usuario, valor) in enumerate(top_closeness, 1):
        print(f"  {i}. {usuario}: {valor:.6f}")
    
    print("\nTop 10 usuários por Centralidade de Intermediação:")
    top_betweenness = sorted(betweenness.items(), key=lambda x: x[1], reverse=True)[:10]
    for i, (usuario, valor) in enumerate(top_betweenness, 1):
        print(f"  {i}. {usuario}: {valor:.6f}")
    
    # PLOTAGEM DOS 20 USUARIOS MAIS CENTRALIZADOS (codigo gerado com IA) -----------------------------------------------------------------------------------
    sample_nodes = [usuario for usuario, _ in sorted(betweenness.items(), key=lambda x: x[1], reverse=True)[:20]]
    sample_graph = nx.DiGraph()
    sample_graph.add_nodes_from(sample_nodes)
    for u, v, w in grafo.edges():
        if u in sample_nodes and v in sample_nodes:
            sample_graph.add_edge(u, v, weight=w)
    if sample_graph.number_of_edges() > 0:
        pos = nx.spring_layout(sample_graph, seed=42)
        edge_labels = nx.get_edge_attributes(sample_graph, "weight")
        plt.figure(figsize=(10, 7))
        nx.draw(sample_graph, pos=pos, with_labels=True, node_size=500, font_size=8)
        nx.draw_networkx_edge_labels(sample_graph, pos, edge_labels=edge_labels, font_size=7)
        plt.title("Sample of top betweenness nodes")
        plt.show()

    print("\n=== ANÁLISE CONCLUÍDA ===")

if __name__ == "__main__":
    grafo = carregar_dataset_github()
    analisar_grafo(grafo)
