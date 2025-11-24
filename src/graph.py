import heapq
import random
from collections import deque, defaultdict

class Graph:
    def __init__(self, directed=False):
        self.directed = directed
        self.adj = defaultdict(dict) # defaultdict para evitar KeyError
        self.nodes_set = set() # set para evitar duplicações

    def add_node(self, v):
        self.nodes_set.add(v)
        if v not in self.adj:
            self.adj[v] = {} # cria um dict vazio para o vértice v

    def add_edge(self, u, v, weight=1):
        self.add_node(u)
        self.add_node(v)
        self.adj[u][v] = weight
        if not self.directed: # se o grafo não for direcionado, tambem adiciona a aresta inversa para manter a simetria
            self.adj[v][u] = weight

    def nodes(self):
        return list(self.nodes_set)

    def edges(self):
        # obs: yield permite iterar pelas arestas uma a uma, sem ter de montar e guardar uma lista com todas elas antes de começar a usar
        seen = set()
        for u in self.adj: # percorre todos os vértices do grafo
            for v, w in self.adj[u].items():
                if self.directed:
                    yield (u, v, w) # se o grafo for direcionado, retorna a aresta direcionada
                else:
                    edge = tuple(sorted((u, v)))
                    if edge not in seen:
                        seen.add(edge)
                        yield (u, v, w) # se o grafo não for direcionado, retorna a aresta não direcionada

    def save_pajek(self, filename):
        nodes = sorted(self.nodes()) # ordena os vértices do grafo pq o Pajek espera que os vértices sejam ordenados
        node_index = {} 
        for i, node in enumerate(nodes):
            node_index[node] = i + 1

        with open(filename, 'w', encoding='utf-8') as f:
            f.write(f'*Vertices {len(nodes)}\n')
            for node in nodes:
                f.write(f'{node_index[node]} "{node}"\n')

            if self.directed:
                f.write('*Arcs\n')
            else:
                f.write('*Edges\n')

            for u, v, w in self.edges():
                f.write(f'{node_index[u]} {node_index[v]} {w}\n')

    @classmethod
    def load_pajek(cls, filename):
        g = None
        node_index = {}

        with open(filename, 'r', encoding='utf-8') as f:
            lines = [line.strip() for line in f if line.strip()]

        i = 0
        while i < len(lines):
            line = lines[i]

            if line.startswith('*Vertices'):
                num_vertices = int(line.split()[1])
                i += 1
                for _ in range(num_vertices):
                    if i < len(lines):
                        parts = lines[i].split(None, 1)
                        idx = int(parts[0])
                        name = parts[1].strip('"') if len(parts) > 1 else str(idx)
                        node_index[idx] = name
                        i += 1

            elif line.startswith('*Edges'):
                if g is None:
                    g = cls(directed=False)
                for node in node_index.values():
                    g.add_node(node)
                i += 1

            elif line.startswith('*Arcs'):
                if g is None:
                    g = cls(directed=True)
                for node in node_index.values():
                    g.add_node(node)
                i += 1

            elif line[0].isdigit():
                parts = line.split()
                if len(parts) >= 2:
                    u_idx = int(parts[0])
                    v_idx = int(parts[1])
                    weight = int(parts[2]) if len(parts) > 2 else 1
                    u = node_index.get(u_idx, str(u_idx))
                    v = node_index.get(v_idx, str(v_idx))
                    g.add_edge(u, v, weight)
                i += 1
            else:
                i += 1

        return g

    def is_connected(self):
        if not self.nodes_set:
            return True

        start = next(iter(self.nodes_set)) # pega o primeiro no do grafo
        visited = set()
        queue = deque([start]) # fila para busac
        visited.add(start)

        while queue: # enquanto a fila não estiver vazia, continua a busca em largura
            node = queue.popleft()
            neighbors = set(self.adj[node].keys()) # pega os vizinhos do no atual

            if self.directed:
                for other in self.adj: # para grafos direcionados, pega os vizinhos de todos os nos
                    if node in self.adj[other]: # se o no atual está ligado a outro no, adiciona o outro no como vizinho
                        neighbors.add(other)

            for neighbor in neighbors: # para cada vizinho, se ele não foi visitado, adiciona na fila e marca como visitado
                if neighbor not in visited:
                    visited.add(neighbor)
                    queue.append(neighbor)

        return len(visited) == len(self.nodes_set) # se o número de nos visitados é igual ao número de nos do grafo, o grafo é conexo

    def components(self):
        remaining = set(self.nodes_set)
        components = []

        while remaining: # enquanto ainda houver nos não visitados, continua a busca em largura
            start = next(iter(remaining))
            visited = set()
            queue = deque([start])
            visited.add(start)

            while queue: # enquanto a fila não estiver vazia
                node = queue.popleft() # pega o primeiro no da fila
                neighbors = set(self.adj[node].keys())

                if self.directed:
                    for other in self.adj:
                        if node in self.adj[other]:
                            neighbors.add(other)

                for neighbor in neighbors:
                    if neighbor not in visited and neighbor in remaining:
                        visited.add(neighbor) # marca o no como visitado
                        queue.append(neighbor) # adiciona o no na fila

            components.append(visited) # adiciona o componente conexo encontrado à lista de componentes
            remaining -= visited

        return components

    def is_eulerian(self): # tem que ser conexo e todos os nos devem ter grau par
        if not self.directed:
            for node in self.nodes_set: # verifica se sao par
                if len(self.adj[node]) % 2 != 0:
                    return False
            return self.is_connected() #depois ve se eh conexo
        else:
            # para grafos direcionados: o grau de entrada deve ser igualao grau de saida
            for node in self.nodes_set:
                in_degree = sum(1 for other in self.adj if node in self.adj[other])
                out_degree = len(self.adj[node])
                if in_degree != out_degree:
                    return False
            return self.is_connected()

    def is_cyclic(self):
        if self.directed:
            visited = set()
            rec_stack = set()

            def has_cycle(node):
                visited.add(node)
                rec_stack.add(node)

                for neighbor in self.adj[node]:
                    if neighbor not in visited:
                        if has_cycle(neighbor):
                            return True
                    elif neighbor in rec_stack:
                        return True

                rec_stack.remove(node)
                return False

            for node in self.nodes_set:
                if node not in visited:
                    if has_cycle(node):
                        return True
            return False
        else:
            visited = set()

            def has_cycle(node, parent):
                visited.add(node)

                for neighbor in self.adj[node]:
                    if neighbor not in visited:
                        if has_cycle(neighbor, node):
                            return True
                    elif neighbor != parent:
                        return True

                return False

            for node in self.nodes_set:
                if node not in visited:
                    if has_cycle(node, None):
                        return True
            return False

    def dijkstra(self, source):
        distances = {node: float('inf') for node in self.nodes_set}
        distances[source] = 0
        pq = [(0, source)]

        while pq:
            current_dist, current = heapq.heappop(pq)

            if current_dist > distances[current]:
                continue

            for neighbor, weight in self.adj[current].items():
                distance = current_dist + weight

                if distance < distances[neighbor]:
                    distances[neighbor] = distance
                    heapq.heappush(pq, (distance, neighbor))

        return distances

    def closeness_centrality(self): # algoritmo de freeman
        centrality = {}
        n = len(self.nodes_set)

        for node in self.nodes_set:
            distances = self.dijkstra(node) # calcula as distancias de todos os nos para o no atual
            total_dist = 0 # soma das distancias de todos os nos para o no atual
            reachable = 0 # qtos nos sao alcançaveis a partir do no atual

            for other, dist in distances.items(): 
                if other != node and dist < float('inf'):
                    total_dist += dist
                    reachable += 1

            if total_dist > 0:
                centrality[node] = (reachable / (n - 1)) * (reachable / total_dist) # formula de freeman
            else:
                centrality[node] = 0.0 # se nao houver nos alcançaveis, a centralidade eh 0

        return centrality

    def betweenness_centrality(self): # alg de brandes
        centrality = {node: 0.0 for node in self.nodes_set} # inicia como 0 para todos
        nodes = list(self.nodes_set)

        for s in nodes:
            stack = []
            predecessors = {node: [] for node in nodes}
            sigma = {node: 0.0 for node in nodes} # qtos caminhos saem de s chegam em cada no
            dist = {node: float('inf') for node in nodes}
            sigma[s] = 1.0
            dist[s] = 0.0

            pq = [(0.0, s)] # fila de prioridade para o algoritmo de Dijkstra
            while pq:
                d, v = heapq.heappop(pq) # d eh a distancia do no atual, v eh o no atual
                if d > dist[v]: # se a distancia do no atual for maior que a distancia do no visitado, pula para o proximo no
                    continue
                stack.append(v) # adiciona o no visitado na pilha

                for w, weight in self.adj[v].items(): # w eh o no vizinho de v, weight eh o peso da aresta entre v e w
                    new_dist = dist[v] + weight
                    if dist[w] > new_dist:
                        dist[w] = new_dist
                        heapq.heappush(pq, (new_dist, w))
                        sigma[w] = sigma[v]
                        predecessors[w] = [v]
                    elif dist[w] == new_dist:
                        sigma[w] += sigma[v]
                        predecessors[w].append(v)

            delta = {node: 0.0 for node in nodes}
            while stack:
                w = stack.pop()
                for v in predecessors[w]: # para cada no predecessor de w, calcula o delta
                    if sigma[w] != 0:
                        delta[v] += (sigma[v] / sigma[w]) * (1.0 + delta[w]) # quanto v contribui para caminhos ue passam por w
                if w != s:
                    centrality[w] += delta[w]

        n = len(nodes)
        if n > 2:
            scale = 1.0 / ((n - 1) * (n - 2)) # normaliza o resultado
            for node in centrality:
                centrality[node] *= scale

        return centrality

    @classmethod
    def random_graph(cls, n_nodes, n_edges, connected=False):
        g = cls(directed=False) # aqui criamos o grafo, não direcionado por padrão
        nodes = [f"Node {i+1}" for i in range(n_nodes)] # cria os nos como Node x em uma lista

        for node in nodes:
            g.add_node(node)

        #verificacoes para garantir que o grafo escolhido é valido
        if n_edges < 0:
            raise ValueError("Number of edges cannot be negative")
        max_total_edges = n_nodes * (n_nodes - 1) // 2 # o numero maximo de arestas em um grafo completo é n * (n - 1) / 2
        if n_edges > max_total_edges:
            raise ValueError("Requested edges exceed the maximum for this graph size")
        if n_nodes == 0:
            if n_edges > 0:
                raise ValueError("Cannot create edges without nodes")
            return g

        candidate_nodes = nodes
        if not connected:
            if n_nodes < 2:
                if n_edges > 0:
                    raise ValueError("Cannot create a disconnected graph with fewer than two nodes")
                return g
            candidate_nodes = nodes[:-1] # pega todos os nos menos o ultimo para garantir que o grafo seja desconexo (esse ultimo no nao sera usado para adicionar arestas, dessa forma, fica sobrando um no)
            max_edges = len(candidate_nodes) * (len(candidate_nodes) - 1) // 2
            if n_edges > max_edges:
                raise ValueError("Requested edges exceed the maximum for a disconnected graph")

        edges_added = 0
        edges_set = set()

        if connected:
            for i in range(n_nodes - 1): # vai adicionar arestas de um no para o seguinte
                u = nodes[i] # u é o no atual
                v = nodes[i + 1] # v é o no seguinte
                weight = random.randint(1, 10)
                g.add_edge(u, v, weight)
                edges_added += 1
                edges_set.add(tuple(sorted([u, v])))

        while edges_added < n_edges:
            if len(candidate_nodes) < 2:
                break
            u = random.choice(candidate_nodes)
            v = random.choice(candidate_nodes)

            if u != v:
                edge = tuple(sorted([u, v]))
                if edge not in edges_set:
                    weight = random.randint(1, 10)
                    g.add_edge(u, v, weight)
                    edges_set.add(edge)
                    edges_added += 1

        return g
