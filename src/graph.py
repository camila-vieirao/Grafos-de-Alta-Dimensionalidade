import heapq
import random
from collections import deque, defaultdict

class Graph:
    def __init__(self, directed=False):
        self.directed = directed
        # adjacency: node -> dict(neighbor -> weight)
        self.adj = defaultdict(dict)
        self.nodes_set = set()

    # -------------------------
    # Basic operations
    # -------------------------
    def add_node(self, v):
        self.nodes_set.add(v)
        _ = self.adj[v]  # ensure entry

    def add_edge(self, u, v, weight=1.0):
        self.add_node(u)
        self.add_node(v)
        self.adj[u][v] = weight
        if not self.directed:
            self.adj[v][u] = weight

    def remove_edge(self, u, v):
        if v in self.adj.get(u, {}):
            del self.adj[u][v]
        if not self.directed and u in self.adj.get(v, {}):
            del self.adj[v][u]

    def degree(self, v):
        return len(self.adj.get(v, {}))

    def indegree(self, v):
        if not self.directed:
            return self.degree(v)
        cnt = 0
        for u in self.adj:
            if v in self.adj[u]:
                cnt += 1
        return cnt

    def outdegree(self, v):
        return self.degree(v)

    def nodes(self):
        return list(self.nodes_set)

    def edges(self):
        seen = set()
        for u in self.adj:
            for v, w in self.adj[u].items():
                if self.directed:
                    yield (u, v, w)
                else:
                    key = tuple(sorted((u, v)))
                    if key not in seen:
                        seen.add(key)
                        yield (u, v, w)

    # -------------------------
    # Pajek I/O
    # -------------------------
    def save_pajek(self, filename):
        nodes = sorted(self.nodes())
        index = {v: i + 1 for i, v in enumerate(nodes)}
        with open(filename, 'w', encoding='utf8') as f:
            f.write(f'*Vertices {len(nodes)}\n')
            for v in nodes:
                f.write(f'{index[v]} "{v}"\n')
            if self.directed:
                f.write('*Arcs\n')
            else:
                f.write('*Edges\n')
            for u, v, w in self.edges():
                f.write(f'{index[u]} {index[v]} {w}\n')

    @classmethod
    def load_pajek(cls, filename, directed=None):
        with open(filename, 'r', encoding='utf8') as f:
            raw_lines = [ln.strip() for ln in f]
        lines = []
        for raw in raw_lines:
            if not raw:
                continue
            stripped = raw.split('%', 1)[0].strip()
            if not stripped:
                continue
            lines.append(stripped)
        base_directed = False if directed is None else directed
        g = cls(directed=base_directed)
        idx2name = {}
        i = 0
        total = len(lines)
        while i < total:
            clean = lines[i].split('//', 1)[0].strip()
            if not clean:
                i += 1
                continue
            lower = clean.lower()
            if lower.startswith('*vertices'):
                parts = clean.split()
                count = int(parts[1]) if len(parts) > 1 else 0
                i += 1
                read = 0
                while read < count and i < total:
                    vertex_line = lines[i].split('//', 1)[0].strip()
                    if vertex_line:
                        vertex_parts = vertex_line.split(None, 1)
                        idx = int(vertex_parts[0])
                        if len(vertex_parts) > 1:
                            name = vertex_parts[1].strip().strip('"')
                        else:
                            name = str(idx)
                        idx2name[idx] = name
                        g.add_node(name)
                        read += 1
                    i += 1
                continue
            if lower.startswith('*edges'):
                if directed is None:
                    g.directed = False
                elif directed:
                    raise ValueError('Pajek file declares undirected edges but directed=True was requested.')
                else:
                    g.directed = False
                i += 1
                continue
            if lower.startswith('*arcs'):
                if directed is None:
                    g.directed = True
                elif not directed:
                    raise ValueError('Pajek file declares directed arcs but directed=False was requested.')
                else:
                    g.directed = True
                i += 1
                continue
            parts = clean.split()
            if len(parts) >= 2:
                a = int(parts[0])
                b = int(parts[1])
                if len(parts) >= 3:
                    weight_value = float(parts[2])
                    weight = int(weight_value) if weight_value.is_integer() else weight_value
                else:
                    weight = 1
                name_a = idx2name.get(a, str(a))
                name_b = idx2name.get(b, str(b))
                g.add_edge(name_a, name_b, weight)
            i += 1
        return g

    # -------------------------
    # Connectivity & components
    # -------------------------
    def _bfs(self, start, use_undirected=False):
        seen = set()
        q = deque([start])
        seen.add(start)
        while q:
            u = q.popleft()
            # neighbors in directed or undirected view
            nbrs = set(self.adj[u].keys())
            if use_undirected:
                # include incoming edges
                for x in self.adj:
                    if u in self.adj[x]:
                        nbrs.add(x)
            for v in nbrs:
                if v not in seen:
                    seen.add(v)
                    q.append(v)
        return seen

    def is_connected(self):
        if not self.nodes_set:
            return True
        # For undirected: standard connectivity
        if not self.directed:
            start = next(iter(self.nodes_set))
            seen = self._bfs(start, use_undirected=False)
            return len(seen) == len(self.nodes_set)
        # For directed: check weak connectivity (problem asks: consider weakly connected components for directed)
        start = next(iter(self.nodes_set))
        seen = self._bfs(start, use_undirected=True)
        return len(seen) == len(self.nodes_set)

    def components(self):
        # returns list of sets (weak components if directed)
        remaining = set(self.nodes_set)
        comps = []
        use_undirected = self.directed
        while remaining:
            start = next(iter(remaining))
            comp = self._bfs(start, use_undirected=use_undirected)
            comps.append(comp)
            remaining -= comp
        return comps

    # -------------------------
    # Eulerian
    # -------------------------
    def is_eulerian(self):
        # Check Eulerian circuit condition
        # Undirected: graph is connected (ignoring isolated vertices?) and all degrees even
        # Directed: every vertex in-degree == out-degree and all vertices with degree>0 in single strongly connected component
        # Here we'll check for Eulerian circuit
        # Consider only nodes with degree>0
        non_isolated = {v for v in self.nodes_set if (self.degree(v) > 0 or (self.directed and (self.indegree(v) > 0 or self.outdegree(v)>0)))}
        if not non_isolated:
            return False  # trivial: no edges -> not Eulerian by convention
        if not self.directed:
            # connected in undirected view?
            start = next(iter(non_isolated))
            seen = self._bfs(start, use_undirected=False)
            if not non_isolated.issubset(seen):
                return False
            # all degrees even
            for v in non_isolated:
                if self.degree(v) % 2 != 0:
                    return False
            return True
        else:
            # in-degree == out-degree for all nodes
            for v in non_isolated:
                if self.indegree(v) != self.outdegree(v):
                    return False
            # need strongly connected on nodes with edges.
            # Quick check: make undirected connected
            start = next(iter(non_isolated))
            seen = self._bfs(start, use_undirected=True)
            if not non_isolated.issubset(seen):
                return False
            # Strong connectivity check (Kosaraju) on subgraph of non_isolated
            # Build adjacency lists restricted
            def kosaraju():
                visited = set()
                order = []
                def dfs(u):
                    visited.add(u)
                    for v in self.adj[u]:
                        if v in non_isolated and v not in visited:
                            dfs(v)
                    order.append(u)
                for u in non_isolated:
                    if u not in visited:
                        dfs(u)
                # transpose graph
                rev = defaultdict(list)
                for u in non_isolated:
                    for v in self.adj[u]:
                        if v in non_isolated:
                            rev[v].append(u)
                visited2 = set()
                def dfs2(u):
                    visited2.add(u)
                    for v in rev[u]:
                        if v not in visited2:
                            dfs2(v)
                # process in reverse order
                dfs2(order[-1])
                return visited2 == non_isolated
            return kosaraju()

    # -------------------------
    # Cycle detection
    # -------------------------
    def is_cyclic(self):
        if self.directed:
            # DFS with recursion stack
            visited = set()
            recstack = set()
            def dfs(u):
                visited.add(u)
                recstack.add(u)
                for v in self.adj[u]:
                    if v not in visited:
                        if dfs(v):
                            return True
                    elif v in recstack:
                        return True
                recstack.remove(u)
                return False
            for node in self.nodes_set:
                if node not in visited:
                    if dfs(node):
                        return True
            return False
        else:
            visited = set()
            def dfs(u, parent):
                visited.add(u)
                for v in self.adj[u]:
                    if v not in visited:
                        if dfs(v, u):
                            return True
                    elif v != parent:
                        return True
                return False
            for node in self.nodes_set:
                if node not in visited:
                    if dfs(node, None):
                        return True
            return False

    # -------------------------
    # Shortest paths (Dijkstra)
    # -------------------------
    def _dijkstra(self, source):
        dist = {v: float('inf') for v in self.nodes_set}
        dist[source] = 0.0
        pq = [(0.0, source)]
        while pq:
            d, u = heapq.heappop(pq)
            if d > dist[u]:
                continue
            for v, w in self.adj[u].items():
                nd = d + w
                if nd < dist[v]:
                    dist[v] = nd
                    heapq.heappush(pq, (nd, v))
        return dist

    # -------------------------
    # Closeness centrality (proximity)
    # -------------------------
    def closeness_centrality(self):
        n = len(self.nodes_set)
        closeness = {}
        for u in self.nodes_set:
            dist = self._dijkstra(u)
            total = 0.0
            reachable = 0
            for v, d in dist.items():
                if u == v:
                    continue
                if d < float('inf'):
                    total += d
                    reachable += 1
            if total > 0 and reachable > 0:
                # standard normalization: (reachable)/(n-1) * (reachable / total)
                closeness[u] = (reachable / (n - 1)) * (reachable / total)
            else:
                closeness[u] = 0.0
        return closeness

    # -------------------------
    # Betweenness centrality (Brandes, weighted)
    # -------------------------
    def betweenness_centrality(self, normalized=True):
        # Brandes algorithm for weighted graphs
        nodes = list(self.nodes_set)
        CB = dict((v, 0.0) for v in nodes)
        for s in nodes:
            # Single-source shortest-paths
            S = []
            P = dict((v, []) for v in nodes)  # predecessors
            sigma = dict((v, 0.0) for v in nodes)  # number of shortest paths
            dist = dict((v, float('inf')) for v in nodes)
            sigma[s] = 1.0
            dist[s] = 0.0
            # priority queue for Dijkstra
            pq = [(0.0, s)]
            while pq:
                (d, v) = heapq.heappop(pq)
                if d > dist[v]:
                    continue
                S.append(v)
                for w, weight in self.adj[v].items():
                    vw_dist = dist[v] + weight
                    if dist[w] > vw_dist:
                        dist[w] = vw_dist
                        heapq.heappush(pq, (dist[w], w))
                        sigma[w] = sigma[v]
                        P[w] = [v]
                    elif dist[w] == vw_dist:
                        sigma[w] += sigma[v]
                        P[w].append(v)
            # accumulation
            delta = dict((v, 0.0) for v in nodes)
            while S:
                w = S.pop()
                for v in P[w]:
                    if sigma[w] != 0:
                        delta_v = (sigma[v] / sigma[w]) * (1.0 + delta[w])
                        delta[v] += delta_v
                if w != s:
                    CB[w] += delta[w]
        # normalization
        if normalized:
            n = len(nodes)
            if n <= 2:
                return CB
            scale = 1.0 / ((n - 1) * (n - 2))
            for v in CB:
                CB[v] *= scale
        return CB

    # -------------------------
    # Random graph generator
    # -------------------------
    @classmethod
    def random_graph(cls, n_nodes, n_edges, directed=False, connected=False, weight_range=(1, 10), seed=None, node_labels=None):
        if seed is not None:
            random.seed(seed)
        if node_labels is not None and len(node_labels) != n_nodes:
            raise ValueError("The number of node labels must match n_nodes.")
        g = cls(directed=directed)
        if node_labels is None:
            nodes = [str(i) for i in range(1, n_nodes + 1)]
        else:
            nodes = list(node_labels)
        for v in nodes:
            g.add_node(v)
        max_edges = n_nodes * (n_nodes - 1)
        if not directed:
            max_edges //= 2
        if n_edges > max_edges:
            raise ValueError("Too many edges for given node count.")
        if connected and n_edges < n_nodes - 1:
            raise ValueError("A connected graph requires at least n_nodes - 1 edges.")
        edges_added = set()
        def next_weight():
            low, high = weight_range
            if isinstance(low, int) and isinstance(high, int):
                return random.randint(low, high)
            return random.uniform(low, high)
        if connected:
            remaining = nodes[:]
            random.shuffle(remaining)
            connected_set = [remaining.pop()]
            while remaining:
                u = random.choice(connected_set)
                v = remaining.pop()
                w = next_weight()
                g.add_edge(u, v, w)
                key = (u, v) if directed else tuple(sorted((u, v)))
                edges_added.add(key)
                connected_set.append(v)
        attempts = 0
        while len(edges_added) < n_edges and attempts < n_edges * 10 + 1000:
            u = random.choice(nodes)
            v = random.choice(nodes)
            if u == v: 
                attempts += 1
                continue
            key = (u, v) if directed else tuple(sorted((u, v)))
            if key in edges_added:
                attempts += 1
                continue
            w = next_weight()
            g.add_edge(u, v, w)
            edges_added.add(key)
            attempts += 1
        if len(edges_added) < n_edges:
            raise RuntimeError("Unable to generate the requested number of edges. Try relaxing constraints.")
        return g
