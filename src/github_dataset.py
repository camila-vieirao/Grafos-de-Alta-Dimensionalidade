import csv
import json
from graph import Graph

def load_github_dataset(edges_file, target_file, features_file=None, max_nodes=5000, max_edges=20000):
    print("=== LOADING GITHUB DATASET ===")
    print(f"Target: exactly {max_nodes} nodes, exactly {max_edges} edges\n")
    
    all_node_info = {}
    print(f"Loading all node metadata from {target_file}...")
    with open(target_file, 'r', encoding='utf8') as f:
        reader = csv.DictReader(f)
        for row in reader:
            node_id = row['id']
            name = row.get('name', node_id)
            ml_target = row.get('ml_target', '0')
            all_node_info[node_id] = {
                'name': name,
                'label': ml_target
            }
    
    print(f"Loaded metadata for {len(all_node_info)} nodes")
    
    print(f"\nLoading edges from {edges_file} (will load extra to ensure {max_edges} valid edges)...")
    all_edges = []
    nodes_in_edges = set()
    
    with open(edges_file, 'r', encoding='utf8') as f:
        reader = csv.DictReader(f)
        for row in reader:
            id_1 = row['id_1']
            id_2 = row['id_2']
            
            if id_1 not in all_node_info:
                all_node_info[id_1] = {'name': f"User_{id_1}", 'label': '0'}
            if id_2 not in all_node_info:
                all_node_info[id_2] = {'name': f"User_{id_2}", 'label': '0'}
            
            all_edges.append((id_1, id_2))
            nodes_in_edges.add(id_1)
            nodes_in_edges.add(id_2)
    
    print(f"Loaded {len(all_edges)} total edges involving {len(nodes_in_edges)} unique nodes")
    
    print(f"Selecting {max_nodes} most connected nodes to maximize edge coverage...")
    from collections import Counter
    node_frequency = Counter()
    for id_1, id_2 in all_edges:
        node_frequency[id_1] += 1
        node_frequency[id_2] += 1
    
    most_connected = [node_id for node_id, _ in node_frequency.most_common(max_nodes)]
    selected_nodes = set(most_connected)
    
    if len(selected_nodes) < max_nodes:
        print(f"Need {max_nodes - len(selected_nodes)} more nodes to reach target...")
        for node_id in all_node_info.keys():
            if len(selected_nodes) >= max_nodes:
                break
            if node_id not in selected_nodes:
                selected_nodes.add(node_id)
    
    print(f"Selected {len(selected_nodes)} nodes")
    
    print("Filtering edges to keep only those between selected nodes...")
    valid_edges = []
    for id_1, id_2 in all_edges:
        if id_1 in selected_nodes and id_2 in selected_nodes:
            valid_edges.append((id_1, id_2))
            if len(valid_edges) >= max_edges:
                break
    
    print(f"Kept {len(valid_edges)} valid edges (target: {max_edges})")
    
    if features_file:
        print(f"\nLoading features from {features_file}...")
        try:
            with open(features_file, 'r', encoding='utf8') as f:
                features = json.load(f)
            for node_id, feature_list in features.items():
                if node_id in all_node_info:
                    all_node_info[node_id]['features'] = feature_list
            print("Loaded features")
        except Exception as e:
            print(f"Warning: Could not load features: {e}")
    
    print(f"\nBuilding graph with {len(selected_nodes)} nodes and {len(valid_edges)} edges...")
    graph = Graph(directed=True)
    
    for node_id in selected_nodes:
        info = all_node_info[node_id]
        label = f"{info['name']}_{info['label']}"
        graph.add_node(label)
    
    for id_1, id_2 in valid_edges:
        info_1 = all_node_info[id_1]
        info_2 = all_node_info[id_2]
        label_1 = f"{info_1['name']}_{info_1['label']}"
        label_2 = f"{info_2['name']}_{info_2['label']}"
        graph.add_edge(label_1, label_2, weight=1)
    
    print("\nFinal graph summary:")
    print(f"- Total nodes: {len(graph.nodes())}")
    print(f"- Total edges: {len(list(graph.edges()))}")
    
    return graph, all_node_info

def analyze_github_graph(graph, output_pajek="github_graph.net"):
    print("\n=== ANALYZING GITHUB GRAPH ===")
    
    print(f"\nSaving to Pajek format: {output_pajek}")
    graph.save_pajek(output_pajek)
    
    print("\n--- Basic Properties ---")
    print(f"Nodes: {len(graph.nodes())}")
    print(f"Edges: {len(list(graph.edges()))}")
    print(f"Directed: {graph.directed}")
    
    print("\n--- Connectivity Analysis ---")
    print("Checking if graph is connected...")
    is_connected = graph.is_connected()
    print(f"Connected (weakly): {'Yes' if is_connected else 'No'}")
    
    print("Finding connected components...")
    components = graph.components()
    print(f"Number of components: {len(components)}")
    if len(components) > 1:
        print("top 10 components with the most nodes:")
        for comp in sorted(components, key=len, reverse=True)[:10]:
           print(f"  {len(comp)} nodes: {list(comp)[:5]}")
    
    print("\n--- Graph Properties ---")
    print("Checking if graph is Eulerian...")
    is_eulerian = graph.is_eulerian()
    print(f"Eulerian: {'Yes' if is_eulerian else 'No'}")
    
    print("Checking if graph is cyclic...")
    is_cyclic = graph.is_cyclic()
    print(f"Cyclic: {'Yes' if is_cyclic else 'No'}")
    
    print("\n--- Centrality Analysis ---")
    print("Computing closeness centrality...")
    closeness = graph.closeness_centrality()
    print("Closeness centrality computed!")
    
    top_closeness = sorted(closeness.items(), key=lambda x: x[1], reverse=True)[:10]
    print("\nTop 10 nodes by closeness centrality:")
    for node, score in top_closeness:
        print(f"  {node}: {score:.6f}")
    
    print("\nComputing betweenness centrality...")
    betweenness = graph.betweenness_centrality()
    print("Betweenness centrality computed!")
    
    top_betweenness = sorted(betweenness.items(), key=lambda x: x[1], reverse=True)[:10]
    print("\nTop 10 nodes by betweenness centrality:")
    for node, score in top_betweenness:
        print(f"  {node}: {score:.6f}")
    
    print("\n=== ANALYSIS COMPLETE ===")
    print(f"Graph saved to: {output_pajek}")
    
    return {
        'connected': is_connected,
        'components': len(components),
        'eulerian': is_eulerian,
        'cyclic': is_cyclic,
        'closeness': closeness,
        'betweenness': betweenness
    }

if __name__ == "__main__":
    graph, node_info = load_github_dataset(
        edges_file="musae_git_edges.csv",
        target_file="musae_git_target.csv",
        features_file="musae_git_features.json"
    )
    
    results = analyze_github_graph(graph, output_pajek="github_graph.net")

