from graph import Graph

def pajek_generator(num_nodes, num_edges, connected, *, directed=False, file_name=None, seed=None, weight_range=(1, 10), preview_lines=20, node_prefix="Node"):
    print("=== RANDOM GRAPH GENERATOR (PAJEK FORMAT) ===")
    try:
        labels = [f"{node_prefix} {index}" for index in range(1, num_nodes + 1)]
        graph = Graph.random_graph(
            num_nodes,
            num_edges,
            directed=directed,
            connected=connected,
            weight_range=weight_range,
            seed=seed,
            node_labels=labels,
        )
        if file_name is None:
            direction_tag = "dir" if directed else "undir"
            file_name = f"grafo_{num_nodes}n_{num_edges}e_{direction_tag}.net"
        graph.save_pajek(file_name)
        print(f"Graph generated and saved to: {file_name}")
        print("\nSummary:")
        print(f"- Nodes: {len(graph.nodes())}")
        print(f"- Edges: {len(list(graph.edges()))}")
        print(f"- Connected: {'Yes' if graph.is_connected() else 'No'}")
        print(f"- Eulerian: {'Yes' if graph.is_eulerian() else 'No'}")
        print(f"- Cyclic: {'Yes' if graph.is_cyclic() else 'No'}")
        print("\nFile preview:")
        with open(file_name, "r", encoding="utf8") as file:
            for index, line in enumerate(file):
                if index >= preview_lines:
                    break
                print(line.rstrip())
    except ValueError as error:
        print(f"Error: {error}")
    except Exception as error:
        print("Unexpected error:", error)

if __name__ == "__main__":
    pajek_generator(500, 2000, connected=False)
