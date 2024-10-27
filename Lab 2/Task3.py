from collections import deque

class Edge:
    def __init__(self, source, destination):
        self.source = source
        self.destination = destination

class Graph:
    def __init__(self, is_directed):
        self.is_directed = is_directed
        self.adjacency_list = {}

    def add_vertex(self, vertex):
        if vertex not in self.adjacency_list:
            self.adjacency_list[vertex] = []

    def add_edge(self, source, destination):
        if source not in self.adjacency_list:
            self.add_vertex(source)
        if destination not in self.adjacency_list:
            self.add_vertex(destination)

        self.adjacency_list[source].append(Edge(source, destination))
        if not self.is_directed:
            self.adjacency_list[destination].append(Edge(destination, source))

    def bfs(self, start_vertex):
        visited = set()
        distance = {}
        queue = deque([start_vertex])
        visited.add(start_vertex)
        distance[start_vertex] = 0   #To Keep track of distance of the node 
        levels = {start_vertex: 0}   # To keep track of the level of the nodes 

        while queue:
            current_vertex = queue.popleft()
            print(f"Visited Vertex: {current_vertex}")

            for edge in self.adjacency_list[current_vertex]:
                neighbor = edge.destination
                if neighbor not in visited:
                    queue.append(neighbor)
                    visited.add(neighbor)
                    distance[neighbor] = distance[current_vertex] + 1
                    levels[neighbor] = distance[neighbor]

        # Calculate number of levels
        num_levels = max(levels.values(), default=0) + 1

        # Output 
        print(f"Distances from {start_vertex}: {distance}")
        print(f"Number of levels in the BFS tree: {num_levels}")

        #BFS tree
        self.draw_bfs_tree(start_vertex, levels)

        return distance
    
    def draw_bfs_tree(self, start_vertex, levels):
        visited = set()
        print("BFS Tree Structure:")

        def _draw_tree(current_vertex, level):
            visited.add(current_vertex)
            print("  " * level + str(current_vertex))  # Indentation for tree structure
            
            # Recursively visit all neighbors (children in the tree)
            for edge in self.adjacency_list[current_vertex]:
                if edge.destination not in visited:
                    _draw_tree(edge.destination, level + 1)  # Go one level deeper

        _draw_tree(start_vertex, 0)

def read_graph_from_file(filename):
    try:
            file = open(filename, 'r')
            # First line in the file contains the number of vertices that graph has and whether it is directed or not

            first_line = file.readline().strip().split('_')
            num_vertices = int(first_line[0])
            is_directed = bool(int(first_line[1]))

            # Second line containes tha vertex names
            vertex_names = file.readline().strip().split()

            # Graph Creation
            graph = Graph(is_directed)

            # Add vertices to the graph
            for vertex in vertex_names:
                graph.add_vertex(vertex)

            # Third line contains the  number of edges
            num_edges = int(file.readline().strip())

            # Next lines contains the edges 
            for _ in range(num_edges):
                edge_line = file.readline().strip().split()
                source = edge_line[0]
                destination = edge_line[1]
                graph.add_edge(source, destination)

            return graph
    except FileNotFoundError:
        print(f"Error: File '{filename}' not found.")
        return None
    except Exception as e:
        print(f"Error while reading the file: {e}")
        return None

if __name__ == "__main__":
    filename = input("Enter the name of the graph file: ").strip()
    graph = read_graph_from_file(filename)

    if graph:
        start_vertex = input("Enter the start vertex for BFS: ").strip()
        print("BFS Traversal Order:")
        graph.bfs(start_vertex)
