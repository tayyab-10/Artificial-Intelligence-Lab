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

    def dfs(self, start_vertex):
        visited = set()
        self._dfs_recursive(start_vertex, visited)
        return visited

    def _dfs_recursive(self, current_vertex, visited):
        visited.add(current_vertex)
        print(current_vertex, end=" ")

        for edge in self.adjacency_list[current_vertex]:
            if edge.destination not in visited:
                self._dfs_recursive(edge.destination, visited)


def dfs_draw_tree(graph, start_vertex):
    visited = set()  # To track visited vertices
    print("DFS Tree Structure:")

    def _dfs_draw(current_vertex, level):
        visited.add(current_vertex)
        print("  " * level + str(current_vertex))  # Indentation for tree structure
        
        # Recursively visit all neighborss
        for edge in graph.adjacency_list[current_vertex]:
            if edge.destination not in visited:
                _dfs_draw(edge.destination, level + 1) 

    _dfs_draw(start_vertex, 0)


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
        start_vertex = input("Enter the start vertex for DFS: ").strip()
        print("DFS Traversal Order:")
        graph.dfs(start_vertex)
        
        print("\nDFS Tree Structure:")
        dfs_draw_tree(graph, start_vertex)
