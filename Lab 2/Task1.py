class Edge:
    def __init__(self, source, destination):
        self.source = source
        self.destination = destination

class Graph:
    def __init__(self, is_directed):
        self.is_directed = is_directed
        self.adjacency_list = {}
    
    # Adding vertex 
    def add_vertex(self, vertex):
        if vertex not in self.adjacency_list:
            self.adjacency_list[vertex] = []
    
    # Creating edges between vertices
    def add_edge(self, source, destination):
        if source in self.adjacency_list and destination in self.adjacency_list:
            self.adjacency_list[source].append(destination)
            if not self.is_directed:  #undirected 
                self.adjacency_list[destination].append(source)

    # Neighbors of the vertex
    def get_neighbors(self, vertex):
        if vertex in self.adjacency_list:
            return self.adjacency_list[vertex]
        return []
    
     # Total number of vertices
    def num_vertices(self):
        return len(self.adjacency_list)
    
    #Total Number of Edges
    def num_edges(self):
        count = 0
        for vertex in self.adjacency_list:
            count += len(self.adjacency_list[vertex])
        return count if self.is_directed else count // 2
    
    def is_directed_graph(self):
        return self.is_directed
    
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

def console_ui(graph):
    if graph is None:
        return

    while True:
        print("\n----- Graph Information Menu ------")
        print("1- Number of Vertices")
        print("2- Number of Edges")
        print("3- Is the Graph Directed?")
        print("4- Find Neighbors of a Vertex")
        print("5- Exit")
        
        choice = input("Enter your choice between (1-5): ").strip()
        
        if choice == '1':
            print(f"The graph has {graph.num_vertices()} vertices.")
        
        elif choice == '2':
            print(f"The graph has {graph.num_edges()} edges.")
        
        elif choice == '3':
            if graph.is_directed_graph():
                print("The graph is Directed.")
            else:
                print("The graph is Undirected.")
        
        elif choice == '4':
            vertex = input("Enter the vertex name: ").strip()
            neighbors = graph.get_neighbors(vertex)
            if neighbors:
                print(f"Neighbors of {vertex}: {', '.join(neighbors)}")
            else:
                print(f"Vertex {vertex} does not exist or has no neighbors.")
        
        elif choice == '5':
            print("Exiting...")
            break
        
        else:
            print("Invalid choice.")

if __name__ == "__main__":
    filename = input("Enter the name of the graph file: ").strip()
    graph = read_graph_from_file(filename)
    console_ui(graph)
