class Edge:
    def __init__(self, source, destination):
        self.source = source
        self.destination = destination
    def __repr__(self):
        return f"Edge({self.source}, {self.destination})"

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
            if not self.is_directed:  # undirected graph
                self.adjacency_list[destination].append(source)

    # Deleting an edge between vertices
    def delete_edge(self, source, destination):
        if source in self.adjacency_list and destination in self.adjacency_list[source]:
            self.adjacency_list[source].remove(destination)
        
        if not self.is_directed and destination in self.adjacency_list and source in self.adjacency_list[destination]:
            self.adjacency_list[destination].remove(source)
    def get_edge(self, node1, node2):
     
     if node1 in self.adjacency_list:
        if node2 in self.adjacency_list[node1]:
            return Edge(node1, node2)
     if not self.is_directed:
        if node2 in self.adjacency_list and node1 in self.adjacency_list[node2]:
            return Edge(node2, node1)
     return None

    # Neighbors of the vertex
    def get_neighbors(self, vertex):
        if vertex in self.adjacency_list:
            return self.adjacency_list[vertex]
        return []
    
    # Total number of vertices
    def num_vertices(self):
        return len(self.adjacency_list)
    
    # Total number of edges
    def num_edges(self):
        count = 0
        for vertex in self.adjacency_list:
            count += len(self.adjacency_list[vertex])
        return count if self.is_directed else count // 2
    
    # Check if the graph is directed
    def is_directed_graph(self):
        return self.is_directed
    
    # Method to print the graph
    def print_graph(self):
        for vertex in self.adjacency_list:
            print(f"{vertex} -> {', '.join(self.adjacency_list[vertex]) if self.adjacency_list[vertex] else 'None'}")

    
# Read graph from a file
def read_graph_from_file(filename):
    try:
        file = open(filename, 'r')
        
        # First line in the file contains the number of vertices and whether it is directed
        first_line = file.readline().strip().split('_')
        num_vertices = int(first_line[0])
        is_directed = bool(int(first_line[1]))

        # Second line contains the vertex names
        vertex_names = file.readline().strip().split()

        # Graph Creation
        graph = Graph(is_directed)

        # Add vertices to the graph
        for vertex in vertex_names:
            graph.add_vertex(vertex)

        # Third line contains the number of edges
        num_edges = int(file.readline().strip())

        # Next lines contain the edges
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

# Console interface for graph interaction
def console_ui(graph):
    if graph is None:
        return

    while True:
        print("\n----- Graph Information Menu ------")
        print("1- Number of Vertices")
        print("2- Number of Edges")
        print("3- Is the Graph Directed?")
        print("4- Find Neighbors of a Vertex")
        print("5- Delete Edge")
        print("6- Print Graph")
        print("7- Get Edge of the graph")
        print("7- Exit")
        
        choice = input("Enter your choice between (1-7): ").strip()
        
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
            print("Enter the source of the edge you want to delete:")
            source = input().strip()

            print("Enter the destination of the edge you want to delete:")
            destination = input().strip()
            print("The Graph before deleting the Edge")
            graph.print_graph()
            graph.delete_edge(source, destination)
            print(f"Edge between {source} and {destination} deleted.")
            print("The Graph after deleting the Edge")
            graph.print_graph()

        elif choice == '6':
            graph.print_graph()

        elif choice == '7':
            print("Enter the First Node:")
            node1 = input().strip()

            print("Enter the second Node: ")
            node2 = input().strip()
            result=graph.get_edge(node1,node2)
            if result:
                 print(f"The edge is: {result}")
            else:
                 print("No edge exists between the given nodes.")
        
        elif choice == '8':
            print("Exiting...")
            break
        
        else:
            print("Invalid choice.")

if __name__ == "__main__":
    filename = input("Enter the name of the graph file: ").strip()
    graph = read_graph_from_file(filename)
    console_ui(graph)
