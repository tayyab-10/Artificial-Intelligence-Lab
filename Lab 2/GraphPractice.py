from collections import deque

# Class to represent an Edge in a graph
class Edge:
    def __init__(self, source, destination, weight):
        self.source = source
        self.destination = destination
        self.weight = weight

# Class to represent a Graph
class Graph:
    def __init__(self, num_vertices):
        # Initialize graph with empty adjacency list for each vertex
        self.num_vertices = num_vertices
        self.adjacency_list = [[] for _ in range(num_vertices)]

    # Function to add an edge to the graph
    def add_edge(self, source, destination, weight):
        edge = Edge(source, destination, weight)
        self.adjacency_list[source].append(edge)

    # Function to print the graph (for visualization)
    def print_graph(self):
        for i in range(self.num_vertices):
            print(f"Vertex {i}: ", end="")
            for edge in self.adjacency_list[i]:
                print(f" -> (Dest: {edge.destination}, Weight: {edge.weight})", end="")
            print()

    # Breadth-First Search function
    def BFS(self, start_vertex):
        visited = [False] * self.num_vertices  # Boolean array to track visited vertices
        queue = deque([start_vertex])          # Initialize a queue and add the start vertex
        visited[start_vertex] = True           # Mark the start vertex as visited

        while queue:
            # Dequeue a vertex and process it
            current_vertex = queue.popleft()
            print(f"Visited {current_vertex}")

            # Get all the adjacent vertices of the dequeued vertex
            for edge in self.adjacency_list[current_vertex]:
                neighbor = edge.destination
                if not visited[neighbor]:
                    queue.append(neighbor)      # Enqueue the unvisited adjacent vertex
                    visited[neighbor] = True    # Mark it as visited

# Driver code to test the BFS function
if __name__ == "__main__":
    num_vertices = 5  # Example: graph with 5 vertices
    graph = Graph(num_vertices)

    # Adding edges (unidirectional)
    graph.add_edge(0, 1, 4)
    graph.add_edge(0, 2, 3)
    graph.add_edge(1, 2, 2)
    graph.add_edge(1, 3, 5)
    graph.add_edge(2, 3, 7)
    graph.add_edge(3, 4, 1)

    # Print the graph (optional for visualization)
    graph.print_graph()

    # Perform BFS starting from vertex 0
    print("\nBFS starting from vertex 0:")
    graph.BFS(0)
