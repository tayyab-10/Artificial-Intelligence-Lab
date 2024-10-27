from collections import deque
import heapq

# Pair class for PQ
class Pair:
    def __init__(self, node, dist):
        self.node = node
        self.dist = dist

    # we are sorting the pair based on its distance to it's neigbors
    def __lt__(self, other):
        return self.dist < other.dist  # Compare by distance 

# Edge class to represent edges with source, destination, and weight
class Edge:
    def __init__(self, source, destination, weight):
        self.source = source
        self.destination = destination
        self.weight = weight

# Graph class for managing the graph's structure
class Graph:
    def __init__(self, is_directed):
        self.is_directed = is_directed
        self.adjacency_list = {}

    def add_vertex(self, vertex):
        if vertex not in self.adjacency_list:
            self.adjacency_list[vertex] = []

    def add_edge(self, source, destination, weight):
        if source not in self.adjacency_list:
            self.add_vertex(source)
        if destination not in self.adjacency_list:
            self.add_vertex(destination)

        # Add weighted edge from source to destination
        self.adjacency_list[source].append((destination, weight))
        if not self.is_directed:
            # If undirected, add the reverse edge
            self.adjacency_list[destination].append((source, weight))

# Function to read graph from file
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
                weight = int(edge_line[2])  # The weight of the edge
                graph.add_edge(source, destination,weight)

            return graph
    except FileNotFoundError:
        print(f"Error: File '{filename}' not found.")
        return None
    except Exception as e:
        print(f"Error while reading the file: {e}")
        return None

# Dijkstra's algorithm for shortest path
def dijkstra(graph, start_vertex, end_vertex):
    pq = []
    heapq.heappush(pq, Pair(start_vertex, 0))  # Initialize PQ 
    
    # WE created Distance dictionary to track the shortest distance and also we are setting the distance from first vertex to all vertices as infinity
    dist = {vertex: float('inf') for vertex in graph.adjacency_list}
    dist[start_vertex] = 0

    while pq:
        current_pair = heapq.heappop(pq)  # get the node with the minimum distance
        u = current_pair.node
        wt = current_pair.dist

        # If it is end vertex than return its weight
        if u == end_vertex:
            return wt

        # Explore the neighbors of the current node
        for neighbor, weight in graph.adjacency_list[u]:
            new_dist = wt + weight

            # If a shorter distance is found we update the distance of that node
            if new_dist < dist[neighbor]:
                dist[neighbor] = new_dist
                heapq.heappush(pq, Pair(neighbor, new_dist))

    # If we did not reach the end node we just return infinity
    return float('inf')

if __name__ == "__main__":
    filename = input("Enter the name of the graph file: ").strip()
    graph = read_graph_from_file(filename)

    if graph:
        start_vertex = input("Enter the start vertex to calculate the shortest distance: ").strip()
        end_vertex = input("Enter the end vertex for the shortest path: ").strip()
        shortest_distance = dijkstra(graph, start_vertex, end_vertex)
        #If the path does not exist we just print that the path does not exist for these vertices
        if shortest_distance == float('inf'):
            print(f"No path exists from {start_vertex} to {end_vertex}.")
        else:
            print(f"The shortest distance from {start_vertex} to {end_vertex} is {shortest_distance}.")
