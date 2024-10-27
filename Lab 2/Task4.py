from collections import deque

class Edge:
    def __init__(self, source, destination):
        self.source = source
        self.destination = destination

class Graph:
    def __init__(self, is_directed):
        self.is_directed = is_directed
        self.adjacency_list = {}
        self.cycles = []  # To store all detected cycles

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

    def is_cycle_directed(self, curr, visited, rec_stack, path):
        visited[curr] = True
        rec_stack[curr] = True
        path.append(curr)  # add current node to path

        # Visit neighbbors
        for edge in self.adjacency_list[curr]:
            neighbor = edge.destination
            if rec_stack[neighbor]:  # Cycle found
                cycle_path = path[path.index(neighbor):] + [neighbor]  # Get cycle path
                self.cycles.append(cycle_path)  #appent to the cycle path
                return True
            if not visited[neighbor]:
                if self.is_cycle_directed(neighbor, visited, rec_stack, path):
                    return True

        # Remove current vertex from rec stack and path
        rec_stack[curr] = False
        path.pop()
        return False

    def detect_cycle(self):
        visited = {vertex: False for vertex in self.adjacency_list}  
        rec_stack = {vertex: False for vertex in self.adjacency_list}  
        path = []  # Path to store the current 

        # To handle disconnected components
        for vertex in self.adjacency_list:
            if not visited[vertex]:
                if self.is_cycle_directed(vertex, visited, rec_stack, path):
                    continue  # Cycle detected but it would continue to check other components

        return len(self.cycles) > 0
    
    def is_cycle_undirected(self, visited, curr, parent, path):
        visited[curr] = True
        path.append(curr)  # We would add current node to the path

        for edge in self.adjacency_list[curr]:
            neighbor = edge.destination
            if visited[neighbor] and neighbor != parent:  # Cycle found
                cycle_path = path[path.index(neighbor):] + [neighbor]  # get cycle path
                self.cycles.append(cycle_path)  #appent to the cycle path
                return True
            elif not visited[neighbor]:
                if self.is_cycle_undirected(visited, neighbor, curr, path):
                    return True

        path.pop()  # Remove current node from path becaue of backtracking
        return False

    def detect_cycle_undirected(self):
        visited = {vertex: False for vertex in self.adjacency_list}
        path = [] 

        # To handle disconnected components
        for vertex in self.adjacency_list:
            if not visited[vertex]:
                if self.is_cycle_undirected(visited, vertex, None, path):
                    continue  # Cycle detected but it would continue to check other components

        return len(self.cycles) > 0

    def print_cycles(self):
        if self.cycles:
            print(f"Number of cycles FOund: {len(self.cycles)}")
            i = 1
            for cycle in self.cycles:
                print(f"Cycle {i}: {' -> '.join(cycle)}")
                i += 1
        else:
            print("No cycles detected.")

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
    filename = input("Enter the name of the graph file that is present in the same directory: ").strip()
    graph = read_graph_from_file(filename)
    
    if graph:
        if graph.is_directed:
            if graph.detect_cycle():
                graph.print_cycles()
            else:
                print("No cycle detected in the Directed Graph.")
        else:
            if graph.detect_cycle_undirected():
                graph.print_cycles()
            else:
                print("No cycle detected in the Directed Graph.")
