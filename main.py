import math
import random
import matplotlib.pyplot as plt
import numpy as np
import networkx as nx


def check_for_success(prob):
    if prob == 0:
        return False
    boundary = prob * pow(10, len(str(prob)))
    rnd = random.randrange(0, pow(10, len(str(prob))))
    if rnd <= boundary:
        return True
    else:
        return False


class Graph:
    connections = []
    diameter = 0
    radius = 0
    cc = 0  # clustering coefficient

    def __init__(self, nodes_amount):
        self.nodes_amount = nodes_amount
        self.nei_amount = np.zeros(nodes_amount)
        self.adjacency_matrix = np.zeros((nodes_amount, nodes_amount))

    def bfs(self, chosen_node):
        levels = [[chosen_node]]
        prev_node = np.full(self.nodes_amount, -1)
        visited = []
        for idx in range(self.nodes_amount):
            visited.append(self.nei_amount[idx] == 0)
        visited[chosen_node] = True
        place_holder = []
        stop_cond = True
        while stop_cond:
            for node in levels[-1]:
                for conn in self.connections:
                    if conn[0] == node and not visited[conn[1]]:
                        place_holder.append(conn[1])
                        prev_node[conn[1]] = node
                        visited[conn[1]] = True
                    elif conn[1] == node and not visited[conn[0]]:
                        place_holder.append(conn[0])
                        prev_node[conn[0]] = node
                        visited[conn[0]] = True
            if not place_holder:
                stop_cond = False
            else:
                levels.append(place_holder.copy())
            place_holder.clear()
        return_list = [levels, prev_node]
        return return_list

    def find_distance(self, end_node, bfs_result):
        levels, prev_nodes = bfs_result
        distance = 0
        for lvl in enumerate(levels[1:len(levels)]):
            distance += 1
            if end_node in lvl[1]:
                break
        return distance

    def find_eccentricity(self):
        max_eccentricity = 0
        min_eccentricity = 0
        for idx1 in range(self.nodes_amount):
            bfs_result = self.bfs(idx1)
            current_path_len = 0
            for idx2 in range(self.nodes_amount):
                if not idx1 == idx2:
                    prev_path_len = self.find_distance(idx2, bfs_result)
                    if prev_path_len > current_path_len:
                        current_path_len = prev_path_len
            if current_path_len > max_eccentricity:
                max_eccentricity = current_path_len
            if current_path_len < min_eccentricity or min_eccentricity == 0:
                min_eccentricity = current_path_len
        return [min_eccentricity, max_eccentricity]

    def find_diameter(self):
        self.diameter = self.find_eccentricity()[1]
        return self.diameter

    def find_radius(self):
        self.radius = self.find_eccentricity()[0]
        return self.radius

    def find_cc(self):
        denominator = math.factorial(self.nodes_amount) / \
                    (math.factorial(2) * math.factorial(self.nodes_amount - 2))
        numerator = len(self.connections)
        return numerator / denominator


class Erdos_Renyi(Graph):

    def __init__(self, nodes_amount, probability=0.5):
        Graph.__init__(self, nodes_amount=nodes_amount)
        self.prob = probability

    def create(self):
        self.connections.clear()
        for idx1 in range(self.nodes_amount):
            for idx2 in range(self.nodes_amount):
                if idx1 == idx2:
                    continue
                if check_for_success(self.prob):
                    if (idx2, idx1) in self.connections:
                        continue
                    self.connections.append((idx1, idx2))
                    self.adjacency_matrix[idx1][idx2] = 1
                    self.adjacency_matrix[idx2][idx1] = 1
                    self.nei_amount[idx1] += 1
                    self.nei_amount[idx2] += 1


class Watts_Strogatz(Graph):

    def __init__(self, nodes_amount, nei_number=2, probability=0.5):
        Graph.__init__(self, nodes_amount)
        if not nei_number % 2 == 0:
            print("Neighbours number not even. "
                  "Changing to closest possible even number")
            if nei_number + 1 == nodes_amount:
                nei_number -= 1
            else:
                nei_number += 1
        self.nei_number = nei_number
        self.prob = probability

    def _initialize_graph(self):
        for idx1 in range(self.nodes_amount):
            for idx2 in range(1, int(self.nei_number / 2) + 1):
                if idx1 == (idx1 + idx2) % self.nodes_amount:
                    continue
                self.connections.append((idx1, (idx1 + idx2) % self.nodes_amount))
                self.adjacency_matrix[idx1][(idx1 + idx2) % self.nodes_amount] = 1
                self.adjacency_matrix[(idx1 + idx2) % self.nodes_amount][idx1] = 1
                self.nei_amount[idx1] += 1
                self.nei_amount[(idx1 + idx2) % self.nodes_amount] += 1

    def _rewire(self):
        for idx1 in range(len(self.connections)):
            if check_for_success(self.prob):
                rand_node = random.randrange(self.nodes_amount)
                while self.connections[idx1][0] == rand_node:
                    rand_node = random.randrange(self.nodes_amount)
                if not ((self.connections[idx1][0], rand_node) in self.connections or
                        (rand_node, self.connections[idx1][0]) in self.connections):
                    self.nei_amount[self.connections[idx1][1]] -= 1
                    self.nei_amount[rand_node] += 1
                    self.connections[idx1] = (self.connections[idx1][0], rand_node)
                    self.adjacency_matrix[self.connections[idx1][0]][rand_node] = 1
                    self.adjacency_matrix[rand_node][self.connections[idx1][0]] = 1

    def create(self):
        self.connections.clear()
        self._initialize_graph()
        self._rewire()
        self.connections.sort()


class Barabasi_Albert(Graph):

    def __init__(self, init_nodes_amount, max_nodes_numb, init_prob):
        Graph.__init__(self, init_nodes_amount)
        self.max_nodes = max_nodes_numb
        self.init_graph = Erdos_Renyi(init_nodes_amount, init_prob)
        self.init_graph.create()
        self.adjacency_matrix = np.zeros((max_nodes_numb, max_nodes_numb))
        for idx1 in range(init_nodes_amount):
            for idx2 in range(init_nodes_amount):
                self.adjacency_matrix[idx1][idx2] = self.init_graph.adjacency_matrix[idx1][idx2]
        self.connections = self.init_graph.connections
        self.nei_amount = self.init_graph.nei_amount.copy()
        self.nei_amount.resize(max_nodes_numb)

    def create(self):
        for idx1 in range(self.nodes_amount, self.max_nodes):
            for idx2 in range(self.nodes_amount):
                if check_for_success(self.nei_amount[idx2] / self.nodes_amount):
                    self.connections.append((idx2, idx1))
                    self.adjacency_matrix[idx1][idx2] = 1
                    self.adjacency_matrix[idx2][idx1] = 1
                    self.nei_amount[idx1] += 1
                    self.nei_amount[idx2] += 1
            self.nodes_amount += 1


graph = Watts_Strogatz(1000, 4, 0.5)
graph.create()

plt.hist(graph.nei_amount, bins=50)
plt.xlabel("Neighbours amount")
plt.ylabel("Vertices amount")
plt.show()

print("Radius: ", graph.find_radius())
print("Diameter: ", graph.find_diameter())
print("Clustering coefficient: ", graph.find_cc())
