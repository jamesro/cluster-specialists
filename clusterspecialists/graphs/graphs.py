import networkx as nx
import numpy as np
from .specialistset import BinarySet, get_fs_set_for_nodes


def depth_first_search(tree, tree_nodes, node_index, visited):
    this_node = tree_nodes[node_index]

    if this_node not in visited:
        visited.append(this_node)
        available_node_indexes = np.nonzero(tree[node_index])[1]

        for node_index in available_node_indexes:
            depth_first_search(tree, tree_nodes, node_index, visited)
    return visited


class Graph:

    def __init__(self, data_adj_matrix):

        # Matrices
        self.adj_mat = data_adj_matrix
        degree_matrix = np.diagflat(self.adj_mat.sum(axis=1))

        self.laplacian = degree_matrix - data_adj_matrix
        lap_pinv = np.linalg.pinv(self.laplacian)  # pseudo-inverse of Laplacian

        self.r_g = np.max(np.diag(lap_pinv))  # resistance diameter (upper bound)
        self.n_nodes = self.adj_mat.shape[0]
        self._nodes = np.array([i for i in range(self.n_nodes)])

        # Build specialist sets (a dict of nodes, where each node contains an np.array
        # of specialist indices which 'cover' that node
        self.fs_set = get_fs_set_for_nodes(self.n_nodes)

        binaryset = BinarySet(self.n_nodes)
        self.bs_set = binaryset.get_spec_ids_for_nodes()

        # keep a record of how many specialists are in each set (for the algorithm)
        self.size_of_fs_set = int((self.n_nodes ** 2 + self.n_nodes) / 2)
        self.size_of_bs_set = binaryset.n_specialists

    def random_spanning_tree(self):
        """
        For some reason, the networkx library doesn't write it's
        matrices 'in order', i.e. the nth row doesn't correspond
        to the nth node, instead it corresponds to the nth element
        of graph.nodes() (which is usually not in order)
        """
        # Uses a random walk on the graph to choose
        visited = []  # Nodes visited
        t_edges = []  # list of tuples

        current_node_index = np.random.choice(range(self.n_nodes))
        current_node = self._nodes[current_node_index]

        visited.append(current_node)
        while len(visited) < self.n_nodes:
            next_node_index = np.random.choice(np.nonzero(self.adj_mat[current_node_index])[1])
            next_node = self._nodes[next_node_index]
            if next_node not in visited:
                visited.append(next_node)
                t_edges.append((current_node, next_node))
            current_node_index = next_node_index
            current_node = next_node

        tree = nx.Graph()  # Use the original Graph class from networkx, not this Graph class
        tree.add_edges_from(t_edges)
        return tree

    def make_spine(self, tree):
        tree_nodes = list(tree.nodes())
        starting_node_index = np.random.choice(range(self.n_nodes))

        tree_adj_mat = nx.to_numpy_matrix(tree)

        spine = depth_first_search(tree_adj_mat, tree_nodes, starting_node_index, visited=[])
        return spine
