import numpy as np
from clusterspecialists.algorithms import Algorithm


class Perceptron(Algorithm):

    def __init__(self, graph, max_radius, **kwargs):
        super().__init__(graph=graph)
        n = self.graph.n_nodes
        lap_pinv = np.linalg.pinv(self.graph.laplacian)  # pseudo-inverse of Laplacian
        self.r_g = np.max(np.diag(lap_pinv))  # resistance diameter (upper bound)
        self.scheme = "perceptron"
        self.w = np.zeros((n, 1))
        self._G_inverse = lap_pinv + (self.r_g * np.ones((n, n)))  # Kernel
        self.radius = 0
        self.max_radius = max_radius  # self.lab.get_optimal_max_radius()

    def predict(self, vi):
        self._trial_number += 1
        # vi_graph_index = self.graph.nodes().index(vi)
        vi_graph_index = vi
        prediction = self.w[vi_graph_index, 0]  # Take the (i)th element of the weight vector
        return prediction

    def update(self, vi, prediction, label):

        if np.sign(prediction) != label:  # Update & Project only when a mistake is made
            self.mistakes += 1
            # vi_graph_index = self.graph.nodes().index(vi)
            vi_graph_index = vi

            # Use adaptive learning rate
            x_t = self._G_inverse[:, vi_graph_index]
            norm_x_sq = self._G_inverse[vi_graph_index, vi_graph_index]
            self.w = self.w + ((x_t * label) / norm_x_sq)

            # # Projection step (Efficient calculation of ||w||_G)
            old_radius = self.radius
            self.radius = np.sqrt((old_radius**2) + (1/norm_x_sq) + (2*label*prediction/norm_x_sq))
            if self.radius > self.max_radius:
                self.w = (self.w / self.radius) * self.max_radius
                self.radius = self.max_radius
                print('PROJECTED')
