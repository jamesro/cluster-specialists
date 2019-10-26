from clusterspecialists.algorithms import Algorithm


class NaiveHeterogeneous(Algorithm):
    def __init__(self, graph, node_majority_labels, **kwargs):
        super().__init__(graph=graph)
        self.scheme = 'naive-heterogeneous'
        self.node_majority_labels = node_majority_labels

    def predict(self, vi):
        self._trial_number += 1
        return self.node_majority_labels[vi]

    def update(self, vi, prediction, label):
        if prediction != label:
            self.mistakes += 1
