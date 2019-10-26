from clusterspecialists.algorithms import Algorithm


class NaiveHomogeneous(Algorithm):
    def __init__(self, graph, universal_majority_label, **kwargs):
        super().__init__(graph=graph)
        self.scheme = 'naive-homogeneous'

        self.universal_majority = universal_majority_label

    def predict(self, vi):
        self._trial_number += 1
        return self.universal_majority

    def update(self, vi, prediction, label):
        if prediction != label:
            self.mistakes += 1

