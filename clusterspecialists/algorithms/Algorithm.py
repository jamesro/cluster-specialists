class Algorithm(object):
    def __init__(self, graph):
        self.mistakes = 0
        self._trial_number = 0
        self.graph = graph

    def predict(self, vi):
        pass

    def update(self, vi, prediction, label):
        pass