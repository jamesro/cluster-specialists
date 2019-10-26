import numpy as np
from clusterspecialists.algorithms import Algorithm


class NotSoNaive(Algorithm):
    def __init__(self, graph, training_labelings, n_trials_per_segment, **kwargs):
        super().__init__(graph=graph)
        self.scheme = 'not-so-naive'
        self.training_labelings = training_labelings
        self.n_trials_per_segment = n_trials_per_segment
        self.n_trained_segments = len(training_labelings)

    def predict(self, vi):
        self._trial_number += 1

        # Calculate the current segment, given the trial number and the number of trials
        # per segment.
        # Then 'wrap' that around so that we're within e.g. 0-24 segments
        segment = int(np.floor(self._trial_number / self.n_trials_per_segment)) % self.n_trained_segments
        return self.training_labelings[segment][vi]

    def update(self, vi, prediction, label):
        if prediction != label:
            self.mistakes += 1
