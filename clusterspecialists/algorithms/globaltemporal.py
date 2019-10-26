from clusterspecialists.algorithms import Algorithm
from clusterspecialists.graphs.optimalparameters import get_universal_majority
import numpy as np


class GlobalTemporal(Algorithm):
    def __init__(self, graph, training_labelings, n_trials_per_segment, **kwargs):
        super().__init__(graph=graph)
        self.scheme = 'global-temporal'
        self.n_trials_per_segment = n_trials_per_segment
        self.training_labelings = training_labelings
        self.n_trained_segments = len(training_labelings)
        self.global_labels = {}
        self.get_global_slices()

    def get_global_slices(self):
        for segment, labeling in self.training_labelings.items():

            # Make a dictionary to pass to get_universal_majority (as it expects
            # a dictionary of different labelings)
            temp_label_dict = {0: labeling}
            self.global_labels[segment] = get_universal_majority(temp_label_dict)

    def predict(self, vi):
        self._trial_number += 1

        # Calculate the current segment, given the trial number and the number of trials
        # per segment.
        # Then 'wrap' that around so that we're within e.g. 0-24 segments
        segment = int(np.floor(self._trial_number / self.n_trials_per_segment)) % self.n_trained_segments
        return self.global_labels[segment]

    def update(self, vi, prediction, label):
        if prediction != label:
            self.mistakes += 1
