import numpy as np
from clusterspecialists.algorithms import Algorithm


class Specialists(Algorithm):

    def __init__(self, graph, scheme, alpha=None,  **kwargs):
        super().__init__(graph=graph)

        self.scheme = scheme
        self._tree = self.graph.random_spanning_tree()
        self.spine = self.graph.make_spine(self._tree)
        self.n_nodes = len(self.spine)

        if alpha is not None:
            self.alpha = alpha
        else:
            self.alpha = None  # this will break if we don't tune alpha and pass it when instantiating
            # self.alpha = get_optimal_specialists_alpha(self.spine, self.scheme, labelings, n_trials_per_segment)

        # for conservative method
        self.w_pos_temp = None
        self.w_neg_temp = None

        if scheme == "fully-covering":
            self._n = self.graph.size_of_fs_set  # Number of 'base' specialists(double for total number)
            self.time_cache = np.ones(self._n)
        elif scheme == "binary-tree":
            self._n = self.graph.size_of_bs_set
            self.time_cache = np.ones(self._n)
        else:
            raise ValueError("\n\nInvalid Specialist Scheme chosen. Choose either:\t'fully-covering'\tor\t'binary-tree'")

        self.w_pos = np.ones(self._n) / (2 * self._n)
        self.w_neg = np.ones(self._n) / (2 * self._n)

    def predict(self, vi):
        self._trial_number += 1

        # Catch up specialists who were sleeping
        self.share_update(vi)

        if self.scheme == 'binary-tree':
            active_set = self.graph.bs_set[self.spine.index(vi)]
        else:
            active_set = self.graph.fs_set[self.spine.index(vi)]

        positive_weight = np.sum(self.w_pos[active_set])
        negative_weight = np.sum(self.w_neg[active_set])

        if positive_weight > negative_weight:
            return 1
        elif positive_weight < negative_weight:
            return -1
        else:
            return np.random.choice([-1, 1])

    def loss_update(self, vi, label):
        if self.scheme == 'binary-tree':
            active_set = self.graph.bs_set[self.spine.index(vi)]
        else:
            active_set = self.graph.fs_set[self.spine.index(vi)]

        positive_specialists = self.w_pos[active_set]
        negative_specialists = self.w_neg[active_set]

        positive_weight = np.sum(positive_specialists)
        negative_weight = np.sum(negative_specialists)
        total_weight_before = positive_weight + negative_weight

        if label == 1:
            # Punish negative predictors
            negative_specialists = negative_specialists * 0
            total_weight_after = positive_weight
        else:
            # Punish positive predictors
            positive_specialists = positive_specialists * 0
            total_weight_after = negative_weight

        renormalization_factor = total_weight_before / total_weight_after

        self.w_pos[active_set] = positive_specialists * renormalization_factor
        self.w_neg[active_set] = negative_specialists * renormalization_factor

    def share_update(self, vi):
        # Conservative method - store initial weights and bring them back
        # if no mistake is made
        self.w_pos_temp = self.w_pos.copy()
        self.w_neg_temp = self.w_neg.copy()

        if self.scheme == 'fully-covering':
            active_set = self.graph.fs_set[self.spine.index(vi)]
        else:
            active_set = self.graph.bs_set[self.spine.index(vi)]

        s_i = self.time_cache[active_set]
        t = self._trial_number
        self.w_neg[active_set] = ((1-self.alpha)**(t - s_i))*self.w_neg[active_set] + \
                                 ((1-(1-self.alpha)**(t-s_i))/(2*self._n))
        self.w_pos[active_set] = ((1-self.alpha)**(t - s_i))*self.w_pos[active_set] + \
                                 ((1-(1-self.alpha)**(t-s_i))/(2*self._n))

    def update(self, vi, prediction, label):  # CONSERVATIVE
        if prediction != label:
            self.mistakes += 1
            if self.scheme == 'fully-covering':
                active_set = self.graph.fs_set[self.spine.index(vi)]
            else:
                active_set = self.graph.bs_set[self.spine.index(vi)]
            # Cache update
            self.time_cache[active_set] = self._trial_number

            # Loss update is the same for both fcs and bts
            self.loss_update(vi, label)
        else:
            self._trial_number -= 1
            self.w_neg = self.w_neg_temp.copy()
            self.w_pos = self.w_pos_temp.copy()  # Undo the share update that we did, don't change time cache
