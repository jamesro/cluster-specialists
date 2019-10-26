import numpy as np


class BinarySet:
    def __init__(self, n_nodes):
        self.n_nodes = n_nodes
        # each specialist is a dict with its left-most ('l') and right-most ('r') nodes as items
        # keep a dict of specialists, each with an id 0,...,N
        # add the first specialist manually

        # NODES START AT 1 !!!!
        self.specialist_dict = {0: {'l': 1, 'r': self.n_nodes}}

        # keep a list of specialists that need splitting
        self.to_split = [0]

        # keep track of the next id for the next specialist
        self.next_specialist_id = 1

        self.n_specialists = None

        self.build_set()

    def build_set(self):
        # keep splitting specialists until we have only 'singletons'

        while len(self.to_split) > 0:
            # pop(0) removes the first item from the list, and passes it as an argument to the split function
            self.split(self.to_split.pop(0))

        self.n_specialists = self.next_specialist_id

    def split(self, specialist_id):
        specialist = self.specialist_dict[specialist_id]

        left = specialist['l']
        right = specialist['r']

        # split looks like:
        # [left -------------------------------------------- right]
        # splits to
        # [abs_left ----- inner_right][inner_left ------ abs_right]
        absolute_left = left
        absolute_right = right
        inner_right = int(np.floor( (left + right)/2 ))
        inner_left = int(np.floor( (left + right)/2 ) + 1)

        left_split = {'l': absolute_left,
                      'r': inner_right}
        right_split = {'l': inner_left,
                       'r': absolute_right}

        # for each new specialist, add it to our dict, check if it needs splitting further, update next id
        for spec in [left_split, right_split]:
            self.specialist_dict[self.next_specialist_id] = spec

            if spec['l'] != spec['r']:
                self.to_split.append(self.next_specialist_id)

            self.next_specialist_id += 1

    def get_spec_ids_for_nodes(self):
        node_dict = {}

        for node in range(self.n_nodes):
            # NODES START AT 1 IN 'self.specialist_dict'
            node_plus = node + 1

            # keep a list of 'active' specialists for each node
            specialist_list = []

            for spec_id, spec in self.specialist_dict.items():
                if (node_plus >= spec['l']) and (node_plus <= spec['r']):
                    specialist_list.append(spec_id)

            node_dict[node] = np.array(specialist_list)

        return node_dict


def get_fs_set_for_nodes(n_nodes):
    cs_left = {}
    cs_right = {i: np.zeros(i+1) for i in range(n_nodes)}
    specialist_id = 0
    nodes = {}
    for node in range(n_nodes):
        first_specialist = specialist_id
        for length in range(n_nodes - node):
            #  Get the left and right ends for the node,
            #  and store them in the cumulative arrays
            right_end = node + length
            cs_right[right_end][node] = specialist_id

            specialist_id += 1

        cs_left[node] = np.array(range(first_specialist, specialist_id))

        if node == 0:  # First node
            nodes[node] = cs_left[node]
        else:
            # Using 'cs_left', node i will it's set of specialists starting at that
            # node (from the left), and will also have the same specialists as it's
            # left-neighbor, except for those specialists which have that neighbor
            # as it's right-end node. These can be quickly found using cs_right.

            # candidates = list(set(nodes[node - 1]).difference(cs_right[node - 1]))
            candidates = np.setdiff1d(nodes[node - 1], cs_right[node - 1])
            nodes[node] = np.concatenate((candidates, cs_left[node]))

    return nodes


if __name__ == "__main__":
    import sys

    n = 8
    bset = BinarySet(n)

    print(bset.get_spec_ids_for_nodes())
    print(bset.n_specialists)
