import numpy as np


def get_labeling_as_array(labeling):
    """
    Converts a given labeling (in dict form) to an np.array
    :param labeling: dict ( {node_id : label} )
    :return: np.array
    """
    n_nodes = len(labeling)
    labeling_array = np.array([labeling[node] for node in range(n_nodes)])

    return labeling_array.reshape(labeling_array.shape[0], 1)


def get_graph_cut_size(labeling, graph):
    """
        Get the cut-size of a given labeling on the original graph

    :param labeling: dict ( {node_id : label} )
    :param graph: custom graph object with laplacian property
    :return: cut-size (integer)

    """
    # Convert labeling to numpy array
    np_labeling = get_labeling_as_array(labeling)

    return int(np.dot(np_labeling.T, np.dot(graph.laplacian, np_labeling))) / 4


def get_spine_cut_size(labeling, spine):
    """
    Get the cut-size of a given labeling on a given spine
    :param labeling: dict ( {node_id : label} )
    :param spine:
    :return:
    """
    cut_size = 0
    for i in range(1, len(spine)):
        if labeling[spine[i]] != labeling[spine[i - 1]]:
            cut_size += 1
    return cut_size


def get_optimal_max_radius(graph, labelings):
    n = graph.n_nodes
    G = graph.laplacian + (1 / ((n ** 2) * graph.r_g)) * np.ones((n, n))  # Inverse of Kernel
    n_segments = len(labelings)

    max_radius = 0
    for j in range(n_segments):
        labeling = labelings[j]
        np_labeling = get_labeling_as_array(labeling)
        radius = np.sqrt(np.dot(np_labeling.T, np.dot(G, np_labeling))[0, 0])
        if radius > max_radius:
            max_radius = radius
    return max_radius


def get_universal_majority(labelings):
    positive_labels = 0
    negative_labels = 0
    for segment, labeling in labelings.items():
        for node, label in labeling.items():
            if label > 0:
                positive_labels += 1
            else:
                negative_labels += 1
    if positive_labels > negative_labels:
        return 1
    elif negative_labels > positive_labels:
        return -1
    else: ## a tie
        return np.random.choice([-1, 1])


def get_node_majorities(labelings):
    n_nodes = len(labelings[0])
    counts = {node: 0 for node in range(n_nodes)}

    for segment, labeling in labelings.items():
        for node, label in labeling.items():
            counts[node] = counts[node] + label

    node_majorities = {}

    for node, count in counts.items():
        if count == 0:  # break a tie arbitrarily
            node_majorities[node] = np.random.choice([-1, 1])
        else:
            node_majorities[node] = np.sign(count)

    return node_majorities