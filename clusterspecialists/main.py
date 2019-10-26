import os
import json
import numpy as np
import matplotlib
matplotlib.use('Agg')  # Stop figures from opening (save only)
from matplotlib.lines import Line2D
from clusterspecialists.graphs.graphs import Graph
from clusterspecialists.algorithms import Perceptron, Specialists, NaiveHeterogeneous, NaiveHomogeneous,\
    NotSoNaive, GlobalTemporal
import matplotlib.pyplot as plt
import seaborn as sns
from multiprocess import Pool as mPool
from scipy.sparse.csgraph import minimum_spanning_tree
from clusterspecialists.graphs.optimalparameters import get_universal_majority, get_node_majorities
sns.set_style('white')  # Style plots


def generate_graph(knn_matrix, distance_matrix):
    """
    Generates a Graph object using the union of a k-nearest neighbours graph and a minimum spanning tree.
    The dir attribute is also defined after this method is called.

    :param k_nearest_neighbor: (int) The value of k in the nearest neighbour algorithm
    :return:
    """
    mst = minimum_spanning_tree(distance_matrix)

    union = knn_matrix + mst

    # Make the matrix symmetric, since the k_nn and mst matrices aren't, and we
    # want an undirected adjacency matrix
    union = union + union.T
    union = union != 0  # Use boolean masking to get an adjacency matrix (mst still has distances)
    union = union.astype(int)

    graph = Graph(data_adj_matrix=union)
    print('Connected Graph generated')
    return graph


def generate_binary_labelings(station_states_filename, station_to_node_map, percentage_threshold):
    """

    :param station_states_filename:
    :param station_to_node_map:
    :param percentage_threshold:
    :return:
    """
    with open(station_states_filename, 'r') as json_file:
        station_states = json.load(json_file)

    labelings = {}

    for segment, data in station_states.items():
        labeling = {}
        for station, percentage in data['percentages'].items():
            node = station_to_node_map[station]
            labeling[node] = int(int(percentage >= percentage_threshold)*2 - 1)

        labelings[int(segment)] = labeling

    return labelings


def get_hourly_timestamps(station_states_filename, resolution):
    timestamps = []

    with open(station_states_filename, 'r') as json_file:
        station_states = json.load(json_file)

    for segment, data in station_states.items():
        if resolution == 'ten_minute':
            if int(segment) % 6 != 0:
                continue
        timestamp = str(data['timestamp'])
        time = timestamp.split('T')[1]
        time = time.split('.')[0]
        timestamps.append(time[:-3])

    return timestamps


# def get_ids_of_switching_nodes(labelings):
#     switching_nodes = set()
#
#     previous_labeling = labelings[0]
#
#     for segment, labeling in labelings.items():
#         if segment == 0:  # skip first labeling
#             continue
#         for node, label in labeling.items():  # The labeling has the form {node_id: label}
#             if label != previous_labeling[node]:
#                 switching_nodes.add(node)
#
#         previous_labeling = labeling
#
#     return list(switching_nodes)


def get_labels_as_array(labelings, sequence, n_trials_per_segment):
    total_t = len(sequence)

    labeling_list = []
    for t in range(total_t):
        segment = int(np.floor(t / n_trials_per_segment))
        current_node = sequence[t]
        labeling_list.append(labelings[segment][current_node])

    return np.array(labeling_list)



class ParallelTest(object):
    """
    Runs experiments on all 6 cores
    """
    def __init__(self, parameters, n_processes):
        self.results = {}
        self.parameters = parameters
        self.n_segments = parameters['n_segments']
        self.n_trials_per_segment = parameters['n_trials_per_segment']
        self.n_processes = n_processes
        self.seed = parameters['seed']


        # Parameters are alpha (specialists) and gamma (perceptron)
        self.alpha = parameters['alpha'] if parameters['alpha'] is not None else None
        self.gamma = parameters['gamma'] if parameters['gamma'] is not None else None

        # For the naive algorithms
        self.training_labelings = parameters['training_labelings']
        self.universal_majority_label = parameters['universal_majority_label']
        self.node_majority_labels = parameters['node_majority_labels']

        self.ensemble = parameters['ensemble']
        self.algorithm = parameters['algorithm']
        self.sequence = parameters['sequence']

    def run_single(self, indices):
        # for the given indices, we instantiate the algorithm and run over the sequence

        alg_params = {'graph': self.parameters['graph'],
                      'scheme': self.parameters['scheme'],
                      'alpha': self.alpha,
                      'max_radius': self.gamma,
                      'training_labelings': self.training_labelings,
                      'universal_majority_label': self.universal_majority_label,
                      'node_majority_labels': self.node_majority_labels,
                      'n_trials_per_segment': self.n_trials_per_segment
        }


        # Multiple process get the same random sequence! Set a seed based on index
        # Each spine algorithm will have the same spine, but when doing ensembles each group will contain different
        # spines (although each group is the same) - this is probably fairer.
        if indices != [(0, 0)]:
            np.random.seed(self.seed + indices[0])
        else:
            np.random.seed(self.seed)

        algorithm = self.algorithm(**alg_params)

        this_algorithms_predictions = []
        t = 0
        for segment in range(self.n_segments):
            for segment_trial in range(self.n_trials_per_segment):
                if t % 500 == 0:
                    print('trial {}'.format(t))

                # Ge this trial's node
                node = self.parameters['sequence'][t]

                prediction = algorithm.predict(node)

                this_algorithms_predictions.append(np.sign(prediction))

                true_label = labelings[segment][node]

                # Update the instance with the true label
                algorithm.update(vi=node, prediction=prediction, label=true_label)

                t += 1

        return this_algorithms_predictions

    def run_parallel(self):
        pool = mPool(processes=self.n_processes)  # , maxtasksperchild=20)

        indices = [[(i, 0)] for i in range(self.ensemble)]
        keys = [i for i in range(self.ensemble)]

        return_values = pool.starmap(self.run_single, indices)

        pool.close()
        pool.join()
        pool.terminate()
        del pool

        return dict(zip(keys, return_values))


if __name__ == "__main__":

    # Specify paths
    output_images_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'results', 'temp'))
    data_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'data'))

    # Do the small graph if mini=True, otherwise do the bigger graph
    # All required files for the small matrix have a prefix of 'mini_' in their filename

    # mini = False
    mini = True

    if mini:
        file_prefix = 'mini_'
    else:
        file_prefix = ''

    # Load data
    # NOTE Station id's do NOT correspond to node id's, so we have to map them
    with open(os.path.join(data_path, file_prefix + 'station_to_node_map.json')) as json_file:
        station_to_node_map = json.load(json_file)  # {station_id : node_id }
    with open(os.path.join(data_path, file_prefix + 'node_to_station_map.json')) as json_file:
        node_to_station_map = json.load(json_file)  # {node_id : station_id }

    plot_colors = {'perceptron': 'blue',
                   'fully-covering': 'red',
                   'binary-tree': 'green',
                   'naive-homogeneous': 'black',
                   'naive-heterogeneous': 'purple',
                   'not-so-naive': 'brown',
                   'global-temporal': 'cyan'}

    squash_factor = 4  # If 1 then 24 hours shown on top, if 2 then 12 hours shown on top etc.

    # Define parameters

    # mode = 'train'
    mode = 'test'
    resolution = 'ten_minute'
    # resolution = 'hourly'
    k_nearest_neighbor = 3
    n_iterations = 25
    percentage_threshold = 50
    ensemble_range = [1, 3, 5, 9, 17, 33, 65]
    n_processes = 12
    seed = 123
    # Optimal values
    scs_b_optimal_alpha = 0.0003
    scs_f_optimal_alpha = 7.4e-10
    sgp_optimal_radius = 3.89

    n_hours = 48
    if resolution == 'hourly':
        n_trials_per_segment = 180
        station_states_filename = os.path.join(data_path, file_prefix + 'test_station_hourly_states.json')
        n_segments = n_hours  # one segment is one hour
    elif resolution == 'ten_minute':
        n_trials_per_segment = 30
        station_states_filename = os.path.join(data_path, file_prefix + 'test_station_ten_minute_states.json')
        n_segments = n_hours * 6


    # Get naive predictions on training data
    naive_station_states_filename = os.path.join(data_path, file_prefix + 'train_station_{}_states.json'.format(resolution))
    training_labelings = generate_binary_labelings(station_states_filename=naive_station_states_filename,
                                                   station_to_node_map=station_to_node_map,
                                                   percentage_threshold=percentage_threshold)
    trained_universal_majority = get_universal_majority(training_labelings)
    trained_node_majorities = get_node_majorities(training_labelings)



    # Prepare the graph

    # Use kneighbors_graph and minimum_spanning_tree
    knn_matrix = np.load(os.path.join(data_path, file_prefix + '{}_knn_matrix.npy'.format(k_nearest_neighbor)))
    distance_matrix = np.load(os.path.join(data_path, file_prefix + 'station_distance_matrix.npy'))

    # Generate Graph
    graph = generate_graph(knn_matrix=knn_matrix, distance_matrix=distance_matrix)

    # Generate Labelings
    labelings = generate_binary_labelings(station_states_filename=station_states_filename,
                                          station_to_node_map=station_to_node_map,
                                          percentage_threshold=percentage_threshold)

    # Get the set of nodes that switch labeling
    # switching_nodes = get_ids_of_switching_nodes(labelings)
    switching_nodes = list(graph._nodes)

    print('{} nodes switch labeling in this sequence'.format(len(switching_nodes)))

    # Algorithms to implement
    schemes = ['naive-homogeneous', 'perceptron', 'naive-heterogeneous',  'not-so-naive', 'fully-covering',
               'binary-tree', 'global-temporal']

    non_spine_schemes = ['perceptron', 'naive-homogeneous', 'naive-heterogeneous', 'not-so-naive', 'global-temporal']

    np.random.seed(seed)

    # Main loop
    for ensemble in ensemble_range:
        print('Running with ensemble size {}'.format(ensemble))

        # Meta-dictionary for all algorithms, all results etc
        results = {scheme: {'total_mistakes': []} for scheme in schemes}

        for iteration in range(n_iterations):
            print('Running iteration {0} of {1}'.format(iteration + 1, n_iterations))

            # Choose a sequence of nodes to test on in this iteration
            sequence = np.random.choice(switching_nodes, n_trials_per_segment*n_segments, replace=True)
            labeling_array = get_labels_as_array(labelings, sequence, n_trials_per_segment)

            # Initialize ensembles
            for scheme in schemes:
                parameters = {'graph': graph,
                              'n_segments': n_segments,
                              'n_trials_per_segment': n_trials_per_segment,
                              'sequence': sequence,
                              'ensemble': ensemble if scheme not in non_spine_schemes else None,
                              'scheme': scheme,
                              'algorithm':Perceptron if scheme == 'perceptron' else
                                          NotSoNaive if scheme == 'not-so-naive' else
                                          NaiveHeterogeneous if scheme == 'naive-heterogeneous' else
                                          NaiveHomogeneous if scheme == 'naive-homogeneous' else
                                          GlobalTemporal if scheme == 'global-temporal' else
                                          Specialists,
                              'alpha': scs_f_optimal_alpha if scheme == 'fully-covering' else
                                      scs_b_optimal_alpha if scheme == 'binary-tree' else None,
                              'gamma': sgp_optimal_radius if scheme == 'perceptron' else None,
                              'training_labelings': training_labelings if scheme in ['not-so-naive', 'global-temporal']
                              else None,
                              'universal_majority_label': trained_universal_majority if scheme == 'naive-homogeneous'
                              else None,
                              'node_majority_labels': trained_node_majorities if scheme == 'naive-heterogeneous'
                              else None,
                              'seed': seed + iteration  # use a different seed on each iteration
                              }

                parallel = ParallelTest(parameters=parameters, n_processes=n_processes)

                if scheme in non_spine_schemes:
                    # A hack. Mimic the multiprocessing with a single instance
                    prediction_lists = {0: parallel.run_single((0, 0))}
                else:
                    prediction_lists = parallel.run_parallel()

                # results are in a dictionary for each instance's predictions
                n_instances = len(prediction_lists)  # number of keys

                prediction_array = np.empty((n_instances, len(sequence)))  # empty matrix for the predictions
                for key, prediction_list in prediction_lists.items():
                    prediction_array[key, :] = np.array(prediction_list)

                final_predictions = np.sign(np.sum(prediction_array, axis=0))

                comparison = final_predictions != labeling_array

                mistake_cumsum = np.cumsum(comparison)

                results[scheme]['total_mistakes'].append(mistake_cumsum)

        # Plotting from results dictionary

        # After all iterations are complete, plot them on the graph
        fig = plt.figure(figsize=(12, 12))
        ax = fig.add_subplot(111)

        # Plot the 'switch markers' (vertical dashed lines)
        vline_loc = n_trials_per_segment - 1
        for _ in range(n_segments):
            plt.axvline(x=vline_loc, color='k', linestyle='--', alpha=0.5, linewidth=0.2)
            vline_loc += n_trials_per_segment

        # Plot the data
        for scheme in schemes:
            results[scheme]['total_mistakes'] = np.array(results[scheme]['total_mistakes'])

            # Plot this scheme's data
            sns.tsplot(results[scheme]['total_mistakes'], color=plot_colors[scheme], legend=True, linestyle='-')

        # Legend
        custom_lines = [Line2D([0], [0], color=plot_colors[scheme], lw=2) for scheme in schemes]
        legend = ax.legend(custom_lines, schemes, title='Algorithm', loc='upper left', labelspacing=0.05)

        plt.title('{} Nodes | Ensemble size {} | {} Iterations'.format(len(switching_nodes),
                                                                       ensemble,
                                                                       n_iterations))

        ax_2 = ax.twinx().twiny()  # instantiate a second axes (x and y tweakable)
        # x axis on top
        timestamps = get_hourly_timestamps(station_states_filename, resolution)
        timestamps.append(timestamps[0])  # for plotting
        timestamps = timestamps[0::squash_factor]
        ax_2.set_xlim(ax.get_xlim())
        ax_2.set_xticks(np.linspace(0, n_segments * n_trials_per_segment, len(timestamps)))
        ax_2.set_xticklabels(timestamps)
        plt.setp(ax_2.xaxis.get_majorticklabels(), rotation=45)
        ax_2.set_xlabel('Time')

        ax.set_xticks(np.arange(0, (n_segments + 1) * n_trials_per_segment, 180 * squash_factor))
        ax.set_xticklabels(np.arange(0, (n_segments + 1) * n_trials_per_segment, 180 * squash_factor))

        # y axis on right
        ax_2.set_ylim(ax.get_ylim())
        ax_2.set_yticks(
            np.array([int(np.round(np.mean(results[scheme]['total_mistakes'][:, -1]))) for scheme in schemes]))
        ax_2.set_yticklabels(
            np.array([int(np.round(np.mean(results[scheme]['total_mistakes'][:, -1]))) for scheme in schemes]))
        ax_2.set_ylabel('Final Mistakes')

        ax.set_xlabel('Trial')
        ax.set_ylabel('Cumulative Mistakes')

        plt.savefig(os.path.join(output_images_path,
                             file_prefix + 'parallel-ensemble{}-{}iterations.png'.format(ensemble, n_iterations)))

        # Save the results
        for scheme in schemes:

            results[scheme]['total_mistakes'] = results[scheme]['total_mistakes'].tolist()

        with open(os.path.join(output_images_path,
                               file_prefix + 'ensemble{0}-iterations{1}.json'.format(ensemble, n_iterations)), 'w') as jsonfile:
            json.dump(results, jsonfile)

        # plt.savefig('test-ensemble{}.png'.format(ensemble))
    # plt.show()
