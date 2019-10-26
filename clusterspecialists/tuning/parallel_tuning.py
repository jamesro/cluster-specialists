import numpy as np
import os
import json
from clusterspecialists.algorithms import Perceptron, Specialists
from clusterspecialists.main import generate_graph, generate_binary_labelings  #, get_ids_of_switching_nodes
from multiprocess import Pool as mPool
import time
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd


# =======================================
# Base Tuning class
# =======================================
class Tuning(object):
    """
    OptimizeParams class for online learning algorithms parameters optimization.
    """
    def __init__(self, parameters, n_processes):
        self.results = {}
        self.parameters = parameters
        self.n_segments = parameters['n_segments']
        self.n_trials_per_segment = parameters['n_trials_per_segment']
        self.seed = self.parameters['seed']

        # Parameters are alpha (specialists) and gamma (perceptron)
        self.alpha_range = parameters['alpha_range'] if parameters['alpha_range'] is not None else None
        self.gamma_range = parameters['gamma_range'] if parameters['gamma_range'] is not None else None

        self.algorithm = parameters['algorithm']
        self.n_processes = n_processes

    def run_opt_single(self, indices):
        # for the given indices, we instantiate the algorithm and run over the sequence

        # get the correct parameter for the correct algorithm
        alpha = self.alpha_range[indices[0]] if self.alpha_range is not None else None
        gamma = self.gamma_range[indices[0]] if self.gamma_range is not None else None

        alg_params = {'graph': self.parameters['graph'],
                  'scheme': self.parameters['scheme'],
                  'alpha': alpha,
                  'max_radius': gamma
        }

        np.random.seed(self.seed + indices[0] + indices[1])  # Different spines on every instance

        algorithm = self.algorithm(**alg_params)

        t = 0
        for segment in range(self.n_segments):
            for segment_trial in range(self.n_trials_per_segment):
                if t % 500 == 0:
                    print('trial {}'.format(t))

                # Ge this trial's node
                node = self.parameters['sequence'][t]

                prediction = algorithm.predict(node)

                true_label = labelings[segment][node]

                # Update the instance with the true label
                algorithm.update(vi=node, prediction=prediction, label=true_label)

                t += 1

        return algorithm.mistakes

    def run_opt_parallel(self):
        pool = mPool(processes=self.n_processes)  # , maxtasksperchild=20)
        if self.gamma_range is not None:  # gamma for SGP
            indices = [[(i,0)] for i in range(len(self.gamma_range))]
            keys = [(i,0) for i in range(len(self.gamma_range))]
        else:  # alpha for specialists
            indices = [[(i,0)] for i in range(len(self.alpha_range))]
            keys = [(i,0) for i in range(len(self.alpha_range))]

        return_values = pool.starmap(self.run_opt_single, indices)

        pool.close()
        pool.join()
        pool.terminate()
        del pool
        return dict(zip(keys, return_values))


if __name__ == "__main__":

    # Specify paths
    output_images_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..', 'tuning'))
    data_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..', 'data'))

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



    # Define parameters
    resolution = 'ten_minute'  # 'hourly'
    percentage_threshold = 50
    k_nearest_neighbor = 3
    n_iterations_per_training_step = 10
    n_tuning_steps = 100
    n_processes = 12
    n_hours = 24
    seed = 123


    if resolution == 'hourly':
        n_trials_per_segment = 180
        station_states_filename = os.path.join(data_path, file_prefix + 'train_station_hourly_states.json')
        n_segments = n_hours  # one segment is one hour
    elif resolution == 'ten_minute':
        n_trials_per_segment = 30
        station_states_filename = os.path.join(data_path, file_prefix + 'train_station_ten_minute_states.json')
        n_segments = n_hours * 6

    scs_f_alpha_range = np.geomspace(1e-12, 1e-6, n_tuning_steps)
    scs_b_alpha_range = np.linspace(0.00001, 0.0005, n_tuning_steps)
    sgp_radius_range = np.linspace(3.5, 5, n_tuning_steps)

    ranges = {'fully-covering': scs_f_alpha_range,
              'binary-tree': scs_b_alpha_range,
              'perceptron': sgp_radius_range
              }

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
    schemes = ['fully-covering']

    result_dict = {'perceptron': np.empty((n_iterations_per_training_step, len(sgp_radius_range))) if 'perceptron' in schemes else [],
                   'fully-covering': np.empty((n_iterations_per_training_step, len(scs_f_alpha_range))) if 'fully-covering' in schemes else [],
                   'binary-tree': np.empty((n_iterations_per_training_step, len(scs_b_alpha_range))) if 'binary-tree' in schemes else []
                   }

    for iteration in range(n_iterations_per_training_step):
        np.random.seed(seed + iteration)

        t1 = time.time()

        sequence = np.random.choice(switching_nodes, n_trials_per_segment * n_segments, replace=True)

        for scheme in schemes:
            tuning_parameters = {'graph': graph,
                                 'scheme': scheme,
                                 'seed': seed,
                                 'alpha_range': scs_b_alpha_range if scheme == 'binary-tree' else scs_f_alpha_range if scheme == 'fully-covering' else None,
                                 'gamma_range': sgp_radius_range if scheme == 'perceptron' else None,
                                 'sequence': sequence,
                                 'n_segments': n_segments,
                                 'n_trials_per_segment': n_trials_per_segment,
                                 'algorithm': Perceptron if scheme == 'perceptron' else Specialists
                                 }

            tuner = Tuning(tuning_parameters, n_processes=n_processes)

            results = tuner.run_opt_parallel()

        for key, result in results.items():
            result_dict[scheme][iteration, key[0]] = result

        t2 = time.time()

        print('Iteration {} of {} complete. Time taken: '.format(iteration+1, n_iterations_per_training_step), t2 - t1)

    for scheme in schemes:
        # avg_results = result_dict[scheme]
        # avg_results /= n_iterations_per_training_step

        f, ax = plt.subplots(1, 1, figsize=(20, 10))

        if scheme == 'fully-covering':
            # plt.plot(scs_f_alpha_range, avg_results)
            sns.tsplot((result_dict[scheme]), err_style="unit_traces", ax=ax)
            sns.tsplot((result_dict[scheme]), err_style="ci_band", ax=ax)
            ax.set_xticks(np.arange(0, len(scs_f_alpha_range)))
            ax.set_xticklabels(scs_f_alpha_range)
            plt.xlabel('alpha')
            plt.ylabel('mistakes')
        elif scheme == 'binary-tree':
            sns.tsplot((result_dict[scheme]), err_style="unit_traces", ax=ax)
            sns.tsplot((result_dict[scheme]), err_style="ci_band", ax=ax)
            plt.xlabel('alpha')
            plt.ylabel('mistakes')
        elif scheme == 'perceptron':
            sns.tsplot((result_dict[scheme]), err_style="unit_traces", ax=ax)
            sns.tsplot((result_dict[scheme]), err_style="ci_band", ax=ax)
            plt.xlabel('radius')
            plt.ylabel('mistakes')

        plt.title('{} tuning | Optimal : {}'.format(scheme,
                                                    np.mean(ranges[scheme][np.argmin(result_dict[scheme], axis=1)])))
        if scheme == 'fully-covering':
            plt.savefig(os.path.join(output_images_path, '{} - {} - {}.png'.format(scheme,
                                                                                   scs_f_alpha_range[0],
                                                                                   scs_f_alpha_range[-1])))
        elif scheme == 'binary-tree':
            plt.savefig(os.path.join(output_images_path, '{} - {} - {}.png'.format(scheme,
                                                                                   scs_b_alpha_range[0],
                                                                                   scs_b_alpha_range[-1])))
        elif scheme == 'perceptron':
            plt.savefig(os.path.join(output_images_path, '{} - {} - {}.png'.format(scheme,
                                                                                   sgp_radius_range[0],
                                                                                   sgp_radius_range[-1])))
