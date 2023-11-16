import os
import sys

import numpy as np

from aux_file import priority_directions

## importa classes
from environment import Env
from explorer_robot import ExplorerRobot
from rescuer import Rescuer
import pickle
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
import pandas as pd
from sklearn.cluster import KMeans
from matplotlib import pyplot as plt

def main(data_folder_name):
    # Set the path to config files and data files for the environment
    current_folder = os.path.abspath(os.getcwd())
    data_folder = os.path.abspath(os.path.join(current_folder, data_folder_name))

    all_victims = {}
    all_victims_pos = []

    # Instantiate the environment
    env = Env(data_folder)

    # config files for the agents
    rescuer_file = os.path.join(data_folder, "rescuer_config.txt")

    explorer_file_1 = os.path.join(data_folder, "explorer_config_q1.txt")
    explorer_file_2 = os.path.join(data_folder, "explorer_config_q2.txt")
    explorer_file_3 = os.path.join(data_folder, "explorer_config_q3.txt")
    explorer_file_4 = os.path.join(data_folder, "explorer_config_q4.txt")

    rescuer_config = os.path.join(data_folder, "rescuer_config.txt")

    # Instantiate agents rescuer and explorer
    #resc = Rescuer(env, rescuer_file)

    # Explorer needs to know rescuer to send the map
    # that's why rescuer is instatiated before
    exp1 = ExplorerRobot(env, explorer_file_1, priority_directions[0])
    exp2 = ExplorerRobot(env, explorer_file_2, priority_directions[1])
    exp3 = ExplorerRobot(env, explorer_file_3, priority_directions[2])
    exp4 = ExplorerRobot(env, explorer_file_4, priority_directions[3])


    # Run the environment simulator
    env.run()

    resc1 = Rescuer(env, rescuer_config)
    resc2 = Rescuer(env, rescuer_config)
    resc3 = Rescuer(env, rescuer_config)
    resc4 = Rescuer(env, rescuer_config)


    #filename = "tree.pkl"
    #arvore = pickle.load(open(filename, 'rb'))

    for id, data in exp1.victims.items():
        all_victims[id] = data
        all_victims_pos.append(data.pos)

    for id, data in exp2.victims.items():
        all_victims[id] = data
        all_victims_pos.append(data.pos)

    for id, data in exp3.victims.items():
        all_victims[id] = data
        all_victims_pos.append(data.pos)

    for id, data in exp4.victims.items():
        all_victims[id] = data
        all_victims_pos.append(data.pos)

    all_victims_pos = np.array(all_victims_pos)

    resc1.learn()
    resc1.classificate(all_victims)
    weighted_pos = resc1.create_weighted_array()

    def plot_centroids(data, km):
        y_km = km.fit_predict(data)
        color_array = ['lightblue', 'lightgreen', 'orange', 'yellow']
        for i in range(len(km.cluster_centers_)):
            plt.scatter(
                data[y_km == i, 0], data[y_km == i, 1],
                s=50, c=color_array[i],
                marker='s', edgecolor='black',
                label=f'cluster {i + 1}'
            )

        # plot the centroids
        plt.scatter(
            km.cluster_centers_[:, 0], km.cluster_centers_[:, 1],
            s=250, marker='*',
            c='red', edgecolor='black',
            label='centroids'
        )
        plt.legend(scatterpoints=1)
        plt.grid()
        plt.show()

    km = KMeans(n_clusters=4, n_init=10)
    km.fit(all_victims_pos)
    plot_centroids(all_victims_pos, km)

    def get_id_from_pos(vict_dict, pos):
        for k, v in vict_dict.items():
            if v.pos[0] == pos[0] and v.pos[1] == pos[1]:
                return k
        return None

    cluster1 = {}
    cluster2 = {}
    cluster3 = {}
    cluster4 = {}

    for pos in all_victims_pos[np.where(km.labels_ == 0)[0]]:
        cluster1[get_id_from_pos(all_victims, pos)] = pos
    for pos in all_victims_pos[np.where(km.labels_ == 1)[0]]:
        cluster2[get_id_from_pos(all_victims, pos)] = pos
    for pos in all_victims_pos[np.where(km.labels_ == 2)[0]]:
        cluster3[get_id_from_pos(all_victims, pos)] = pos
    for pos in all_victims_pos[np.where(km.labels_ == 3)[0]]:
        cluster4[get_id_from_pos(all_victims, pos)] = pos

    resc1.assign_victims(cluster1)
    resc2.assign_victims(cluster2)
    resc3.assign_victims(cluster3)
    resc4.assign_victims(cluster4)
    resc1.genetic_algorithm()
    resc2.genetic_algorithm()
    resc3.genetic_algorithm()
    resc4.genetic_algorithm()

if __name__ == '__main__':
    """ To get data from a different folder than the default called data
    pass it by the argument line"""

    if len(sys.argv) > 1:
        data_folder_name = sys.argv[1]
    else:
        #data_folder_name = os.path.join("datasets", "data_100x80_132vic")
        data_folder_name = os.path.join("datasets", "data_100x80_225vic")

    main(data_folder_name)
