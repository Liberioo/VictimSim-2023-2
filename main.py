import os
import sys
import numpy as np
from aux_file import priority_directions
from sklearn.cluster import KMeans
from matplotlib import pyplot as plt
from sklearn import datasets
import random

## importa classes
from environment import Env
from explorer_robot import ExplorerRobot
from rescuer import Rescuer
import genetic_algorithm as ga


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

    # Instantiate agents rescuer and explorer
    resc = Rescuer(env, rescuer_file)

    # Explorer needs to know rescuer to send the map
    # that's why rescuer is instatiated before
    exp1 = ExplorerRobot(env, explorer_file_1, priority_directions[0])
    exp2 = ExplorerRobot(env, explorer_file_2, priority_directions[1])
    exp3 = ExplorerRobot(env, explorer_file_3, priority_directions[2])
    exp4 = ExplorerRobot(env, explorer_file_4, priority_directions[3])


    # Run the environment simulator
    env.run()

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

    for id, data in all_victims.items():
        print(f"id: {id}, pos: {data.pos}")

    all_victims_pos = np.array(all_victims_pos)



    def plot_centroids(data, km):
        y_km = km.fit_predict(data)
        color_array = ['lightblue', 'lightgreen', 'orange', 'yellow', 'pink', 'blue', 'green']
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

    iris = datasets.load_iris()
    X = iris.data
    Y = iris.target

    model = KMeans(n_clusters=4, n_init=10)
    model.fit(all_victims_pos)

    def ClusterIndicesNumpy(clustNum, labels_array):  # numpy
        return np.where(labels_array == clustNum)[0]

    victim_pos_dict_0 = {}
    victim_pos_dict_1 = {}
    victim_pos_dict_2 = {}
    victim_pos_dict_3 = {}
    for p in all_victims_pos[ClusterIndicesNumpy(0, model.labels_)]:
        for k, v in all_victims.items():
            if v.pos[0] == p[0] and v.pos[1] == p[1]:
                victim_pos_dict_0[k] = p
    for p in all_victims_pos[ClusterIndicesNumpy(1, model.labels_)]:
        for k, v in all_victims.items():
            if v.pos[0] == p[0] and v.pos[1] == p[1]:
                victim_pos_dict_1[k] = p
    for p in all_victims_pos[ClusterIndicesNumpy(2, model.labels_)]:
        for k, v in all_victims.items():
            if v.pos[0] == p[0] and v.pos[1] == p[1]:
                victim_pos_dict_2[k] = p
    for p in all_victims_pos[ClusterIndicesNumpy(3, model.labels_)]:
        for k, v in all_victims.items():
            if v.pos[0] == p[0] and v.pos[1] == p[1]:
                victim_pos_dict_3[k] = p


    # Define the cities and their coordinates
    victims = victim_pos_dict_0

    # Define parameters
    population_size = 50
    generations = 1000
    mutation_rate = 0.01

    # Function to calculate the total distance of a route
    def calculate_distance(route):
        total_distance = 0
        for i in range(len(route) - 1):
            victim1, victim2 = route[i], route[i + 1]
            total_distance += np.linalg.norm(np.array(victims[victim1]) - np.array(victims[victim2]))
        return total_distance

    # Function to generate an initial population of routes
    def generate_population(size):
        population = []
        victims_list = list(victims.keys())
        for _ in range(size):
            route = random.sample(victims_list, len(victims_list))
            population.append(route)
        return population

    # Function to select parents using tournament selection
    def tournament_selection(population, k=5):
        tournament = random.sample(population, k)
        return min(tournament, key=calculate_distance)

    # Function to perform crossover (order crossover)
    def crossover(parent1, parent2):
        start, end = sorted(random.sample(range(len(parent1)), 2))
        child = [None] * len(parent1)
        child[start:end] = parent1[start:end]
        remaining = [victim for victim in parent2 if victim not in child]
        child[:start] = remaining[:start]
        child[end:] = remaining[start:]
        return child

    # Function to perform mutation (swap mutation)
    def mutate(route):
        idx1, idx2 = sorted(random.sample(range(len(route)), 2))
        route[idx1], route[idx2] = route[idx2], route[idx1]
        return route

    # Main genetic algorithm function
    def genetic_algorithm():
        population = generate_population(population_size)

        for generation in range(generations):
            population = sorted(population, key=calculate_distance)[:population_size]

            new_population = []

            for _ in range(population_size // 2):
                parent1 = tournament_selection(population)
                parent2 = tournament_selection(population)
                child1 = crossover(parent1, parent2)
                child2 = crossover(parent2, parent1)

                if random.random() < mutation_rate:
                    child1 = mutate(child1)
                if random.random() < mutation_rate:
                    child2 = mutate(child2)

                new_population.extend([child1, child2])

            population = new_population

        best_route = min(population, key=calculate_distance)
        best_distance = calculate_distance(best_route)

        print("Best Route:", best_route)
        print("Best Distance:", best_distance)

    genetic_algorithm()
    victims = victim_pos_dict_1
    genetic_algorithm()
    victims = victim_pos_dict_2
    genetic_algorithm()
    victims = victim_pos_dict_3
    genetic_algorithm()



    plot_centroids(all_victims_pos, model)

if __name__ == '__main__':
    """ To get data from a different folder than the default called data
    pass it by the argument line"""

    if len(sys.argv) > 1:
        data_folder_name = sys.argv[1]
    else:
        # data_folder_name = os.path.join("datasets", "data_100x80_132vic")
        data_folder_name = os.path.join("datasets", "data_100x80_225vic")
        # data_folder_name = os.path.join("datasets", "data_20x20_42vic")

    main(data_folder_name)
