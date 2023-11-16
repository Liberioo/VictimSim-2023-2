from pprint import pprint

from rescuer import Rescuer
from sklearn.cluster import KMeans
import numpy as np


class RescueOrganizer:

    def __init__(self, env, config_file):
        self.rescuers = [Rescuer(env, config_file) for _ in range(0, 4)]
        self.finished_explorers = 0
        self.all_victims = {}
        self.pos_matrix = None

    def update_finished_explorers(self, victims, pos_matrix):
        self.finished_explorers += 1
        for k, v in victims.items():
            self.all_victims[k] = v
        if self.pos_matrix is None:
            self.pos_matrix = pos_matrix[:]
        else:
            for i in range(0, len(pos_matrix)):
                for j in range(0, len(pos_matrix)):
                    if pos_matrix[i][j] != -1 and self.pos_matrix[i][j] == -1:
                        self.pos_matrix[i][j] = pos_matrix[i][j]
                    elif pos_matrix[i][j] < self.pos_matrix[i][j] and pos_matrix[i][j] != -1:
                        self.pos_matrix[i][j] = pos_matrix[i][j]

        if self.finished_explorers == 2:
            self.prepare_and_send_rescuers()

    def prepare_and_send_rescuers(self):
        for i in range(len(self.pos_matrix)):
            for j in range(len(self.pos_matrix)):
                if self.pos_matrix[i][j] == -1:
                    self.pos_matrix[i][j] = 10000000

        rescuer_leader = self.rescuers[0]
        rescuer_leader.learn()
        rescuer_leader.classificate(self.all_victims)

        pos_list = []
        for k, v in self.all_victims.items():
            pos_list.append(v.pos)
        pos_list = np.array(pos_list)
        weighted_list = rescuer_leader.create_weighted_array()

        model = KMeans(n_clusters=4, n_init=10)
        model.fit(weighted_list)
        model.predict(pos_list)

        cluster1 = {}
        cluster2 = {}
        cluster3 = {}
        cluster4 = {}

        for pos in weighted_list[np.where(model.labels_ == 0)[0]]:
            cluster1[get_id_from_pos(self.all_victims, pos)] = pos
        for pos in weighted_list[np.where(model.labels_ == 1)[0]]:
            cluster2[get_id_from_pos(self.all_victims, pos)] = pos
        for pos in weighted_list[np.where(model.labels_ == 2)[0]]:
            cluster3[get_id_from_pos(self.all_victims, pos)] = pos
        for pos in weighted_list[np.where(model.labels_ == 3)[0]]:
            cluster4[get_id_from_pos(self.all_victims, pos)] = pos

        self.rescuers[0].assign_victims(cluster1)
        self.rescuers[0].genetic_algorithm(self.pos_matrix)

        self.rescuers[1].assign_victims(cluster2)
        self.rescuers[1].genetic_algorithm(self.pos_matrix)

        self.rescuers[2].assign_victims(cluster3)
        self.rescuers[2].genetic_algorithm(self.pos_matrix)

        self.rescuers[3].assign_victims(cluster4)
        self.rescuers[3].genetic_algorithm(self.pos_matrix)

        self.rescuers[0].go_save_victims(None, None)
        self.rescuers[1].go_save_victims(None, None)
        self.rescuers[2].go_save_victims(None, None)
        self.rescuers[3].go_save_victims(None, None)


def get_id_from_pos(vict_dict, pos):
    for k, v in vict_dict.items():
        if v.pos[0] == pos[0] and v.pos[1] == pos[1]:
            return k
    return None
