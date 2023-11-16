##  RESCUER AGENT
### @Author: Tacla (UTFPR)
### Demo of use of VictimSim

import os
import numpy as np
import random
import pandas as pd
from sklearn.metrics import accuracy_score, confusion_matrix, precision_score, recall_score, ConfusionMatrixDisplay
from sklearn.model_selection import RandomizedSearchCV, train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from abstract_agent import AbstractAgent
from physical_agent import PhysAgent
from abc import ABC, abstractmethod


## Classe que define o Agente Rescuer com um plano fixo
class Rescuer(AbstractAgent):
    def __init__(self, env, config_file):
        """ 
        @param env: a reference to an instance of the environment class
        @param config_file: the absolute path to the agent's config file"""

        super().__init__(env, config_file)

        # Specific initialization for the rescuer
        self.mutation_rate = 0.01
        self.generations = 1000
        self.population_size = 50
        self.plan = []              # a list of planned actions
        self.rtime = self.TLIM      # for controlling the remaining time
        self.model = RandomForestClassifier(criterion='entropy', max_depth=40)
        self.tree = DecisionTreeClassifier(criterion='entropy', max_depth=40)
        self.victims_assigned = {}
        
        # Starts in IDLE state.
        # It changes to ACTIVE when the map arrives
        self.body.set_state(PhysAgent.IDLE)

        # planning
        self.__planner()
    
    def go_save_victims(self, walls, victims):
        """ The explorer sends the map containing the walls and
        victims' location. The rescuer becomes ACTIVE. From now,
        the deliberate method is called by the environment"""
        self.body.set_state(PhysAgent.ACTIVE)

    def assign_victims(self, victim_dict):
        self.victims_assigned = victim_dict.copy()

    # Function to calculate the total distance of a route
    def calculate_distance(self, route):
        total_distance = 0
        for i in range(len(route) - 1):
            victim1, victim2 = route[i], route[i + 1]
            total_distance += np.linalg.norm(np.array(self.victims_assigned[victim1]) - np.array(self.victims_assigned[victim2]))
        return total_distance

    # Function to generate an initial population of routes
    def generate_population(self, size):
        population = []
        cities_list = list(self.victims_assigned.keys())
        for i in range(size):
            route = random.sample(cities_list, len(cities_list))
            population.append(route)
        return population

    # Function to select parents using tournament selection
    def tournament_selection(self, population, k=5):
        tournament = random.sample(population, k)
        return min(tournament, key=self.calculate_distance)

    # Function to perform crossover (order crossover)
    def crossover(self, parent1, parent2):
        start, end = sorted(random.sample(range(len(parent1)), 2))
        child = [None] * len(parent1)
        child[start:end] = parent1[start:end]
        remaining = [city for city in parent2 if city not in child]
        child[:start] = remaining[:start]
        child[end:] = remaining[start:]
        return child

    # Function to perform mutation (swap mutation)
    def mutate(self, route):
        idx1, idx2 = sorted(random.sample(range(len(route)), 2))
        route[idx1], route[idx2] = route[idx2], route[idx1]
        return route

    # Main genetic algorithm function
    def genetic_algorithm(self):
        population = self.generate_population(self.population_size)

        for generation in range(self.generations):
            population = sorted(population, key=self.calculate_distance)[:self.population_size]

            new_population = []

            for _ in range(self.population_size // 2):
                parent1 = self.tournament_selection(population)
                parent2 = self.tournament_selection(population)
                child1 = self.crossover(parent1, parent2)
                child2 = self.crossover(parent2, parent1)

                if random.random() < self.mutation_rate:
                    child1 = self.mutate(child1)
                if random.random() < self.mutation_rate:
                    child2 = self.mutate(child2)

                new_population.extend([child1, child2])

            population = new_population

        best_route = min(population, key=self.calculate_distance)
        best_distance = self.calculate_distance(best_route)

        print("Best Route:", best_route)
        print("Best Distance:", best_distance)

    #gabriel
    def classificate(self, victims):

        df = pd.DataFrame(columns=['id', 'qPA', 'pulso', 'freq_resp'])
        posistions_array_x = []
        posistions_array_y = []
        pos_array = []

        for id, data in victims.items():
            #print(f"id: {id}, qpa: {data.qPA}, pulse: {data.pulse}, resp: {data.resp}")
            pos_array.append(data.pos)

            posistions_array_x.append(data.pos[0])
            posistions_array_y.append(data.pos[1])

            new_row = {'id': data.id, 'qPA': data.qPA, 'pulso': data.pulse, 'freq_resp': data.resp}
            df.loc[len(df.index)] = new_row
                #df = pd.concat([df, new_df], ignore_index=True)
        print(df.info())
        df2 = df.copy()
        df['classe'] = self.model.predict(df)
        df['grav'] = 0
        df['pos'] = pos_array
        df['x'] = posistions_array_x
        df['y'] = posistions_array_y
        df.to_csv('dataframe.csv', index=False)
        clean = ["id", "x", "y", "grav", "classe"]
        clean_df = df[clean]
        clean_df.to_csv('file_predict.txt', index=False)

        df2['classe'] = self.tree.predict(df2)
        df2['grav'] = 0
        df2['pos'] = pos_array
        df2['x'] = posistions_array_x
        df2['y'] = posistions_array_y
        clean2 = ["id", "x", "y", "grav", "classe"]
        clean_df2 = df2[clean]
        clean_df2.to_csv('file_predict2.txt', index=False)

    def learn(self):
        col_names = ["id", "psist", "pdiast", "qPA", "pulso", "freq_resp", "gravidade", "classe"]
        df = pd.read_csv("sinais_balanceados.txt", header=None, names=col_names)
        df.dropna(axis=0, inplace=True)
        print(df['classe'].value_counts)
        print('aashfdiausdsa')
        feature_cols = ["id", "qPA", "pulso", "freq_resp"]
        x = df[feature_cols]
        y = df['classe']

        x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.3, stratify=y)

        self.model.fit(x_train, y_train)
        y_pred = self.model.predict(x_test)




        self.tree.fit(x_train, y_train)
        y_pred2 = self.tree.predict(x_test)
        accuracy = accuracy_score(y_test, y_pred2)
        print("Tree Accuracy:", accuracy)


        # Create the confusion matrix
        cm = confusion_matrix(y_test, y_pred)

        # ConfusionMatrixDisplay(confusion_matrix=cm).plot()
        accuracy = accuracy_score(y_test, y_pred)
        precision = precision_score(y_test, y_pred, average='micro')
        recall = recall_score(y_test, y_pred, average='micro')

        print("Random Forest Accuracy:", accuracy)
        print("Precision:", precision)
        print("Recall:", recall)

        # Create a series containing feature importances from the model and feature names from the training data
        feature_importances = pd.Series(self.model.feature_importances_, index=x_train.columns).sort_values(
            ascending=False)

        # Plot a simple bar chart
        # feature_importances.plot.bar()
    def create_weighted_array(self):
        df = pd.read_csv('dataframe.csv')
        positions_list = []

        for index, row in df.iterrows():
            positions_list.extend([row['pos']] * row['classe'])

        positions_array = np.array(positions_list)
        return positions_array
    
    def __planner(self):
        """ A private method that calculates the walk actions to rescue the
        victims. Further actions may be necessary and should be added in the
        deliberata method"""

        # This is a off-line trajectory plan, each element of the list is
        # a pair dx, dy that do the agent walk in the x-axis and/or y-axis
        self.plan.append((0,1))
        self.plan.append((1,1))
        self.plan.append((1,0))
        self.plan.append((1,-1))
        self.plan.append((0,-1))
        self.plan.append((-1,0))
        self.plan.append((-1,-1))
        self.plan.append((-1,-1))
        self.plan.append((-1,1))
        self.plan.append((1,1))
        
    def deliberate(self) -> bool:
        """ This is the choice of the next action. The simulator calls this
        method at each reasonning cycle if the agent is ACTIVE.
        Must be implemented in every agent
        @return True: there's one or more actions to do
        @return False: there's no more action to do """

        # No more actions to do
        if self.plan == []:  # empty list, no more actions to do
           return False

        # Takes the first action of the plan (walk action) and removes it from the plan
        dx, dy = self.plan.pop(0)

        # Walk - just one step per deliberation
        result = self.body.walk(dx, dy)

        # Rescue the victim at the current position
        if result == PhysAgent.EXECUTED:
            # check if there is a victim at the current position
            seq = self.body.check_for_victim()
            if seq >= 0:
                res = self.body.first_aid(seq) # True when rescued             

        return True

