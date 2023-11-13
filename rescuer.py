##  RESCUER AGENT
### @Author: Tacla (UTFPR)
### Demo of use of VictimSim

import os
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
        self.plan = []              # a list of planned actions
        self.rtime = self.TLIM      # for controlling the remaining time
        self.model = RandomForestClassifier(criterion='entropy', max_depth=40)
        self.tree = DecisionTreeClassifier(criterion='entropy', max_depth=40)
        
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

    #gabriel
    def classificate(self,victims):

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
        df = pd.read_csv("sinais_pesos.txt", header=None, names=col_names)
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

        ConfusionMatrixDisplay(confusion_matrix=cm).plot()
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
        feature_importances.plot.bar()
    
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

