import os
import sys


from aux_file import priority_directions

## importa classes
from environment import Env
from explorer_robot import ExplorerRobot
from rescuer import Rescuer
import pickle
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
import pandas as pd
from sklearn.cluster import KMeans

def main(data_folder_name):
    # Set the path to config files and data files for the environment
    current_folder = os.path.abspath(os.getcwd())
    data_folder = os.path.abspath(os.path.join(current_folder, data_folder_name))

    all_victims = {}

    # Instantiate the environment
    env = Env(data_folder)

    # config files for the agents
    rescuer_file = os.path.join(data_folder, "rescuer_config.txt")

    explorer_file_1 = os.path.join(data_folder, "explorer_config_q1.txt")
    explorer_file_2 = os.path.join(data_folder, "explorer_config_q2.txt")
    explorer_file_3 = os.path.join(data_folder, "explorer_config_q3.txt")
    explorer_file_4 = os.path.join(data_folder, "explorer_config_q4.txt")

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

    resc1 = Rescuer(env,explorer_file_1)



    #filename = "tree.pkl"
    #arvore = pickle.load(open(filename, 'rb'))

    for id, data in exp1.victims.items():
        all_victims[id] = data
        print(data.resp)

    for id, data in exp2.victims.items():
        all_victims[id] = data

    for id, data in exp3.victims.items():
        all_victims[id] = data

    for id, data in exp4.victims.items():
        all_victims[id] = data

    resc1.learn()
    resc1.classificate(all_victims)


    #for id, data in all_victims.items():
        #print(f"id: {id}, qpa: {data.qPA}, pulse: {data.pulse}, resp: {data.resp}")

        #new_row = {'id': data.id, 'qPA': data.qPA, 'pulso': data.pulse, 'freq_resp': data.resp}
        #df = df.append(new_row, ignore_index=True)


    # df['id'] = df['id'].astype(int)
    #
    # print(type(all_victims))
    # print(df.head())
    # #df['classe'] = arvore.predict(df)
    # df['grav'] = 0
    # df['pos'] = pos_array
    # df['x'] = posistions_array_x
    # df['y'] = posistions_array_y

    # clean = ["id", "x","y", "grav", "classe"]
    # clean_df = df[clean]
    # salvos = clean_df.sample(frac = 0.8)
    # print(clean_df.head())
    # clean_df.to_csv('resultados.csv', index=False)
    # salvos.to_csv('file_predict.txt', index = False)

    # model = KMeans(n_clusters = 4, n_init = 10)
    # model.fit(pos_array)
    # predict = model.predict(pos_array)
    # clusterizado = clean_df
    # clusterizado['cluster'] = predict
    #
    # mask = clusterizado['cluster'] == 0
    # cluster1 = clusterizado[mask]
    # cluster2 = clusterizado[clusterizado['cluster'] == 1]
    # cluster3 = clusterizado[clusterizado['cluster'] == 2]
    # cluster4 = clusterizado[clusterizado['cluster'] == 3]
    #
    # cluster1.to_csv('cluster1.txt', index=False)
    # cluster2.to_csv('cluster2.txt', index=False)
    # cluster3.to_csv('cluster3.txt', index=False)
    # cluster4.to_csv('cluster4.txt', index=False)



if __name__ == '__main__':
    """ To get data from a different folder than the default called data
    pass it by the argument line"""

    if len(sys.argv) > 1:
        data_folder_name = sys.argv[1]
    else:
        #data_folder_name = os.path.join("datasets", "data_100x80_132vic")
        data_folder_name = os.path.join("datasets", "teste_cego")

    main(data_folder_name)
