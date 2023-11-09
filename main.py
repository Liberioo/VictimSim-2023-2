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
    resc = Rescuer(env, rescuer_file)

    # Explorer needs to know rescuer to send the map
    # that's why rescuer is instatiated before
    exp1 = ExplorerRobot(env, explorer_file_1, priority_directions[0])
    exp2 = ExplorerRobot(env, explorer_file_2, priority_directions[1])
    exp3 = ExplorerRobot(env, explorer_file_3, priority_directions[2])
    exp4 = ExplorerRobot(env, explorer_file_4, priority_directions[3])


    # Run the environment simulator
    env.run()

    df = pd.DataFrame(columns=['id', 'qPA', 'pulso', 'freq_resp'])
    posistions_array_x = []
    posistions_array_y = []
    pos_array = []


    filename = "arvore.pkl"
    arvore = pickle.load(open(filename, 'rb'))

    for id, data in exp1.victims.items():
        all_victims[id] = data
        print(data.resp)

    for id, data in exp2.victims.items():
        all_victims[id] = data

    for id, data in exp3.victims.items():
        all_victims[id] = data

    for id, data in exp4.victims.items():
        all_victims[id] = data
    
    for id, data in all_victims.items():
        print(f"id: {id}, qpa: {data.qPA}, pulse: {data.pulse}, resp: {data.resp}")

        pos_array.append(data.pos)
        posistions_array_x.append(data.pos[0])
        posistions_array_y.append(data.pos[1])

        new_row = {'id': data.id, 'qPA': data.qPA, 'pulso': data.pulse, 'freq_resp': data.resp}
        df = df.append(new_row, ignore_index=True)
    print(type(all_victims))
    print(df.head())
    df['classe'] = arvore.predict(df)
    df['grav'] = 0
    #df['x'] = posistions_array_x  caso você queira as posicoes no df em x,y ao inves de tuple
    #df['y'] = posistions_array_y   é só desfazer esses comentários
    df['pos'] = pos_array

    clean = ["id", "pos", "grav", "classe"] # eai aqui trocar "pos" por "x", "y"
    clean_df = df[clean]
    print(clean_df.head())
    clean_df.to_csv('resultados.csv', index=False)


    #print(df.head())





    #df = pd.DataFrame(all_victims)
    #df = df.T
    #print(df.head())
    #filename = "arvore.pkl"
    #arvere = pickle.load(open(filename, 'rb'))
    #print(all_victims)
if __name__ == '__main__':
    """ To get data from a different folder than the default called data
    pass it by the argument line"""

    if len(sys.argv) > 1:
        data_folder_name = sys.argv[1]
    else:
        # data_folder_name = os.path.join("datasets", "data_100x80_132vic")
        data_folder_name = os.path.join("datasets", "data_20x20_42vic")

    main(data_folder_name)
