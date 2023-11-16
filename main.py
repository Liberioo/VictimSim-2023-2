import os
import sys

from aux_file import priority_directions

## importa classes
from environment import Env
from explorer_robot import ExplorerRobot
from rescue_organizer import RescueOrganizer

def main(data_folder_name):
    # Set the path to config files and data files for the environment
    current_folder = os.path.abspath(os.getcwd())
    data_folder = os.path.abspath(os.path.join(current_folder, data_folder_name))


    # Instantiate the environment
    env = Env(data_folder)

    # config files for the agents
    explorer_file = os.path.join(data_folder, "explorer_config_q1.txt")

    rescuer_config = os.path.join(data_folder, "rescuer_config.txt")

    # Instantiate agents rescuer and explorer
    #resc = Rescuer(env, rescuer_file)

    rescue_organizer = RescueOrganizer(env, rescuer_config)

    # Explorer needs to know rescuer to send the map
    # that's why rescuer is instatiated before
    exp1 = ExplorerRobot(env, explorer_file, priority_directions[0], rescue_organizer)
    exp2 = ExplorerRobot(env, explorer_file, priority_directions[1], rescue_organizer)
    exp3 = ExplorerRobot(env, explorer_file, priority_directions[2], rescue_organizer)
    exp4 = ExplorerRobot(env, explorer_file, priority_directions[3], rescue_organizer)


    # Run the environment simulator
    env.run()

if __name__ == '__main__':
    """ To get data from a different folder than the default called data
    pass it by the argument line"""

    if len(sys.argv) > 1:
        data_folder_name = sys.argv[1]
    else:
        #data_folder_name = os.path.join("datasets", "data_100x80_132vic")
        data_folder_name = os.path.join("datasets", "data_100x80_225vic")

    main(data_folder_name)
