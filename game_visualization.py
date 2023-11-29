# script to visualize how agent plays a game
# useful to study different iterations

import numpy as np
from agent import DeepQLearningAgent
from game_environment import Snake, SnakeNumpy
from utils import visualize_game
import json

# some global variables
version = "v17.5"

with open("model_config/{:s}.json".format(version), "r") as f:
    m = json.loads(f.read())
    board_size = m["board_size"]
    frames = m["frames"]  # keep frames >= 2
    max_time_limit = m["max_time_limit"]
    supervised = bool(m["supervised"])
    n_actions = m["n_actions"]
    obstacles = bool(m["obstacles"])

# iteration_list = [4 * (10**5) - 1000 * x for x in range(400)]
# iteration_list = [300000 - 500 * x for x in range(2)]
iteration_list = [774500]
max_time_limit = 398

# setup the environment
env = Snake(
    board_size=board_size,
    frames=frames,
    max_time_limit=max_time_limit,
    obstacles=obstacles,
    version=version,
)
s = env.reset()
n_actions = env.get_num_actions()

# setup the agent
# K.clear_session()
agent = DeepQLearningAgent(
    board_size=board_size,
    frames=frames,
    n_actions=n_actions,
    buffer_size=10,
    version=version,
)
# agent = PolicyGradientAgent(board_size=board_size, frames=frames, n_actions=n_actions, buffer_size=10)
# agent = AdvantageActorCriticAgent(board_size=board_size, frames=frames, n_actions=n_actions, buffer_size=10)
# agent = HamiltonianCycleAgent(board_size=board_size, frames=frames, n_actions=n_actions, buffer_size=10)
# agent = BreadthFirstSearchAgent(board_size=board_size, frames=frames, n_actions=n_actions, buffer_size=10)

for iteration in iteration_list:
    agent.load_model(file_path="models/{:s}".format(version), iteration=iteration)

    for i in range(5):
        visualize_game(
            env,
            agent,
            path="images/game_visual_{:s}_{:d}_14_ob_{:d}.mp4".format(
                version, iteration, i
            ),
            debug=False,
            animate=True,
            fps=12,
            min_food_count=-1,
        )
