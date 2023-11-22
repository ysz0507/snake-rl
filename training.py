"""
script for training the agent for snake using various methods
"""
# run on cpu
import os

# os.environ['CUDA_VISIBLE_DEVICES'] = '-1'

import numpy as np
from tqdm import tqdm
from collections import deque
import pandas as pd
import time
from utils import play_game, play_game2
from game_environment import SnakeNumpy
from agent import DeepQLearningAgent
import json
import sys

# some global variables
version = sys.argv[1] if len(sys.argv) > 1 else "v17.1"

# get training configurations
with open("model_config/{:s}.json".format(version), "r") as f:
    m = json.loads(f.read())
    board_size = m["board_size"]
    frames = m["frames"]  # keep frames >= 2
    max_time_limit = m["max_time_limit"]
    n_actions = m["n_actions"]
    obstacles = bool(m["obstacles"])
    buffer_size = m["buffer_size"]

# define no of episodes, logging frequency
episodes = 4 * (10**5)
# episodes = 10000
log_frequency = 500
games_eval = 8

# setup the agent
agent = DeepQLearningAgent(
    board_size=board_size,
    frames=frames,
    n_actions=n_actions,
    buffer_size=buffer_size,
    version=version,
    gamma=0.98,
)
agent.print_models()

# setup the epsilon range and decay rate for epsilon
# define rewrad type and update frequency, see utils for more details
epsilon, epsilon_end = 1, 0.01
reward_type = "current"
sample_actions = False
n_games_training = 8 * 16
decay = 0.97


# use only for DeepQLearningAgent
# play some games initially to fill the buffer
# or load from an existing buffer (supervised)

# setup the environment
games = 512
env = SnakeNumpy(
    board_size=board_size,
    frames=frames,
    max_time_limit=max_time_limit,
    games=games,
    frame_mode=True,
    obstacles=obstacles,
    version=version,
)
ct = time.time()
_ = play_game2(
    env,
    agent,
    n_actions,
    n_games=games,
    record=True,
    epsilon=epsilon,
    verbose=True,
    reset_seed=False,
    frame_mode=True,
    total_frames=games * 64,
)
print("Playing {:d} frames took {:.2f}s".format(games * 64, time.time() - ct))

env = SnakeNumpy(
    board_size=board_size,
    frames=frames,
    max_time_limit=max_time_limit,
    games=n_games_training,
    frame_mode=True,
    obstacles=obstacles,
    version=version,
)
env2 = SnakeNumpy(
    board_size=board_size,
    frames=frames,
    max_time_limit=max_time_limit,
    games=games_eval,
    frame_mode=True,
    obstacles=obstacles,
    version=version,
)

# training loop
model_logs = {
    "iteration": [],
    "reward_mean": [],
    "length_mean": [],
    "games": [],
    "loss": [],
    "epsilon": [],
}


def generator():
    while True:
        yield


# for index in tqdm(range(episodes)):
index = 0
for _ in tqdm(generator()):
    index += 1
    # make small changes to the buffer and slowly train
    _, _, _ = play_game2(
        env,
        agent,
        n_actions,
        epsilon=epsilon,
        n_games=n_games_training,
        record=True,
        sample_actions=sample_actions,
        reward_type=reward_type,
        frame_mode=True,
        total_frames=n_games_training,
        stateful=True,
    )
    loss = agent.train_agent(
        batch_size=64, num_games=n_games_training, reward_clip=True
    )

    # check performance every once in a while
    if (index + 1) % log_frequency == 0:
        # keep track of agent rewards_history
        current_rewards, current_lengths, current_games = play_game2(
            env2,
            agent,
            n_actions,
            n_games=games_eval,
            epsilon=-1,
            record=False,
            sample_actions=False,
            frame_mode=True,
            total_frames=-1,
            total_games=games_eval,
        )

        model_logs["iteration"].append(index + 1)
        model_logs["reward_mean"].append(round(int(current_rewards) / current_games, 2))
        # model_logs['reward_dev'].append(round(np.std(current_rewards), 2))
        model_logs["length_mean"].append(round(int(current_lengths) / current_games, 2))
        model_logs["games"].append(current_games)
        model_logs["loss"].append(loss)
        model_logs["epsilon"].append(epsilon)
        pd.DataFrame(model_logs)[
            ["iteration", "reward_mean", "length_mean", "games", "loss", "epsilon"]
        ].to_csv("model_logs/{:s}.csv".format(version), index=False)

    # copy weights to target network and save models
    if (index + 1) % log_frequency == 0:
        agent.update_target_net()
        agent.save_model(file_path="models/{:s}".format(version), iteration=(index + 1))
        # keep some epsilon alive for training
        # if index >= 400 * log_frequency:
        epsilon = max(epsilon * decay, epsilon_end)
