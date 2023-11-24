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
from agent import DeepQLearningAgent, BreadthFirstSearchAgent
import json
import sys


def train(
    version="v17.1",
    episodes=3 * (10**5),
    log_frequency=500,
    games_eval=8,
    gamma=0.98,
    load_buffer=False,
    update_buffer=True,
    load_pretrained=False,
    pretrained_iteration=300000,
    buffer_iteration=1,
    epsilon_start=1.0,
    epsilon_end=0.1,
    decay=0.989,
    log_label=None,
):
    # get training configurations
    with open("model_config/{:s}.json".format(version), "r") as f:
        m = json.loads(f.read())
        board_size = m["board_size"]
        frames = m["frames"]  # keep frames >= 2
        max_time_limit = m["max_time_limit"]
        n_actions = m["n_actions"]
        obstacles = bool(m["obstacles"])
        buffer_size = m["buffer_size"]

    # setup the agent
    agent = DeepQLearningAgent(
        board_size=board_size,
        frames=frames,
        n_actions=n_actions,
        buffer_size=buffer_size,
        version=version,
        gamma=gamma,
    )
    agent.print_models()

    if load_pretrained:
        agent.load_model("models/" + version, iteration=pretrained_iteration)

    # setup the epsilon range and decay rate for epsilon
    # define rewrad type and update frequency, see utils for more details
    epsilon = epsilon_start
    reward_type = "current"
    sample_actions = False
    n_games_training = 8 * 16
    # buffer_path = "models/v17.1"

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
    if load_buffer:
        agent.load_buffer("models/" + version, iteration=buffer_iteration)
    else:
        ct = time.time()
        if update_buffer:
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
                total_frames=games * 64 * 2,
            )
            print(
                "Playing {:d} frames took {:.2f}s".format(
                    games * 64 * 2, time.time() - ct
                )
            )

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

    best_reward = float("-inf")
    index = 0
    for _ in tqdm(range(episodes) if (episodes >= 0) else generator()):
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
            reward_mean = round(int(current_rewards) / current_games, 2)
            model_logs["reward_mean"].append(reward_mean)
            if reward_mean - best_reward > 5:
                best_reward = reward_mean
                agent.save_model(
                    file_path="models/{:s}".format(version), iteration=(index + 1)
                )
                agent.save_buffer(
                    file_path="models/{:s}".format(version), iteration=(index + 1)
                )
                print(f"new highscore at {reward_mean}")
            model_logs["length_mean"].append(
                round(int(current_lengths) / current_games, 2)
            )
            model_logs["games"].append(current_games)
            model_logs["loss"].append(loss)
            model_logs["epsilon"].append(epsilon)
            pd.DataFrame(model_logs)[
                ["iteration", "reward_mean", "length_mean", "games", "loss", "epsilon"]
            ].to_csv(
                "model_logs/{:s}{:s}.csv".format(
                    version, f"_{log_label}" if log_label else ""
                ),
                index=False,
            )

            agent.update_target_net()
            # keep some epsilon alive for training
            # if index >= 400 * log_frequency:
            epsilon = max(epsilon * decay, epsilon_end)


train(
    version="v17.1",
    episodes=3 * (10**5),
    log_frequency=500,
    games_eval=8,
    gamma=0.98,
    load_buffer=False,
    update_buffer=True,
    load_pretrained=False,
    pretrained_iteration=300000,
    buffer_iteration=1,
    epsilon_start=1.0,
    epsilon_end=0.1,
    decay=0.989,
    log_label="test",
)
