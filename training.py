import numpy as np
from tqdm import tqdm
from collections import deque
import pandas as pd
import time
from utils import play_game, play_game2
from game_environment import Snake, SnakeNumpy
from agent import DeepQLearningAgent
import json
import torch
import numpy

# some global variables
version = "v17.7"
print(version)
torch.manual_seed(42)
numpy.random.seed(42)

# get training configurations
with open("model_config/{:s}.json".format(version), "r") as f:
    m = json.loads(f.read())
    board_size = m["board_size"]
    frames = m["frames"]  # keep frames >= 2
    max_time_limit = m["max_time_limit"]
    supervised = bool(m["supervised"])
    n_actions = m["n_actions"]
    obstacles = bool(m["obstacles"])
    buffer_size = m["buffer_size"]

# define no of episodes, logging frequency
# episodes = 2 * (10**5)
episodes = -1
log_frequency = 500
games_eval = 8

# setup the agent
agent = DeepQLearningAgent(
    board_size=board_size,
    frames=frames,
    n_actions=n_actions,
    buffer_size=buffer_size,
    version=version,
    gamma=0.999,
)
# agent = PolicyGradientAgent(board_size=board_size, frames=frames, n_actions=n_actions,
# buffer_size=2000, version=version)
# agent = AdvantageActorCriticAgent(board_size=board_size, frames=frames, n_actions=n_actions,
# buffer_size=10000, version=version)
# agent.print_models()

agent_type = "DeepQLearningAgent"

# setup the epsilon range and decay rate for epsilon
# define rewrad type and update frequency, see utils for more details
epsilon, epsilon_end = 1, 0.01
reward_type = "current"
sample_actions = False
n_games_training = 8 * 16
decay = 0.985
create_checkpoints = True
if supervised:
    # lower the epsilon since some starting policy has already been trained
    epsilon = 0.01
    # load the existing model from a supervised method
    # or some other pretrained model
    # agent.load_model(file_path="models/{:s}".format(version), iteration=173000)
    # agent.set_weights_trainable()

# decay = np.exp(np.log((epsilon_end/epsilon))/episodes)

# play some games initially to fill the buffer
# or load from an existing buffer (supervised)
if supervised:
    try:
        agent.load_buffer(file_path="models/{:s}".format(version), iteration=1)
        print("Buffer loaded")
    except FileNotFoundError:
        pass
else:
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
        model_logs["reward_mean"].append(round(int(current_rewards) / current_games, 2))
        if create_checkpoints and reward_mean - best_reward > 0:
            best_reward = reward_mean
            agent.save_model(
                file_path="models/{:s}".format(version), iteration=(index + 1)
            )
            agent.save_buffer(
                file_path="models/{:s}".format(version), iteration=(index + 1)
            )
        model_logs["length_mean"].append(round(int(current_lengths) / current_games, 2))
        model_logs["games"].append(current_games)
        model_logs["loss"].append(loss)
        pd.DataFrame(model_logs)[
            ["iteration", "reward_mean", "length_mean", "games", "loss"]
        ].to_csv("model_logs/{:s}.csv".format(version), index=False)

    # copy weights to target network and save models
    if (index + 1) % log_frequency == 0:
        agent.update_target_net()
        # agent.save_model(file_path='models/{:s}'.format(version), iteration=(index+1))
        # keep some epsilon alive for training
        epsilon = max(epsilon * decay, epsilon_end)
