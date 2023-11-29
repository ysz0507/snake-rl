import os
import time
from utils import play_game2
from game_environment import SnakeNumpy
from agent import BreadthFirstSearchAgent
import json


def generate_buffer(version, buffer_size=80000, n_games_training=100):
    print(f"Generate buffer for {version}")

    # get training configurations
    with open("model_config/{:s}.json".format(version), "r") as f:
        m = json.loads(f.read())
        board_size = m["board_size"]
        frames = m["frames"]  # keep frames >= 2
        max_time_limit = m["max_time_limit"]
        n_actions = m["n_actions"]
        obstacles = bool(m["obstacles"])

    env = SnakeNumpy(
        board_size=board_size,
        frames=frames,
        games=n_games_training,
        max_time_limit=max_time_limit,
        obstacles=obstacles,
        version=version,
    )
    n_actions = env.get_num_actions()

    agent = BreadthFirstSearchAgent(
        board_size=board_size,
        frames=frames,
        n_actions=n_actions,
        buffer_size=buffer_size,
        version=version,
    )
    curr_time = time.time()
    _, _, _ = play_game2(
        env,
        agent,
        n_actions,
        epsilon=-1,
        n_games=n_games_training,
        record=True,
        reward_type="current",
        frame_mode=True,
        total_frames=buffer_size,
        stateful=True,
        showProgressBar=True,
    )
    print(
        "Buffer size {:d} filled in {:.2f}s".format(
            agent.get_buffer_size(), time.time() - curr_time
        )
    )
    file_path = "models/{:s}".format(version)
    if not os.path.exists(file_path):
        os.mkdir(file_path)
    agent.save_buffer(file_path=file_path, iteration=1)
