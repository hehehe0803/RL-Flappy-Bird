import argparse
import json
from typing import Dict, Union

import flappy_bird_gym
import models

import os
os.environ['CUDA_VISIBLE_DEVICES'] = '0' # Run on cpu

MODEL_WEIGHTS_PATH = "model_weights/"
HISTORY_PATH = "history/"
CONFIG_PATH = "configs/"

def get_args() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        """Input config file"""
    )
    parser.add_argument(
        "--config_file",
        type=str,
        default="dqn_03.json",
        help="Input config file"
    )

    args = parser.parse_args()
    return args

def get_config(args) -> Dict[str, Union[str, int, Dict]]:
    with open(CONFIG_PATH + args.config_file, 'r') as config_file:
        cfg = json.load(config_file)

    return cfg
if __name__ == '__main__':
    cfg = get_config(get_args())

    env = flappy_bird_gym.make(cfg["env_name"])
    agent = getattr(models, cfg["agent_name"])(cfg)
    agent.train(env=env)

    env.close()

    agent.save_model(MODEL_WEIGHTS_PATH + "dqn_final.h5")
