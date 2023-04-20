import time
import flappy_bird_gymnasium
import gymnasium
from models.DQL import DQLAgent, DoubleDQLReplayAgent

import os
os.environ['CUDA_VISIBLE_DEVICES'] = '-1' # Run cuda on cpu

MODEL_WEIGHTS_PATH = "model_weights/"
REWARDS_PATH = "rewards_history/"


if __name__ == '__main__':
    n_episodes = 100

    env = gymnasium.make("FlappyBird-v0")

    n_states = env.observation_space.shape[0]
    n_actions = env.action_space.n
    ddqlr_agent = DoubleDQLReplayAgent(
        gamma=0.9,
        epsilon=0.3,
        epsilon_decay=0.9995,
        gap_penalty_w=0.1
    )
    ddqlr_agent.train(env=env, n_episodes=2000, save_interval=100)
    ddqlr_agent.save_model(MODEL_WEIGHTS_PATH + "DDQLReplay_final.h5")
    ddqlr_agent.save_reward(REWARDS_PATH + "DDQLReplay_final.json")
