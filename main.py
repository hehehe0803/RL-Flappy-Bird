import flappy_bird_gym
from ben_c_dqn import Agent

import os
os.environ['CUDA_VISIBLE_DEVICES'] = '-1' # Run cuda on cpu

MODEL_WEIGHTS_PATH = "model_weights/"
HISTORY_PATH = "history/"


if __name__ == '__main__':
    n_episodes = 100

    env = flappy_bird_gym.make("FlappyBird-v0")

    n_states = env.observation_space.shape[0]
    n_actions = env.action_space.n
    agent = Agent(
        alpha=1e-3,
        gamma=0.95,
        epsilon=1e-4,
        epsilon_reduction=0,
        epsilon_min=1e-4,
        batch_size=32,
        buffer_size=25000
    )
    agent.load_model(MODEL_WEIGHTS_PATH + "model_7400.h5")
    agent.load_experience("memory/ddqlr_02_7400.plk")
    agent.train(env=env, n_episodes=(20000 - 7400), save_interval=200)

    env.close()

    agent.save_model(MODEL_WEIGHTS_PATH + "ddqlr_02_final.h5")
