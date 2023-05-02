import json
import pickle
import time
import shutil
from random import random
from typing import Dict, NamedTuple

import cv2
import gym
import keras
import numpy as np
from keras import Input, Model
from keras.initializers import HeUniform
from keras.layers import Dense
from keras.optimizers import Adam
import tensorflow as tf

RANDOM_SEED = 42
rng = np.random.RandomState(RANDOM_SEED)

MODEL_WEIGHTS_PATH = "model_weights/"
HISTORY_PATH = "history/"
MEMORY_PATH = "memory/"
MODEL_INIT_PATH = "model_init/"

class Transition(NamedTuple):
    state: np.ndarray
    action: int
    reward: int
    next_state: np.ndarray
    terminal: bool

class Replay:
    def __init__(self, cfg: Dict):
        self.buffer_size = int(cfg["buffer_size"])
        self.index = 0
        self.full = False

        self.states = np.zeros((self.buffer_size, 2))
        self.actions = np.zeros(self.buffer_size, dtype=np.uint8)
        self.rewards = np.zeros(self.buffer_size, dtype=np.uint8)
        self.next_states = np.zeros((self.buffer_size, 2))
        self.terminal = np.zeros(self.buffer_size, dtype=np.bool_)
        self.priorities = np.zeros(self.buffer_size)

    def append(self, transition: Transition):
        """
        Saves the transition in memory
        remove an element at random when the buffer is full,
         and then append the new element
        """

        if self.full:
            # Remove an element at random
            self.index = rng.choice(self.buffer_size)

        self.states[self.index] = transition.state
        self.actions[self.index] = transition.action
        self.rewards[self.index] = transition.reward
        self.next_states[self.index] = transition.next_state
        self.terminal[self.index] = transition.terminal
        self.priorities[self.index] = np.max(self.priorities)  # Default priority

        if self.index + 1 == self.buffer_size:
            self.full = True
        else:
            self.index += 1

    def sample(self, size: int) -> tuple:
        if self.full:
            limit = self.buffer_size
        else:
            limit = self.index - 1

        if np.max(self.priorities) == 0:
            weights = None
        else:
            self.priorities += 0.001
            weights = self.priorities[:limit] / np.sum(self.priorities[:limit])

        sample_indices = rng.choice(np.arange(limit), size=size, replace=False, p=weights)

        return (
            self.states[sample_indices],
            self.actions[sample_indices],
            self.rewards[sample_indices],
            self.next_states[sample_indices],
            self.terminal[sample_indices],
            sample_indices
        )

    def update_priorities(self, td_errors: np.ndarray, indices: np.ndarray):
        self.priorities[indices] = td_errors

    def save_experience(self, name: str):
        with open(MEMORY_PATH + name, 'wb') as f:
            pickle.dump(self, f)


class DQN(object):
    def __init__(self, cfg):
        self.reward_scaling = cfg["reward_scaling"]
        self.update_frequency = cfg["experience_update_frequency"]
        self.experience = Replay(cfg)

        self.gamma = cfg["gamma"]

        self.epsilon = cfg["epsilon"]["start"]
        self.epsilon_step = int(cfg["epsilon"]["step"])
        self.epsilon_end = cfg["epsilon"]["end"]
        self.inc = (self.epsilon_end - self.epsilon) / self.epsilon_step
        self.bound = max

        self.batch_size = int(cfg["batch_size"])

        self.feature_dim = (2,)
        self.action_space = [0, 1]

        self.step_count = 0
        self.network_update_frequency = cfg["network_update_frequency"]
        self.model = self._make_model(cfg)
        self.target_model = self._make_model(cfg)
        self.update_target_network()

        self.experiment_name = cfg["experiment_name"]
        self.score = list()

    def _make_model(self, cfg: Dict[str, int]) -> keras.Model:
        kernel_init = HeUniform(seed=int(cfg["random_seed"]))

        inputs = Input(shape=self.feature_dim)
        x = Dense(32, activation='relu', kernel_initializer=kernel_init)(inputs)
        x = Dense(32, activation='relu', kernel_initializer=kernel_init)(x)
        outputs = Dense(2, activation='linear', kernel_initializer=kernel_init)(x)

        model = Model(inputs=inputs, outputs=outputs)
        model.compile(loss="mse", optimizer=Adam(learning_rate=cfg["learning_rate"]))

        return model

    def choose_action(self, state) -> int:
        if np.random.rand() < self.epsilon:
            action = np.random.choice(2, p=[0.9,0.1])
        else:
            q_values = self.model(np.reshape(state, [1, 2])).numpy()
            action = np.argmax(q_values)
        return action


    def custom_reward(self, state, reward):
        x_pos = state[0]
        y_dif = state[1]

        x_pos_scaling = 1 / (1 + x_pos)

        reward -= self.reward_scaling * x_pos_scaling * abs(y_dif)
        reward = np.round(reward, 4)

        return reward

    def learn(self):
        states, actions, rewards, next_states, terminals, indices = self.experience.sample(self.batch_size)

        q_values = self.model(states).numpy()
        q_target_next_values = self.target_model(next_states).numpy()

        best_action = np.argmax(q_values, axis=1)
        updates = rewards + self.gamma * q_target_next_values[np.arange(self.batch_size), best_action] * ~terminals

        q_values_target = q_values
        q_values_target[np.arange(self.batch_size), actions] = updates

        td_error = np.abs(updates - q_values_target[np.arange(self.batch_size), actions])
        self.experience.update_priorities(td_error, indices)

        history = self.model.fit(states, q_values_target, batch_size=self.batch_size, verbose=False)

        if self.epsilon > self.epsilon_end:
            self.epsilon = self.bound(self.epsilon + self.inc, self.epsilon_end)

        return history.history["loss"][0]

    def update_target_network(self):
        if self.step_count % self.network_update_frequency == 0:
            self.target_model.set_weights(self.model.get_weights())

    def train(self, env: gym.Env, num_episodes: int = 20000, save_interval=250):
        start_time = time.time()

        for episode in range(num_episodes):
            state = env.reset()

            terminal = False
            self.step_count = 1
            training_rewards = 0
            episode_loss = 0

            while not terminal:
                action = self.choose_action(state)
                next_state, reward, terminal, info = env.step(action)
                reward = self.custom_reward(next_state, reward)

                self.step_count += 1
                self.experience.append(Transition(state, action, reward, next_state, terminal))

                if self.experience.full and self.step_count % self.update_frequency == 0:
                    episode_loss += self.learn()


                training_rewards += reward
                state = next_state

                self.update_target_network()

            self.score.append(info["score"])

            # Save intermediate models
            if (episode + 1) % save_interval == 0:
                self.save_model(self.experiment_name + f"{episode + 1:05d}.h5")
                self.save_score(self.experiment_name + f"{episode + 1:05d}.json")
                self.experience.save_experience(self.experiment_name + "experience.plk")
                self.save_epsilon(self.experiment_name + "epsilon.json")

            print(
                f"Episode: {episode + 1:<4} "
                f"Total Rewards: {training_rewards:9.4f} "
                f"Score: {info['score']:<4} "
                f"Loss: {episode_loss / self.step_count:7.4f} "
                f"Epsilon: {self.epsilon:.4f} "
                f"Time Elapsed: {time.time() - start_time:.2f} s"
            )

        return self.score

    def save_model(self, model_name: str = 'flap_dqn.h5'):
        self.model.save("temp.h5")
        shutil.move("temp.h5", MODEL_WEIGHTS_PATH + model_name)

    def load_model(self, model_name: str):
        self.model.load_weights(model_name)
        self.target_model.load_weights(model_name)

    def save_score(self, name:str):
        with open(HISTORY_PATH + name, "w") as f:
            json.dump(self.score, f)
        self.score.clear()

    def save_epsilon(self, name: str):
        with open("model_init/" + name, "w") as f:
            json.dump(
                {
                    "epsilon": self.epsilon
                },
                f
            )