import json
import pickle
import time
from random import random
from typing import Dict, NamedTuple

import cv2
import gym
import keras
import numpy as np
from keras import Input, Model
from keras.initializers.initializers_v2 import HeUniform
from keras.layers import Conv2D, Dense, Flatten
from keras.optimizers import Adam
import tensorflow as tf

RANDOM_SEED = 42
rng = np.random.RandomState(42)

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

class RGBReplay:
    def __init__(self, cfg: Dict):
        self.buffer_size = int(cfg["buffer_size"])
        self.index = 0
        self.full = False

        self.states = np.zeros((self.buffer_size, 144, 256, 1))
        self.actions = np.zeros(self.buffer_size, dtype=np.uint8)
        self.rewards = np.zeros(self.buffer_size, dtype=np.uint8)
        self.next_states = np.zeros((self.buffer_size, 144, 256, 1))
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


class RGBDQN(object):
    def __init__(self, cfg):
        self.gamma = cfg["gamma"]

        self.epsilon_start = cfg["epsilon"]["start"]
        self.epsilon_step = int(cfg["epsilon"]["step"])
        self.epsilon_end = cfg["epsilon"]["end"]
        self.inc = (self.epsilon_end - self.epsilon_start) / self.epsilon_step
        self.bound = max

        self.batch_size = int(cfg["batch_size"])

        self.frame_dim = (144, 256, 1)
        self.action_space = [0, 1]

        self.model = self._make_model(cfg)
        self.target_model = self._make_model(cfg)
        self.update_target_network()

        self.experience = RGBReplay(cfg)

        self.experiment_name = cfg["experiment_name"]
        self.score = list()


    def _make_model(self, cfg: Dict[str, int]) -> keras.Model:
        kernel_init = HeUniform(seed=int(cfg["random_seed"]))

        inputs = Input(shape=self.frame_dim)
        conv1 = Conv2D(
            32,
            kernel_size=8,
            strides=(4, 4),
            padding="valid",
            activation='relu',
            kernel_initializer=kernel_init,
        )(inputs)
        conv2 = Conv2D(
            64,
            kernel_size=4,
            strides=(2, 2),
            padding="valid",
            activation='relu',
            kernel_initializer=kernel_init,
        )(conv1)
        conv3 = Conv2D(
            64,
            kernel_size=3,
            strides=(1, 1),
            padding="valid",
            activation='relu',
            kernel_initializer=kernel_init,
        )(conv2)
        flat = Flatten()(conv3)

        fc1 = Dense(512, activation='relu', kernel_initializer=kernel_init)(flat)
        outputs = Dense(2, kernel_initializer=kernel_init)(fc1)

        model = Model(inputs=inputs, outputs=outputs)
        model.compile(loss="mse", optimizer=Adam(learning_rate=cfg["learning_rate"]))

        return model

    def process_image(self, state: np.ndarray) -> np.ndarray:
        resized_state = cv2.resize(state, (256, 144))
        gray_state = cv2.cvtColor(resized_state, cv2.COLOR_BGR2GRAY)
        normalized_state = gray_state / 255.0
        img_expanded = normalized_state[None ,:, :, None]
        return img_expanded

    def choose_action(self, step_count, state) -> int:
        self.epsilon = self.bound(self.epsilon_start + step_count * self.inc, self.epsilon_end)
        if np.random.rand() < self.epsilon:
            action = np.random.randint(0, len(self.action_space))
        else:
            q_values = self.model(tf.convert_to_tensor(state)).numpy()
            action = np.argmax(q_values)
        return action


    def learn(self):
        states, actions, rewards, next_states, terminals, indices = self.experience.sample(self.batch_size)

        q_values = self.model(tf.convert_to_tensor(states))
        q_target_next_values = self.target_model(next_states)

        updates = rewards + self.gamma * np.max(q_target_next_values, axis=1) * ~terminals

        q_values_target = q_values.numpy()
        q_values_target[np.arange(self.batch_size), actions] = updates

        td_error = np.abs(updates - q_values_target[np.arange(self.batch_size), actions])
        self.experience.update_priorities(td_error, indices)

        loss = self.model.train_on_batch(states, q_values_target)

        return loss

    def update_target_network(self):
        self.target_model.set_weights(self.model.get_weights())

    def train(self, env: gym.Env, num_episodes: int = 20000, save_interval=200):
        start_time = time.time()

        for episode in range(num_episodes):
            state = env.reset()
            state = self.process_image(state)

            terminal = False
            step_count = 1
            training_rewards = 0
            episode_loss = 0

            while not terminal:
                action = self.choose_action(step_count, state)
                next_state, reward, terminal, info = env.step(action)
                next_state = self.process_image(next_state)

                self.experience.append(Transition(state, action, reward, next_state, terminal))

                if self.experience.full:
                    episode_loss += self.learn()
                    step_count += 1

                training_rewards += reward
                state = next_state

            self.update_target_network()

            self.score.append(info["score"])

            # Save intermediate models
            if (episode + 1) % save_interval == 0:
                self.save_model(self.experiment_name + f"{episode + 1:05d}.h5")
                self.save_score(self.experiment_name + f"{episode + 1:05d}.json")
                self.experience.save_experience(self.experiment_name + f"{episode + 1:05d}{episode + 1:05d}.plk")

            print(
                f"Episode: {episode + 1:<4}"
                f"Total Rewards: {training_rewards:<5}"
                f"Score: {info['score']:<4}"
                f"Loss: {episode_loss / step_count:7.2f} "
                f"Epsilon: {self.epsilon:.4f}"
                f"Time Elapsed: {time.time() - start_time:.2f} s"
            )

        return self.score

    def save_model(self, model_name: str = 'flap_dqn.h5'):
        self.model.save(MODEL_WEIGHTS_PATH + model_name)

    def save_score(self, name:str):
        with open(HISTORY_PATH + name, "w") as f:
            json.dump(self.score, f)
        self.score.clear()