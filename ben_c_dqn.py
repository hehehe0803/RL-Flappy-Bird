"""
Reinforcement learning project - Learning to play Flappy Bird with DQN

**************************************************
*  THIS IS A WORK IN PROGRESS. WILL NOT RUN YET  *
**************************************************

There are many potential parameters to adjust to test training speed
    UPDATE_FREQUENCY: How many steps to take before updating the model
    COPY_FREQUENCY: How many steps to take before copying the model parameters to the target model
    EPSILON_DECAY_RATE: Exponential decay rate of epsilon
    MIN_EPSILON: Smallest possible value for epsilon
    INITIALISER: Weight initialisation to use for the neural networks
    LOSS: Loss function to use for the neural networks
"""

import flappy_bird_gymnasium
import gymnasium

import numpy as np
import tensorflow as tf
from keras.models import Model, Sequential
from keras.layers import Dense, Input
from keras.initializers.initializers_v2 import HeUniform
from keras.losses import Huber
from keras.optimizers import Adam

from collections import deque


RANDOM_SEED = 457
EPISODES = 100
EPSILON_DECAY_RATE = 0.1
MIN_EPSILON = 0.01
UPDATE_FREQUENCY = 4  # Modify to alter model update frequency
COPY_FREQUENCY = 100  # Modify to alter weight copying frequency
MINIBATCH_SIZE = 64

# Maybe prioritise recent experience and drop old experience?


env = gymnasium.make('FlappyBird-v0')
rng = np.random.RandomState(RANDOM_SEED)


def create_model(init=HeUniform(), loss=Huber()) -> Model:
    """Creates a simple neural network"""
    model = Sequential()
    model.add(Dense(16, input_shape=12, activation='relu', kernel_initializer=init))
    model.add(Dense(2, activation='linear', kernel_initializer=init))

    model.compile(loss=loss, optimizer=Adam(), metrics=['accuracy'])

    return model


def train_model():
    pass


model = create_model()
target_model = create_model()

epsilon = 0.05
experience_buffer = deque()

for episode in range(EPISODES):
    obs = env.reset()
    terminated = False

    training_rewards = 0
    update_steps = 0

    while not terminated:
        update_steps += 1

        # Choose action using epsilon greedy exploration
        if rng.random() < epsilon:
            action = env.action_space.sample()
        else:
            state_tensor = tf.convert_to_tensor(obs)
            state_tensor = tf.expand_dims(state_tensor, 0)
            pred = model(state_tensor, training=False)
            action = tf.argmax(pred)

        new_obs, reward, terminated, _, info = env.step(action)
        experience_buffer.append((obs, action, reward, new_obs, terminated))

        # Update main network
        if update_steps % UPDATE_FREQUENCY == 0:
            train_model()

        training_rewards += reward
        obs = new_obs

    # Decay epsilon
    epsilon = np.max(MIN_EPSILON, np.exp(-EPSILON_DECAY_RATE * episode))


env.close()
