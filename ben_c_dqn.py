"""
Reinforcement learning project - Learning to play Flappy Bird with DQN

**************************************************
*                DOESN'T WORK YET                *
**************************************************

There are many potential parameters to adjust to test training speed
    UPDATE_FREQUENCY: How many steps to take before updating the model
    COPY_FREQUENCY: How many steps to take before copying the model parameters to the target model
    EPSILON_DECAY_RATE: Exponential decay rate of epsilon
    MIN_EPSILON: Smallest possible value for epsilon
    INITIALISER: Weight initialisation to use for the neural networks
    LOSS: Loss function to use for the neural networks
"""
import random
import time
from collections import deque, namedtuple

# noinspection PyUnresolvedReferences
import flappy_bird_gymnasium
import gymnasium
import numpy as np
from tqdm import tqdm

import keras
from keras.models import Model, Sequential
from keras.layers import Dense
from keras.initializers.initializers_v2 import HeUniform
from keras.losses import Huber
from keras.optimizers import Adam


RANDOM_SEED = 457
EPISODES = 300
EPSILON_DECAY_RATE = 0.98
MIN_EPSILON = 0.01
UPDATE_FREQUENCY = 4  # Modify to alter model update frequency
COPY_FREQUENCY = 100  # Modify to alter weight copying frequency
MINIBATCH_SIZE = 64
DISCOUNT_FACTOR = 0.99

Transition = namedtuple('Transition', ['state', 'action', 'reward', 'next_state', 'terminal'])

env = gymnasium.make('FlappyBird-v0')
rng = np.random.RandomState(RANDOM_SEED)


def create_model() -> Model:
    """Creates a simple neural network"""
    init = HeUniform(seed=RANDOM_SEED)

    model = Sequential()
    model.add(Dense(16, input_shape=(12,), activation='relu', kernel_initializer=init))
    model.add(Dense(2, activation='linear', kernel_initializer=init))

    model.compile(loss=Huber(), optimizer=Adam(), metrics=['accuracy'])

    return model


def train_model():
    """Applies Bellman equations to update"""
    # Don't update if we have insufficient experience
    if len(experience_buffer) < MINIBATCH_SIZE:
        return

    # Sample a minibatch
    minibatch = random.sample(experience_buffer, MINIBATCH_SIZE)

    # Convert to numpy arrays
    state_array = np.array([t.state for t in minibatch])
    action_array = np.array([t.action for t in minibatch], dtype=np.int_)
    reward_array = np.array([t.reward for t in minibatch])
    next_state_array = np.array([t.next_state for t in minibatch])
    terminal_array = np.array([t.terminal for t in minibatch])

    # Make q-value predictions using models
    current_q_array = model.predict(state_array, verbose=False)
    next_q_array = target_model.predict(next_state_array, verbose=False)

    # Vectorised equation
    max_next_q_values = np.max(next_q_array, axis=1)
    max_next_q_values[terminal_array] = 0  # Equals zero when terminal
    new_q = reward_array + DISCOUNT_FACTOR * max_next_q_values

    current_q_array[np.arange(MINIBATCH_SIZE), action_array] = new_q

    x = state_array
    y = current_q_array

    model.fit(x, y, batch_size=MINIBATCH_SIZE, verbose=False, shuffle=True)


experience_buffer = deque(maxlen=10 ** 6)

model = create_model()
target_model = keras.models.clone_model(model)
epsilon = 0.5
episode_rewards = []

for episode in range(EPISODES):
    state, info = env.reset()
    terminal = False

    training_rewards = 0
    update_steps = 0

    while not terminal:
        update_steps += 1

        # Choose action using epsilon greedy exploration
        if rng.random() < epsilon:
            action = env.action_space.sample()
        else:
            pred = model.predict(np.reshape(state, (1, -1)), verbose=False)
            action = np.argmax(pred)

        next_state, reward, terminal, _, info = env.step(action)
        experience_buffer.append(Transition(state, action, reward, next_state, terminal))

        # Update main network
        if update_steps % UPDATE_FREQUENCY == 0:
            train_model()

        training_rewards += reward
        state = next_state

        # Copy model
        if update_steps % COPY_FREQUENCY == 0:
            target_model.set_weights(model.get_weights())

        if info['score'] > 100:
            terminal = True

        if terminal:
            print(f'Episode: {episode}\tTotal Rewards: {training_rewards:.2f}')

    # Decay epsilon
    epsilon = max(MIN_EPSILON, epsilon * EPSILON_DECAY_RATE)
    episode_rewards.append(training_rewards)


state, info = env.reset()
terminal = False

while not terminal:
    pred = model.predict(np.reshape(state, (1, -1)), verbose=False)
    action = np.argmax(pred)

    new_state, reward, terminal, _, info = env.step(action)

    env.render()
    time.sleep(1 / 30)


env.close()
model.save('flap_dqn.h5')
