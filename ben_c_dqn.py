"""
Attempted implementation of DDQN algorithm with prioritised experience replay
"""
import random
import time
from typing import NamedTuple

# noinspection PyUnresolvedReferences
import flappy_bird_gymnasium
import gymnasium
import numpy as np
from tqdm import tqdm

from keras.models import Model, Sequential
from keras.layers import Dense
from keras.initializers.initializers_v2 import HeUniform
from keras.losses import Huber
from keras.optimizers import Adam

# Constants
RANDOM_SEED = 45
WARMUP_EPISODES = 100
EPISODES = 1000
INIT_EPSILON = 0.1
EPSILON_REDUCTION = 0.00001
EPSILON_DECAY_RATE = 0.98
MIN_EPSILON = 0.005
UPDATE_FREQUENCY = 4
COPY_FREQUENCY = 100
MINIBATCH_SIZE = 32
DISCOUNT_FACTOR = 0.99
BUFFER_SIZE = 50000

# RNG initialisation
random.seed(RANDOM_SEED)
rng = np.random.RandomState(RANDOM_SEED)


class Transition(NamedTuple):
    state: np.ndarray
    action: int
    reward: float
    next_state: np.ndarray
    terminal: bool


class ExperienceReplay:
    def __init__(self, max_len: int):
        self.max_len = max_len
        self.index = 0
        self.full = False

        self.states = np.zeros((max_len, 12))
        self.actions = np.zeros(max_len, dtype=np.uint8)
        self.rewards = np.zeros(max_len)
        self.next_states = np.zeros((max_len, 12))
        self.terminal = np.zeros(max_len, dtype=np.bool_)
        self.priorities = np.zeros(max_len)

    def append(self, transition: Transition):
        """Saves the transition in memory"""
        self.states[self.index] = transition.state
        self.actions[self.index] = transition.action
        self.rewards[self.index] = transition.reward
        self.next_states[self.index] = transition.next_state
        self.terminal[self.index] = transition.terminal
        self.priorities[self.index] = np.max(self.priorities)  # Default priority

        if self.index + 1 == self.max_len:
            self.full = True
            self.index = 0
        else:
            self.index += 1

    def sample(self, size: int) -> tuple:
        if self.full:
            limit = self.max_len
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


def create_model() -> Model:
    """Creates a simple neural network"""
    init = HeUniform(seed=RANDOM_SEED)

    model = Sequential()
    model.add(Dense(16, input_shape=(12,), activation='relu', kernel_initializer=init))
    model.add(Dense(16, activation='relu', kernel_initializer=init))
    model.add(Dense(2, activation='linear', kernel_initializer=init))

    model.compile(loss=Huber(), optimizer=Adam(), metrics=['accuracy'])
    return model


class Agent:
    def __init__(self):
        self.epsilon = INIT_EPSILON
        self.epsilon_reduction = EPSILON_REDUCTION
        self.batch_size = MINIBATCH_SIZE
        self.action_space = [0, 1]

        self.model = create_model()
        self.target_model = create_model()
        self.target_model.set_weights(self.model.get_weights())

        self.experience = ExperienceReplay(BUFFER_SIZE)

    def save_experience(self, state, action, reward, next_state, terminal):
        """Saves the experience to the replay buffer"""
        self.experience.append(Transition(state, action, reward, next_state, terminal))

    def choose_action(self, state) -> int:
        """Chooses an action using an epsilon-greedy policy"""
        if rng.random() < self.epsilon:
            action = random.choice(self.action_space)
        else:
            pred = self.model.predict(np.reshape(state, (1, -1)), verbose=False)
            action = np.argmax(pred)

        # Decrement epsilon
        if self.epsilon > MIN_EPSILON:
            self.epsilon -= self.epsilon_reduction

        return action

    def learn(self):
        state, action, reward, next_state, terminal, sample_indices = self.experience.sample(self.batch_size)

        model_q_predictions = self.model.predict(state, verbose=False)
        target_q_predictions = self.target_model.predict(next_state, verbose=False)

        best_action = np.argmax(model_q_predictions, axis=1)
        target = reward + DISCOUNT_FACTOR * target_q_predictions[np.arange(self.batch_size), best_action] * ~terminal

        target_q = model_q_predictions
        target_q[np.arange(self.batch_size), action] = target

        td_error = np.abs(target - model_q_predictions[np.arange(self.batch_size), action])
        self.experience.update_priorities(td_error, sample_indices)

        self.model.fit(state, target_q, batch_size=self.batch_size, verbose=False)

    def copy_model(self):
        """Copies the model weights to the target model"""
        self.target_model.set_weights(self.model.get_weights())

    def save_model(self, model_name: str = 'flap_dqn.h5'):
        self.model.save(model_name)


def main():
    env = gymnasium.make('FlappyBird-v0')
    agent = Agent()

    steps_taken = 0
    start_time = time.time()

    # Warmup
    print('Doing warmup episodes')
    for episode in range(WARMUP_EPISODES):
        state, _ = env.reset()
        terminal = False

        for i in range(52):
            action = int(i % 18 == 0)  # Flap every 18 frames, keeps the bird close to middle
            state, *_ = env.step(action)

        while not terminal:
            action = rng.choice([0, 1], p=[0.92, 0.08])

            next_state, reward, terminal, _, info = env.step(action)
            steps_taken += 1

            agent.save_experience(state, action, reward, next_state, terminal)

            state = next_state

    # Main training
    print('Doing training')
    for episode in range(EPISODES):
        state, info = env.reset()
        terminal = False
        training_rewards = 0

        # Start the bird near the first pipe
        for i in range(52):
            action = int(i % 18 == 0)  # Flap every 18 frames, keeps the bird close to middle
            state, *_ = env.step(action)

        while not terminal:
            action = agent.choose_action(state)

            next_state, reward, terminal, _, info = env.step(action)
            steps_taken += 1
            agent.save_experience(state, action, reward, next_state, terminal)

            if steps_taken % UPDATE_FREQUENCY == 0 and steps_taken > 3000:
                agent.learn()

            training_rewards += reward
            state = next_state

            # Copy model
            if steps_taken % COPY_FREQUENCY == 0:
                agent.copy_model()

            # The most optimistic line in all of computing
            if info['score'] > 100:
                terminal = True

        print(f'Episode: {episode:<3} Total Rewards: {training_rewards:5.1f} Score: {info["score"]:<3} Time Elapsed: {time.time() - start_time:.2f} s')
        # Save intermediate models
        if episode % 100 == 0:
            agent.save_model(f'model_{episode}.h5')

    env.close()
    agent.save_model()


if __name__ == '__main__':
    main()
