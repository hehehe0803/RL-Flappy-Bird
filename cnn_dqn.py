"""
Implementation of DQN and DDQN algorithms
"""
import random
import time
from typing import NamedTuple
import os
import json

# noinspection PyUnresolvedReferences
import flappy_bird_gymnasium
import gymnasium
import numpy as np
import cv2 as cv

from keras.models import Model
from keras.layers import Dense, Input, Conv2D, MaxPooling2D, Flatten
from keras.losses import Huber
from keras.optimizers import RMSprop

from powermanagement import long_running

# Constants
RANDOM_SEED = 45
ACTION_LEN = 2
WARMUP_EPISODES = 100
EPISODES = 10000
INIT_EPSILON = 0.15
EPSILON_REDUCTION = 1e-6
MIN_EPSILON = 0.01
UPDATE_FREQUENCY = 4
TAU = 0.01
COPY_FREQUENCY = 100
MINIBATCH_SIZE = 32
DISCOUNT_FACTOR = 0.99
BUFFER_SIZE = 50000
RESCALE_SIZE = 96
OPTIMISER_LEARNING_RATE = 1e-5
OPTIMISER_MOMENTUM = 0.95

X_DIM = 288 // 4
Y_DIM = 512 // 4

# RNG initialisation
random.seed(RANDOM_SEED)
rng = np.random.RandomState(RANDOM_SEED)


class EnvWrapper:
    def __init__(self, record=False):
        self.env = gymnasium.make('FlappyBird-rgb-v0')
        self.state_stack = np.zeros((X_DIM, Y_DIM, 4), np.bool_)

    def reset(self) -> (np.ndarray, dict):
        state, info = self.env.reset()

        state = self._transform_state(state).astype(np.bool_)
        self.state_stack = np.tile(np.expand_dims(state, axis=-1), 4)

        return self.state_stack, info

    def step(self, action: int) -> (np.ndarray, float, bool, dict):
        state, reward, terminal, _, info = self.env.step(action)

        state = self._transform_state(state)
        self.state_stack = np.roll(self.state_stack, -1)
        self.state_stack[:, :, -1] = state

        return self.state_stack, reward, terminal, info

    def render(self):
        self.env.render()

    def close(self):
        self.env.close()

    @staticmethod
    def _transform_state(state: np.ndarray) -> np.ndarray:
        state = cv.resize(state, (0, 0), fx=0.25, fy=0.25)
        state = cv.cvtColor(state, cv.COLOR_RGB2GRAY)
        _, state = cv.threshold(state, 0, 128, cv.THRESH_BINARY)

        return state


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

        self.states = np.zeros((max_len, X_DIM, Y_DIM, 4), np.bool_)
        self.actions = np.zeros(max_len, dtype=np.uint8)
        self.rewards = np.zeros(max_len)
        self.next_states = np.zeros((max_len, X_DIM, Y_DIM, 4), np.bool_)
        self.terminal = np.zeros(max_len, dtype=np.bool_)

    def append(self, transition: Transition):
        """Saves the transition in memory"""
        self.states[self.index] = transition.state
        self.actions[self.index] = transition.action
        self.rewards[self.index] = transition.reward
        self.next_states[self.index] = transition.next_state  # This is a crime against optimisation
        self.terminal[self.index] = transition.terminal

        if self.index + 1 == self.max_len:
            self.full = True
            self.index = 0
        else:
            self.index += 1

    def sample(self, size: int) -> tuple:
        """Random sample"""
        if self.full:
            limit = self.max_len
        else:
            limit = self.index - 1

        sample_indices = rng.choice(np.arange(limit), size=size, replace=False)

        return (
            self.states[sample_indices],
            self.actions[sample_indices],
            self.rewards[sample_indices],
            self.next_states[sample_indices],
            self.terminal[sample_indices],
            None
        )


class CombinedExperienceReplay(ExperienceReplay):
    def __init__(self, max_len: int):
        super().__init__(max_len)

    def sample(self, size: int) -> tuple:
        """Samples with Combined Experience Replay"""
        if self.full:
            limit = self.max_len
        else:
            limit = self.index - 1

        sample_indices = rng.choice(np.arange(limit), size=size, replace=False)
        sample_indices[0] = (self.index - 1) % self.max_len

        return (
            self.states[sample_indices],
            self.actions[sample_indices],
            self.rewards[sample_indices],
            self.next_states[sample_indices],
            self.terminal[sample_indices],
            None
        )

    def update_priorities(self, sample_indices: np.ndarray, priorities: np.ndarray):
        raise NotImplementedError


class PrioritisedExperienceReplay(ExperienceReplay):
    def __init__(self, max_len: int):
        super().__init__(max_len)

        self.priorities = np.zeros(max_len)

    def append(self, transition: Transition):
        self.priorities[self.index] = np.max(self.priorities)

        super().append(transition)

    def sample(self, size: int) -> tuple:
        """Sample using prioritised experience replay"""
        if self.full:
            limit = self.max_len
        else:
            limit = self.index - 1

        weights = (self.priorities[:limit] + 1e-6) / (np.sum(self.priorities[:limit]) + 1e-6 * limit)

        sample_indices = rng.choice(np.arange(limit), size=size, replace=False, p=weights)

        return (
            self.states[sample_indices],
            self.actions[sample_indices],
            self.rewards[sample_indices],
            self.next_states[sample_indices],
            self.terminal[sample_indices],
            sample_indices
        )

    def update_priorities(self, sample_indices: np.ndarray, priorities: np.ndarray):
        """Saves the recalculated priorities"""
        self.priorities[sample_indices] = priorities


class Agent:
    def __init__(
            self,
            max_epsilon: float,
            min_epsilon: float,
            combined_experience_replay: bool = False,
            prioritised_experience_replay: bool = False
    ):
        assert not (combined_experience_replay and prioritised_experience_replay)

        self.epsilon = max_epsilon
        self.min_epsilon = min_epsilon
        self.epsilon_reduction = EPSILON_REDUCTION
        self.batch_size = MINIBATCH_SIZE
        self.action_space = [0, 1]

        self.env = EnvWrapper()
        self.model = self._create_model(ACTION_LEN)

        if combined_experience_replay:
            self.experience = CombinedExperienceReplay(BUFFER_SIZE)
        elif prioritised_experience_replay:
            self.experience = PrioritisedExperienceReplay(BUFFER_SIZE)
        else:
            self.experience = ExperienceReplay(BUFFER_SIZE)

        self.steps_taken = 0
        self.name = 'agent'

    def train(self, train_episodes: int, warmup_episodes: int = 0, skip_frames: int = 0):
        """Trains the agent"""
        for _ in range(warmup_episodes):
            state, info = self.env.reset()
            terminal = False

            for i in range(skip_frames):
                state, *_ = self.env.step(int(i % 18 == 0))

            while not terminal:
                action = rng.choice([0, 1], p=[0.92, 0.08])

                next_state, reward, terminal, info = self.env.step(action)
                self.steps_taken += 1

                self.save_experience(state, action, reward, next_state, terminal)

                state = next_state

        start_time = time.time()
        rewards_list = []
        for episode in range(train_episodes):
            state, info = self.env.reset()
            terminal = False
            training_rewards = 0

            for i in range(skip_frames):
                state, *_ = self.env.step(int(i % 18 == 0))

            while not terminal:
                action = self.choose_action(state)

                next_state, reward, terminal, info = self.env.step(action)
                self.steps_taken += 1

                self.save_experience(state, action, reward, next_state, terminal)

                if self.steps_taken % UPDATE_FREQUENCY == 0 and self.steps_taken > MINIBATCH_SIZE * 32:
                    self.learn()

                training_rewards += reward
                state = next_state

                # The most optimistic line in all of computing
                if info['score'] > 100:
                    terminal = True

            rewards_list.append(training_rewards)
            print(
                f'Episode: {episode:<5} '
                f'Total Rewards: {training_rewards:5.1f} '
                f'Score: {info["score"]:<3} '
                f'Time Elapsed: {time.time() - start_time:.2f} s '
                f'Eps: {self.epsilon:.4f}')

            # Save intermediate models
            if episode % 200 == 0:
                self.save_model(f'{self.name}_{episode}.h5')
                self.save_history(rewards_list)

        self.save_model(f'{self.name}_{episode}.h5')
        self.save_history(rewards_list)

    def save_experience(self, state, action, reward, next_state, terminal):
        """Saves the experience to the replay buffer"""
        self.experience.append(Transition(state, action, reward, next_state, terminal))

    def choose_action(self, state) -> int:
        """Chooses an action using an epsilon-greedy policy"""
        if rng.random() < self.epsilon:
            action = random.choice(self.action_space)
        else:
            pred = self.model(np.expand_dims(state, axis=0), training=False)
            action = int(np.argmax(pred))

        # Decrement epsilon
        if self.epsilon > self.min_epsilon:
            self.epsilon = max(self.min_epsilon, self.epsilon - self.epsilon_reduction)

        return action

    def learn(self):
        """Sample from experience replay to model update weights"""
        raise NotImplementedError

    def save_model(self, model_name: str):
        """Save the model"""
        if not os.path.isdir('models'):
            os.mkdir('models')

        self.model.save(f'models/{model_name}')

    def save_history(self, rewards_list: list):
        """Saves the list of rewards so far to a json file"""
        if not os.path.isdir('history'):
            os.mkdir('history')

        with open(f'history/{self.name}_history.json', 'w') as json_file:
            json.dump(rewards_list, json_file)

    @staticmethod
    def _create_model(action_size: int) -> Model:
        """Creates a simple CNN"""
        inputs = Input((X_DIM, Y_DIM, 4))
        x = Conv2D(32, kernel_size=(7, 7), strides=4, activation='relu', padding='same')(inputs)
        x = MaxPooling2D(pool_size=(2, 2))(x)
        x = Conv2D(64, kernel_size=(5, 5), strides=2, activation='relu', padding='same')(x)
        x = MaxPooling2D(pool_size=(2, 2))(x)
        x = Conv2D(64, kernel_size=(3, 3), strides=1, activation='relu', padding='same')(x)
        x = MaxPooling2D(pool_size=(2, 2))(x)

        x = Flatten()(x)
        x = Dense(128, activation='relu')(x)
        output = Dense(action_size, activation='linear')(x)

        model = Model(inputs=inputs, outputs=output)
        model.compile(
            optimizer=RMSprop(OPTIMISER_LEARNING_RATE, momentum=OPTIMISER_MOMENTUM),
            loss=Huber(),
            metrics=['accuracy']
        )
        return model


class DQNAgent(Agent):
    def __init__(
            self,
            max_epsilon: float,
            min_epsilon: float,
            combined_experience_replay: bool = False,
            prioritised_experience_replay: bool = False
    ):
        super().__init__(max_epsilon, min_epsilon, combined_experience_replay, prioritised_experience_replay)

        name = 'dqn'
        if self.experience.__class__.__name__ == 'CombinedExperienceReplay':
            name += '_cer'
        elif self.experience.__class__.__name__ == 'PrioritisedExperienceReplay':
            name += '_per'
        self.name = name

    def learn(self):
        state, action, reward, next_state, terminal, sample_indices = self.experience.sample(self.batch_size)

        model_q_predictions = self.model(state, training=False).numpy()
        target_q_predictions = self.model(next_state, training=False).numpy()

        batch_index = np.arange(self.batch_size, dtype=np.int_)
        best_action = np.argmax(target_q_predictions, axis=1)
        target = reward + DISCOUNT_FACTOR * target_q_predictions[batch_index, best_action] * ~terminal

        target_q = model_q_predictions.copy()
        target_q[batch_index, action] = target

        if self.experience.__class__.__name__ == 'PrioritisedExperienceReplay':
            td_error = np.abs(target - model_q_predictions[batch_index, action])
            self.experience.update_priorities(sample_indices, td_error)

        self.model.fit(state, target_q, batch_size=self.batch_size, verbose=False)


class DDQNAgent(Agent):
    def __init__(
            self,
            max_epsilon: float,
            min_epsilon: float,
            combined_experience_replay: bool = False,
            prioritised_experience_replay: bool = False,
            soft_updates: bool = False
    ):
        super().__init__(max_epsilon, min_epsilon, combined_experience_replay, prioritised_experience_replay)

        self.target_model = self._create_model(ACTION_LEN)
        self.target_model.set_weights(self.model.get_weights())

        self.use_soft_updates = soft_updates

        name = 'ddqn'
        if self.experience.__class__.__name__ == 'CombinedExperienceReplay':
            name += '_cer'
        elif self.experience.__class__.__name__ == 'PrioritisedExperienceReplay':
            name += '_per'
        if soft_updates:
            name += '_su'
        self.name = name

    def learn(self):
        state, action, reward, next_state, terminal, sample_indices = self.experience.sample(self.batch_size)

        model_q_predictions = self.model(state, training=False).numpy()
        target_q_predictions = self.target_model(next_state, training=False).numpy()

        batch_index = np.arange(self.batch_size, dtype=np.int_)
        best_action = np.argmax(self.model(next_state, training=False).numpy(), axis=1)
        target = reward + DISCOUNT_FACTOR * target_q_predictions[batch_index, best_action] * ~terminal

        target_q = model_q_predictions.copy()
        target_q[batch_index, action] = target

        if self.experience.__class__.__name__ == 'PrioritisedExperienceReplay':
            td_error = np.abs(target - model_q_predictions[batch_index, action])
            self.experience.update_priorities(sample_indices, td_error)

        self.model.fit(state, target_q, batch_size=self.batch_size, verbose=False)
        self.copy_model()

    def copy_model(self):
        """Copies the model weights to the target model using soft backups"""
        if self.use_soft_updates:
            new_target_model_weights = []
            for target_model_weight, model_weight in zip(self.target_model.get_weights(), self.model.get_weights()):
                new_target_model_weights.append(TAU * model_weight + (1 - TAU) * target_model_weight)

            self.target_model.set_weights(new_target_model_weights)
        elif self.steps_taken % COPY_FREQUENCY:
            self.target_model.set_weights(self.model.get_weights())


@long_running
def main():
    agent = DDQNAgent(INIT_EPSILON, MIN_EPSILON, prioritised_experience_replay=True, soft_updates=True)
    agent.train(EPISODES, WARMUP_EPISODES, skip_frames=53)


if __name__ == '__main__':
    main()
