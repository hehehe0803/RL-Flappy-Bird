"""
Attempted implementation of DDQN algorithm with prioritised experience replay
"""
import json
import pickle
import random
import time
from typing import NamedTuple, List

# noinspection PyUnresolvedReferences
import gym
import numpy as np
from keras import Input

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

MODEL_WEIGHTS_PATH = "model_weights/"
HISTORY_PATH = "history/"
MEMORY_PATH = "memory/"
MODEL_INIT_PATH = "model_init/"

# RNG initialisation
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

        self.states = np.zeros((max_len, 2))
        self.actions = np.zeros(max_len, dtype=np.uint8)
        self.rewards = np.zeros(max_len)
        self.next_states = np.zeros((max_len, 2))
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

    def save_experience(self, name: str):
        with open(MEMORY_PATH + name, 'wb') as f:
            pickle.dump(self, f)



class Agent:
    def __init__(
            self,
            alpha: float = 1e-3,
            gamma: float = 0.95,
            epsilon: float = 0.2,
            epsilon_reduction: float = 2e-5,
            epsilon_min: float = 1e-4,
            batch_size: int = 32,
            buffer_size: int = 25000
    ):
        self.alpha = alpha
        self.gamma = gamma
        self.epsilon = epsilon
        self.epsilon_reduction = epsilon_reduction
        self.epsilon_min = epsilon_min
        self.batch_size = batch_size
        self.action_space = [0, 1]

        self.model = self._create_model(self)
        self.target_model = self._create_model(self)
        self.target_model.set_weights(self.model.get_weights())

        self.buffer_size = buffer_size
        self.experience = ExperienceReplay(buffer_size)

        self.history = list()

    @staticmethod
    def _create_model(self) -> Model:
        """Creates a simple neural network"""
        init = HeUniform(seed=RANDOM_SEED)

        inputs = Input(shape=(2,))
        x = Dense(16, activation='relu', kernel_initializer=init)(inputs)
        x = Dense(16, activation='relu', kernel_initializer=init)(x)
        outputs = Dense(2, activation='linear', kernel_initializer=init)(x)

        model = Model(inputs=inputs, outputs=outputs)
        model.compile(loss='mse', optimizer=Adam(learning_rate=self.alpha))

        return model



    def save_experience(self, state, action, reward, next_state, terminal):
        """Saves the experience to the replay buffer"""
        self.experience.append(Transition(state, action, reward, next_state, terminal))

    def choose_action(self, state) -> int:
        """Chooses an action using an epsilon-greedy policy"""
        if rng.random() < self.epsilon:
            action = random.choice(self.action_space)
        else:
            pred = self.model(np.reshape(state, [1, 2])).numpy()
            action = np.argmax(pred)

        return action

    def learn(self):
        state, action, reward, next_state, terminal, sample_indices = self.experience.sample(self.batch_size)

        model_q_predictions = self.model(state).numpy()
        target_q_predictions = self.target_model(next_state).numpy()

        best_action = np.argmax(model_q_predictions, axis=1)
        target = reward + self.gamma * target_q_predictions[np.arange(self.batch_size), best_action] * ~terminal

        target_q = model_q_predictions
        target_q[np.arange(self.batch_size), action] = target

        td_error = np.abs(target - model_q_predictions[np.arange(self.batch_size), action])
        self.experience.update_priorities(td_error, sample_indices)

        self.model.fit(state, target_q, batch_size=self.batch_size, verbose=False)

    def train(self, env: gym.Env, n_episodes: int, save_interval: int):
        self.save_attribute("ddqlr_02_init.json")

        steps_taken = 3000
        start_time = time.time()

        # Main training
        print('Doing training')
        for episode in range(n_episodes):
            state = env.reset()
            terminal = False
            training_rewards = 0
            #
            # # Start the bird near the first pipe
            # for i in range(52):
            #     action = int(i % 18 == 0)  # Flap every 18 frames, keeps the bird close to middle
            #     state, *_ = env.step(action)

            while not terminal:
                action = self.choose_action(state)

                next_state, reward, terminal, info = env.step(action)
                steps_taken += 1
                self.save_experience(state, action, reward, next_state, terminal)

                if steps_taken % UPDATE_FREQUENCY == 0 and steps_taken > 3000:
                    self.learn()

                training_rewards += reward
                state = next_state

                # Copy model
                if steps_taken % COPY_FREQUENCY == 0:
                    self.copy_model()

                # # The most optimistic line in all of computing
                # if info['score'] > 50000:
                #     terminal = True

            # add episode score to history
            self.history.append(info["score"])

            print(
                f"Episode: {episode + 7401:<4} "
                f"Total Rewards: {training_rewards:<5} "
                f"Score: {info['score']:<4} "
                f"Epsilon: {self.epsilon:.4f} "
                f"Time Elapsed: {time.time() - start_time:.2f} s"
            )

            # Save intermediate models
            if (episode + 1) % save_interval == 0:
                self.save_model(f"model_{episode + 7401}.h5")
                self.save_history(f"ddqlr_02_{episode + 7401}.json")
                self.experience.save_experience(f"ddqlr_02_{episode + 7401}.plk")

            # Decrement epsilon
            if self.epsilon > self.epsilon_min:
                self.epsilon = max(self.epsilon - self.epsilon_reduction, self.epsilon_min)

    def copy_model(self):
        """Copies the model weights to the target model"""
        self.target_model.set_weights(self.model.get_weights())

    def save_model(self, model_name: str = 'flap_dqn.h5'):
        self.model.save(MODEL_WEIGHTS_PATH + model_name)

    def load_model(self, model_name: str):
        self.model.load_weights(model_name)
        self.target_model.load_weights(model_name)

    def save_history(self, name:str):
        with open(HISTORY_PATH + name, "w") as f:
            json.dump(self.history, f)
        self.history.clear()

    def load_experience(self, name: str):
        with open(name, 'rb') as f:
             self.experience = pickle.load(f)

    def save_attribute(self, name: str):
        attribute = {
            "gamma": self.gamma,
            "epsilon": self.epsilon,
            "epsilon_reduction": self.epsilon_reduction,
            "epsilon_min": self.epsilon_min,
            "batch_size": self.batch_size,
            "buffer_size": self.buffer_size
        }
        with open(MODEL_INIT_PATH + name, "w") as f:
            json.dump(attribute, f)

    def play(self, env):
        state = env.reset()

        terminated = False
        while not terminated:
            q_values = self.model(np.reshape(state, [1, 2])).numpy()
            action = np.argmax(q_values)
            state, _, terminated, info = env.step(action)

            # Rendering the game
            env.render()
            time.sleep(1 / 60)  # FPS


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



    env.close()
    agent.save_model()


if __name__ == '__main__':
    main()
