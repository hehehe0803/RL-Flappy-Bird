import numpy as np
np.set_printoptions(precision=4)
from keras.layers import Input, Dense
from keras.optimizers import Adam
from keras.models import Model

from collections import deque
import random
import time
import json

MODEL_WEIGHTS_PATH = "model_weights/"
REWARDS_PATH = "rewards_history/"

class DQLAgent:
    def __init__(
            self,
            n_states=12,
            n_actions=2,
            alpha=0.01,
            gamma=0.9,
            epsilon=0.1,
            epsilon_min=0.01,
            epsilon_decay=0.995,
            flap_penalty_w=0.5,
            gap_penalty_w=0.5
    ):
        self.n_states = n_states
        self.n_actions = n_actions

        self.gamma = gamma
        self.epsilon = epsilon
        self.epsilon_min = epsilon_min
        self.epsilon_decay = epsilon_decay
        self.alpha = alpha

        self.flap_penalty_w = flap_penalty_w
        self.gap_penalty_w = gap_penalty_w

        self.model = self._build_model()
        self.reward_history = None

    def _build_model(self):
        inputs = Input(shape=(self.n_states,))
        x = Dense(16, activation='relu')(inputs)
        x = Dense(16, activation='relu')(x)
        outputs = Dense(self.n_actions, activation='linear')(x)
        model = Model(inputs=inputs, outputs=outputs)
        model.compile(loss='mse', optimizer=Adam(learning_rate=self.alpha))
        return model

    def choose_action(self, state):
        if np.random.rand() <= self.epsilon:
            action = random.randrange(self.n_actions)
        else:
            q_values = self.model(state).numpy()
            action = np.argmax(q_values[0])
        return action

    def custom_reward(self, state, reward):
        """
        Because there is no pipe in the first ~300 frames, the only
        way to end the episode is if the bird hits the ground, so
        the agent learns to flap constantly, leading to a biased
        approximation. To fix this, we add a custom reward function
        for the agent to remain in the middle vertically.
        """
        # Get bird x, y position
        bird_x_pos = state[0][0] if state[0][0] >= 0 else state[0][3]
        bird_y_pos = state[0][9]

        # get middle position of next pipe
        pipe_top_y_pos = state[0][1] if state[0][0] >= 0 else state[0][4]
        pipe_bottom_y_pos = state[0][2] if state[0][0] >= 0 else state[0][5]
        middle_y_pos = (pipe_top_y_pos + pipe_bottom_y_pos) / 2

        # Scale by x position of bird
        x_pos_scaling = 1 / (1 + bird_x_pos)

        # Gap penalty
        gap_penalty = self.gap_penalty_w * x_pos_scaling * abs(bird_y_pos - middle_y_pos)

        # Final reward
        reward -= gap_penalty

        return reward, gap_penalty

    def train(self, env, n_episodes, save_interval):
        reward_history = []

        for episode in range(n_episodes):
            start = time.time()

            state, _ = env.reset()
            state = np.reshape(state, [1, self.n_states])
            terminated = False

            total_rewards = 0
            # tot_flap_penalty = 0
            total_gap_penalty = 0

            while not terminated:
                action = self.choose_action(state)
                next_state, reward, terminated, _, info = env.step(action)
                next_state = np.reshape(next_state, [1, self.n_states])

                reward, gap_penalty = self.custom_reward(state, reward)
                # tot_flap_penalty += flap_penalty
                total_gap_penalty += gap_penalty

                target = reward
                if not terminated:
                    target = (reward + self.gamma * np.amax(self.model(next_state).numpy()[0]))
                target_f = self.model(state).numpy()
                target_f[0][action] = target
                self.model.fit(state, target_f, epochs=1, verbose=0)

                state = next_state
                total_rewards += reward

            reward_history.append(total_rewards)

            # Save the model every 'n' episodes
            if (episode + 1) % save_interval == 0:
                self.save_model(MODEL_WEIGHTS_PATH + f"naive_dql_{episode + 1}.h5")
                self.save_reward(REWARDS_PATH + f"naive_dql_{episode + 1}.json")

            print(
                f"Episode: {episode + 1}/{n_episodes},"
                f" Reward: {total_rewards:.4f},"
                f" Score: {info['score']},"
                f" Epsilon: {self.epsilon:.4f},"
                # f" Flap penalty: {tot_flap_penalty:.2f},"
                f" Gap penalty: {total_gap_penalty:.2f}"
                f" Time: {time.time() - start:.2f}"
            )

            if self.epsilon > self.epsilon_min:
                self.epsilon *= self.epsilon_decay
            self.reward_history = reward_history

    def save_model(self, name):
        self.model.save_weights(name)

    def load_model(self, name):
        self.model.load_weights(name)

    def save_reward(self, name):
        with open(name, 'w') as f:
            json.dump(self.reward_history, f)

    def load_reward(self, name):
        self.reward_history = json.load(open(name))

    def play(self, env):
        state, _ = env.reset()
        state = np.reshape(state, [1, self.n_states])
        terminated = False
        while not terminated:
            q_values = self.model(state)
            action = np.argmax(q_values)
            state, _, terminated, _, info = env.step(action)
            state = np.reshape(state, [1, self.n_states])

            # Rendering the game
            env.render()
            time.sleep(1 / 60)  # FPS


class DoubleDQLReplayAgent(DQLAgent):
    def __init__(
            self,
            n_states=12,
            n_actions=2,
            alpha=0.001,
            gamma=0.9,
            epsilon=0.2,
            epsilon_min=0.01,
            epsilon_decay=0.995,
            gap_penalty_w=0.1,
            flap_pentaly_w=0.2,
            batch_size=64
    ):
        super().__init__(
            n_states,
            n_actions,
            alpha,
            gamma,
            epsilon,
            epsilon_min,
            epsilon_decay,
            gap_penalty_w,
            flap_pentaly_w
        )
        self.batch_size = batch_size
        self.memory = deque(maxlen=2000)
        self.target_model = self._build_model()
        self.update_target_model()

    def remember(self, state, action, reward, next_state, done):
        self.memory.append((state, action, reward, next_state, done))

    def update_target_model(self):
        self.target_model.set_weights(self.model.get_weights())

    def replay(self, batch_size):
        minibatch = random.sample(self.memory, batch_size)

        for state, action, reward, next_state, terminated in minibatch:
            target = self.model(state).numpy()
            if terminated:
                target[0][action] = reward
            else:
                action_target = np.argmax(self.model(next_state).numpy()[0])
                target[0][action] = reward + self.gamma * self.target_model(next_state).numpy()[0][action_target]
            self.model.fit(state, target, epochs=1, verbose=0)

    def train(self, env, n_episodes, save_interval):
        reward_history = []

        for episode in range(n_episodes):
            start = time.time()

            state, _ = env.reset()
            state = np.reshape(state, [1, self.n_states])
            terminated = False

            total_reward = 0
            total_gap_penalty = 0

            while not terminated:
                action = self.choose_action(state)
                next_state, reward, terminated, _, info = env.step(action)
                next_state = np.reshape(next_state, [1, self.n_states])

                reward, gap_penalty = self.custom_reward(state, action)

                self.remember(state, action, reward, next_state, terminated)

                state = next_state

                total_reward += reward
                total_gap_penalty += gap_penalty

                if len(self.memory) > self.batch_size:
                    self.replay(self.batch_size)

            self.update_target_model()

            if self.epsilon > self.epsilon_min:
                self.epsilon *= self.epsilon_decay

            reward_history.append(total_reward)

            # Save the model every 'x' episodes
            if (episode + 1) % save_interval == 0:
                self.save_model(MODEL_WEIGHTS_PATH + f"DDQLReplay_{episode + 1}.h5")
                self.save_reward(REWARDS_PATH + f"DDQLReplay_{episode + 1}.json")

            print(
                f"Episode: {episode + 1}/{n_episodes},"
                f" Reward: {total_gap_penalty:.4f},"
                f" Score: {info['score']},"
                f" Epsilon: {self.epsilon:.4f},"
                # f" Flap penalty: {tot_flap_penalty:.2f},"
                f" Gap penalty: {total_gap_penalty:.2f}"
                f" Time: {time.time() - start:.2f}"
            )

        return reward_history
