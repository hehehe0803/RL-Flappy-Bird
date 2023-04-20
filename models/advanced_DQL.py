import time

from models.DQL import DQLAgent

from collections import deque
import random
import numpy as np


class DQLReplayAgent(DQLAgent):
    def __init__(
            self,
            n_states,
            n_actions,
            alpha=0.001,
            gamma=0.9,
            epsilon=0.1,
            epsilon_min=0.01,
            epsilon_decay=0.995
    ):
        super().__init__(n_states, n_actions, alpha, gamma, epsilon, epsilon_min, epsilon_decay)
        self.memory = deque(maxlen=2000)

    def remember(self, state, action, reward, next_state, terminated):
        self.memory.append((state, action, reward, next_state, terminated))

    def replay(self, batch_size):
        minibatch = random.sample(self.memory, batch_size)
        for state, action, reward, next_state, terminated in minibatch:
            target = reward
            if not terminated:
                target = (reward + self.gamma * np.amax(self.model.predict(next_state, verbose=0)[0]))
            target_f = self.model.predict(state, verbose=0)
            target_f[0][action] = target
            self.model.fit(state, target_f, epochs=1, verbose=0)

        # epsilon and flap penalty decay
        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay

    def train(self, env, n_episodes, batch_size=32):
        reward_history = []
        gap_penalty_weight = 0.1

        for e in range(n_episodes):
            start = time.time()
            state, _ = env.reset()
            state = np.reshape(state, [1, self.state_size])
            terminated = False
            episode_reward = 0

            while not terminated:
                action = self.choose_action(state)
                next_state, reward, terminated, _, _ = env.step(action)
                next_state = np.reshape(next_state, [1, self.state_size])

                # Get the bird's vertical position from the state
                bird_position = state[0][9]
                top_pipe_position = state[0][4]
                bottom_pipe_position = state[0][5]

                # Calculate the position penalty
                middle_gap_position = (top_pipe_position + bottom_pipe_position) / 2
                gap_penalty = abs(bird_position - middle_gap_position)

                # Apply the gap penalty to the reward
                reward -= gap_penalty_weight * gap_penalty

                self.remember(state, action, reward, next_state, terminated)
                state = next_state
                episode_reward += reward

                if len(self.memory) > batch_size:
                    self.replay(batch_size)

            reward_history.append(episode_reward)
            print(f"Episode: {e + 1}/{n_episodes}, Reward: {episode_reward}, Time: {time.time() - start}")
