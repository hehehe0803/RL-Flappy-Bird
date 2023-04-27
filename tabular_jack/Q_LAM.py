"""Implementation of Watkins's Q-learning Lambda tabular method
Discretises the state-action space to either 1dp"""

import random
import datetime
import json
import time
import numpy as np
import matplotlib.pyplot as plt
from collections import defaultdict

import flappy_bird_gym
import gym

# hyperparameteres
episodes = 10000
init_epsilon = 0.1
init_learning_rate = 0.7

# create the agent

class Agent:
    def __init__(self):
        # hyperparameters
        self.episodes = episodes
        self.epsilon = init_epsilon
        self.min_epsilon = 0.00001
        self.epsilon_decay = 0.9993
        self.learning_rate = init_learning_rate
        self.min_learning_rate = 0.1
        self.learning_rate_decay = (self.learning_rate - self.min_learning_rate) / self.episodes
        self.lambda_val = 0.1
        self.discount_factor = 0.95

        # initialise Q table and eligibility trace
        self.action_space = [0, 1]
        self.q_table = defaultdict(lambda: np.zeros(len(self.action_space)))
        self.eligibility_trace = defaultdict(lambda: np.zeros(len(self.action_space)))

    def choose_action(self, state) -> int: # epsilon greedy policy

        state = (round(state[0], 1), round(state[1], 1))
        
        if np.random.random() < self.epsilon:
            action = np.random.choice(self.action_space)
        else:
            q_values = self.q_table[state]
            action = np.argmax(q_values)
        
        # update epsilon
        self.epsilon = max(self.epsilon * self.epsilon_decay, self.min_epsilon)

        return action

    def learn(self, history):

        for state, action, reward, next_state, next_action in history:
            
            state = (round(state[0], 1), round(state[1], 1)) # round to 1 decimal place
            next_state = (round(next_state[0], 1), round(next_state[1], 1))

            # check if state is in Q table
            if state not in self.q_table: # add if not in table
                self.q_table[state] = np.zeros(len(self.action_space))
            
            if next_state not in self.q_table: # add if not in table
                self.q_table[next_state] = np.zeros(len(self.action_space))

            # retrieve values
            current_q = self.q_table[state][action]
            greedy_next_action = np.argmax(self.q_table[next_state])  # get the greedy action
            next_q = self.q_table[next_state][greedy_next_action]

            td_error = reward + self.discount_factor * next_q - current_q # calculate temporal difference error
            self.eligibility_trace[state][action] += 1 # update eligibility trace

            # update Q vals and eligibility trace
            for s, actions in self.q_table.items():
                for a in self.action_space:
                    self.q_table[s][a] += self.learning_rate * td_error * self.eligibility_trace[s][a]

                    # reset eligibility traces for non-greedy actions
                    if a == greedy_next_action:
                        self.eligibility_trace[s][a] *= self.discount_factor * self.lambda_val
                    else:
                        self.eligibility_trace[s][a] = 0
                        
            # update learning rate
            self.learning_rate = max(self.learning_rate - self.learning_rate_decay, self.min_learning_rate)

def main():
    env = gym.make('FlappyBird-v0')
    agent = Agent()
    start_time = time.time()
    scores = []
    episode_num = []
    av_scores = []

    print('Doing training')
    for episode in range(agent.episodes):
        state = env.reset()
        terminal = False
        training_rewards = 0
        action = agent.choose_action(state)
        history = []

        while not terminal:
            # take step and choose next action
            next_state, reward, terminal, info = env.step(action)
            next_action = agent.choose_action(next_state)

            # set max score to 1000
            if info['score'] == 1000:
                terminal = True

            # custom reward function
            if terminal:
                reward += -100
                reward += info['score']^2

            # update rewards and history
            training_rewards += reward
            history.append((state, action, reward, next_state, next_action))

            # update state and action
            state = next_state
            action = next_action

            # env.render()
            # time.sleep(1 / 200) # FPS

        # update q tables after episode
        agent.learn(history)
        history.clear()

        # track score and episode number
        scores.append(info['score'])
        episode_num.append(episode)

        # calculate rolling av score
        if episode >= 100:
            av_score = sum(scores[-100:]) / 100
            av_scores.append(av_score)
        else:
            av_scores.append(0)

        print(f'Episode: {episode:<3} Reward: {training_rewards:5.1f} Score: {info["score"]:<3} Time Elapsed: {time.time() - start_time:.2f} s Q-table size: {len(agent.q_table)} High Score: {max(scores)}')

    env.close()

    # save Q table in json file
    timestamp = datetime.datetime.now().strftime('%Y_%m_%d_%H_%M')
    q_table_file = f'q_table_{timestamp}.json'

    q_table_dict = {}
    for i, j in agent.q_table.items():
        q_table_dict[str(i)] = j.tolist()
    with open(q_table_file, 'w') as f:
        json.dump(q_table_dict, f)

    # plot the learning curve
    plt.figure(figsize = (10, 6))
    plt.plot(episode_num, scores, '.', color = 'black', markersize = 2, label = 'Score')
    plt.plot(episode_num, av_scores, color = 'blue', label = 'Average Score')
    plt.xlabel('Episode')
    plt.ylabel('Score')
    plt.title(f'Flappy Bird Learning Curve\n' \
            f'Init Epsilon: {init_epsilon}, Epsilon Decay Rate: {agent.epsilon_decay}, ' \
            f'Discount Factor: {agent.discount_factor}, Init Learning Rate: {init_learning_rate}, ' \
            f'Lambda: {agent.lambda_val}')
    plt.legend()

    # save learning curve
    learning_curve_file = f'learning_curve_{timestamp}.png'
    plt.savefig(learning_curve_file)
    plt.show()

if __name__ == '__main__':
    main()