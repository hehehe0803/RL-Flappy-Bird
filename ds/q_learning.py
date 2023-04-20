import time
import flappy_bird_gymnasium
import gymnasium
import json

import numpy as np
import random
import matplotlib.pyplot as plt

class Agent:
    """A Q Learning Agent designed to learn how to play flappy bird."""

    def __init__(self, alpha, epsilon, gamma):
        self.alpha = alpha
        self.epsilon = epsilon
        self.gamma = gamma

    def init_q(self, initial_state):
        self.q = {}

        init_state_key = self.state_to_key(initial_state)
        self.q[init_state_key] = [0.0, 0.0]

    def update_function(self, current_state, new_state, act, reward):
        """Update the q table using updating function"""
        current_state_key = self.state_to_key(current_state)
        new_state_key = self.state_to_key(new_state)

        # Check if index already exists
        if not new_state_key in self.q:
            # If index does not exist, create new item and append to array
            self.q[new_state_key] = [0.0, 0.0]

        # Get the current Q-value for the state and action
        current_q = self.q[current_state_key][act]

        # Find the maximum Q-value for the next state
        max_q = max(self.q[new_state_key])

        # Update the Q-value for the current state and action
        new_q = current_q + self.alpha * (reward + self.gamma * max_q - current_q)

        self.update_q_state(current_state, act, new_q)

    def epsilon_greedy(self, current_state):
        """Use epsilon greedy algorithm to choose action"""
        current_state_key = self.state_to_key(current_state)
        qs = self.q[current_state_key]

        action = 0
        if random.uniform(0, 1) < self.epsilon:
            # Select a random action
            action = random.randint(0, 1)
        else:
            # Select the greedy action
            max_q = max(qs) # 3rd item is state index
            count = qs.count(max_q)
            if count > 1:
                # If multiple actions have the same value, select do nothing
                action = 0
            else:
                action = qs.index(max_q)

        return action

    def state_to_key(self, state):
        # Clamp x and y values to be slightly below upper bounds
        x = min(state[0], 0.99)
        y = min(state[1], 0.99)
        b = min(state[2], 0.99)
        v = min(state[3], 0.9)
        
        # Calculate row and column indices from x and y values
        x = int(x * 100)
        y = int(y * 100)
        b = int(b * 100)
        v = int(v * 10)
        
        # Calculate unique index from row and column indices
        key = "{}_{}_{}_{}".format(x, y, b, v)

        return key
        
    def update_q_state(self, state, act, value):
        key = self.state_to_key(state)
        self.q[key][act] = value

def train(agent, episodes, target_score, time_start):
    """The main training loop"""
    # the environment
    env = gymnasium.make('FlappyBird-v0')

    episode_scores = []
    episode_rewards = []
    highest_score = 0

    # set initial state
    init_state = (env.reset())[0]
    next_h_pos = round(init_state[3], 2)
    next_v_pos = round(init_state[5], 2)
    bird_v_pos = round(init_state[9], 2)
    bird_vel = round(init_state[10], 1)
        
    state = (next_h_pos, next_v_pos, bird_v_pos, bird_vel)
    agent.init_q(state)

    # train over numerous episodes
    for e in range(0, episodes):
        print("EPISODE {}/{}".format(e, episodes))

        # reset state for each episode
        init_state = (env.reset())[0]
        terminal = False
        this_score = 0
        this_reward = 0

        # set initial state
        next_h_pos = round(init_state[3], 2)
        next_v_pos = round(init_state[5], 2)
        bird_v_pos = round(init_state[9], 2)
        bird_vel = round(init_state[10], 1)
        
        state = (next_h_pos, next_v_pos, bird_v_pos, bird_vel)

        count = 0
        while not terminal:
            # choose action
            action = agent.epsilon_greedy(state)

            # take action
            obs, reward, terminated, _, info = env.step(action)

            # get new state
            new_next_h_pos = round(obs[3], 2)
            new_next_v_pos = round(obs[5], 2)
            new_bird_v_pos = round(obs[9], 2)
            new_bird_vel = round(obs[10], 1)
            
            new_state = (new_next_h_pos, new_next_v_pos, new_bird_v_pos, new_bird_vel)

            # custom rewards
            if reward == -1:
                reward = -1000 
            else:
                reward = 1
            
            # print("ACTION -> {}".format(action))
            # print("REWARD -> {}".format(reward))
            # print("OLD Qs -> {}".format(agent.q[agent.state_to_index(state)]))
            # print("STATE -> {}".format(state))
            #print("UPDATING STATE-INDEX {}".format(agent.state_to_index(state)))
            # update q tables
            agent.update_function(state, new_state, action, reward)
            #print("NEW Qs -> {}".format(agent.q[agent.state_to_index(state)]))
            #print('')
            
            # set new state
            state = new_state

            # record score
            score = info['score']
            if score > this_score:
                this_score = score
            
            # record reward
            this_reward += reward


            # render environment
            env.render()
            time.sleep(1 / 250)  # FPS

            # reset environment if died
            if terminated and reward == -1000:
                reset = (env.reset())[0]
                reset_next_h_pos = round(reset[3], 2)
                reset_next_v_pos = round(reset[5], 2)
                reset_bird_v_pos = round(reset[9], 2)
                reset_bird_vel = round(reset[10], 1)
                
                state = (reset_next_h_pos, reset_next_v_pos, reset_bird_v_pos, reset_bird_vel)


            # terminate if highest or target score hit
            if score > min(highest_score, target_score):
                highest_score = score
                terminal = True

            count += 1

            if count % 10000 == 0:
                # save q table
                with open('ds/q_table.json', 'w') as f:
                    json.dump(agent.q, f, indent=4)

        episode_scores.append(this_score)
        episode_rewards.append(this_reward)

        # save q table
        with open('ds/q_table.json', 'w') as f:
            json.dump(agent.q, f, indent=4)
    
    return episode_scores, episode_rewards

# some parameters we can tweak
EPISODES = 100
ALPHA = 0.7
EPSILON = 0.0
GAMMA = 0.95
TARGET_SCORE = 3

# our agent
FLAPPY = Agent(ALPHA, EPSILON, GAMMA)

start_time = time.time()
scores, rewards = train(FLAPPY, EPISODES, TARGET_SCORE, start_time)
end_time = time.time()

time_took = ((end_time - start_time) / 60) / 60 # hours
print("Took {} Hours".format(time_took))
print("High Score {}".format(max(scores)))

# Uncropped Learning Curves
plt.figure(figsize=(10.666, 6))
plt.plot(range(EPISODES), rewards, label = "Rewards")
#plt.plot(range(EPISODES), scores, label = "Scores", color='orange')
plt.title("Flappy Bird Learning Curve")
plt.xlabel("Episodes Played")
plt.ylabel("Reward")
plt.legend()
plt.grid()
plt.show()
