import time
import flappy_bird_gymnasium
import gymnasium
import json

import random
import matplotlib.pyplot as plt
import math

class Agent:
    """A Q Learning Agent designed to learn how to play flappy bird."""

    def __init__(self, alpha, epsilon, gamma, n, min_max_q):
        self.alpha = alpha
        self.epsilon = epsilon
        self.gamma = gamma
        self.n = n
        self.min_max_q = min_max_q
        self.history = []

    def init_q(self, q_table, initial_state):
        init_state_key = self.state_to_key(initial_state)
        if q_table is None:
            self.q = {}
            self.q[init_state_key] = [0.0, 0.0]
        else:
            self.q = q_table
            if not init_state_key in self.q:
                self.q[init_state_key] = [0.0, 0.0]

    def update_function(self):
        """Update the q table using updating function"""
        n_step_reward = sum([self.gamma**(i) * self.history[i][2] for i in range(0, self.n)])

        n_step_q_val = sum([self.gamma**(i-1) * self.q[self.state_to_key(self.history[i][0])][self.history[i][1]] for i in range(1, len(self.history))])
        
        new_q = self.alpha * (n_step_reward + self.gamma**self.n * n_step_q_val - self.q[self.state_to_key(self.history[0][0])][ self.history[0][1]])

        self.q[self.state_to_key(self.history[0][0])][ self.history[0][1]] = max(-self.min_max_q, min(self.min_max_q, (self.q[self.state_to_key(self.history[0][0])][ self.history[0][1]] + new_q)))
        
        if math.isnan(self.q[self.state_to_key(self.history[0][0])][ self.history[0][1]]):
            print("NaN :(")

    def epsilon_greedy(self, current_state):
        """Use epsilon greedy algorithm to choose action"""
        current_state_key = self.state_to_key(current_state)
        
        # Check if index already exists
        if not current_state_key in self.q:
            # If index does not exist, create new item and append to array
            self.q[current_state_key] = [0.0, 0.0]

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
        v = min(state[2], 0.9)
        
        # Calculate row and column indices from x and y values
        x = int(x * 100)
        y = int(y * 100)
        v = int(v * 10)
        
        # Calculate unique index from row and column indices
        key = "{}_{}_{}".format(x, y, v)

        return key

    def add_to_history(self, state, action, reward):
        
        self.history.append((state, action, reward))

        if len(self.history) == self.n:
            # if buffer is full, update q table
            self.update_function()
            self.history.pop(0) # remove the oldest value

    def clear_history(self):
        self.history = []

def train(agent, episodes, target_score, time_start, starting_q, render):
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
    bird_pipe_v_diff = round(bird_v_pos - next_v_pos, 2)
            
    state = (next_h_pos, bird_pipe_v_diff, bird_vel)
    agent.init_q(starting_q, state)

    # train over numerous episodes
    for e in range(0, episodes):
        terminal = False
        this_score = 0
        this_reward = 0

        count = 0
        while not terminal:
            # choose action
            action = agent.epsilon_greedy(state)

            # take action
            obs, reward, terminated, _, info = env.step(action)

            # custom rewards
            if reward == -1:
                reward = -100
            elif reward == 1:
                reward = 1000
            else:
                reward = 0
            
            # update history
            agent.add_to_history(state, action, reward)

            # get new state
            new_next_h_pos = round(obs[3], 2)
            new_next_v_pos = round(obs[5], 2)
            new_bird_v_pos = round(obs[9], 2)
            new_bird_vel = round(obs[10], 1)
            new_bird_pipe_v_diff = round(new_bird_v_pos - new_next_v_pos, 2)
            
            state = (new_next_h_pos, new_bird_pipe_v_diff, new_bird_vel)

            # record score
            score = info['score']
            if score > this_score:
                this_score = score
            
            # record reward
            this_reward += reward


            # render environmen
            if render:
                env.render()
                time.sleep(1 / 75)  # FPS

            # reset environment if died
            if terminated:
                reset = (env.reset())[0]
                reset_next_h_pos = round(reset[3], 2)
                reset_next_v_pos = round(reset[5], 2)
                reset_bird_v_pos = round(reset[9], 2)
                reset_bird_vel = round(reset[10], 1)
                reset_bird_pipe_v_diff = round(reset_bird_v_pos - reset_next_v_pos, 2)
            
                state = (reset_next_h_pos, reset_bird_pipe_v_diff, reset_bird_vel)

                agent.clear_history()


            # terminate if highest or target score hit
            if score > min(highest_score, target_score):
                highest_score = score
                terminal = True

                # also reset state
                reset = (env.reset())[0]
                reset_next_h_pos = round(reset[3], 2)
                reset_next_v_pos = round(reset[5], 2)
                reset_bird_v_pos = round(reset[9], 2)
                reset_bird_vel = round(reset[10], 1)
                reset_bird_pipe_v_diff = round(reset_bird_v_pos - reset_next_v_pos, 2)
            
                state = (reset_next_h_pos, reset_bird_pipe_v_diff, reset_bird_vel)

                agent.clear_history()

            count += 1
            if count % 500000 == 0:
                print(f'Time Elapsed: {time.time() - time_start:.2f}s')
                # save q table
                try:
                    with open('ds/q_table.json', 'w') as f:
                        json.dump(agent.q, f, indent=4)
                except:
                    print("Could Not Save Q Table")

        episode_scores.append(this_score)
        episode_rewards.append(this_reward)

        print(f'Episode: {e:<3} Total Rewards: {this_reward:5.1f} Score: {this_score:<3} Time Elapsed: {time.time() - time_start:.2f}s')

        # save q table
        try:
            with open('ds/q_table.json', 'w') as f:
                json.dump(agent.q, f, indent=4)
        except:
            print("Could Not Save Q Table")
    
    return episode_scores, episode_rewards

# some parameters we can tweak
EPISODES = 100
ALPHA = 0.7
EPSILON = 0.05
GAMMA = 0.95
N = 40
MIN_MAX_Q = 1e+100
TARGET_SCORE = 3
Q_JSON = 'q_table_1.json'
RENDER = True

Q = None
# with open('ds/{}'.format(Q_JSON), 'r') as f:
#     # Load the file contents into a dictionary
#     Q = json.load(f)

# our agent
FLAPPY = Agent(ALPHA, EPSILON, GAMMA, N, MIN_MAX_Q)

start_time = time.time()
scores, rewards = train(FLAPPY, EPISODES, TARGET_SCORE, start_time, Q, RENDER)
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
