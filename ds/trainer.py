import gymnasium
import flappy_bird_gymnasium
import agent
import time
import json

class AgentTrainer:
    """A class which trains an agent to play flappy bird"""

    def __init__(self, epsilon, epsilon_decay, epsilon_min, gamma, rewards, q):
        self.rewards = rewards
        # create the environment
        self.env = gymnasium.make('FlappyBird-v0')

        # set initial state
        self.state = self.convert_state(self.env.reset()[0])

        # create the agent
        self.agent = agent.Agent(epsilon, epsilon_decay, epsilon_min, gamma, self.state, q)


    def train(self, episodes, render, learn) -> dict:
        """Complete training cycles and return statistics"""
        total_rewards = []
        average_rewards = []
        all_scores = []
        average_scores = []
        highest_scores = []
        durations = []
        new_states_found_list = []

        self.start_time = time.time() # record start time
        for e in range(0, episodes):

            # train this episode
            stats = self.train_episode(render, learn)

            # save episode stats
            average_rewards.append(stats['average_rewards'])
            total_rewards.append(stats['total_reward'])
            all_scores.append(stats['score'])
            average_scores.append(sum(all_scores) / len(all_scores))
            durations.append(stats['duration'])
            highest_scores.append(max(all_scores))
            new_states_found_list.append(stats['new_states_found'])
            
            print('Episode {} - Score {} - Reward {} - Duration {}s - Epsilon {}'.format(e, stats['score'], stats['total_reward'], round(stats['duration'], 2), self.agent.epsilon))

        # save q table
        try:
            with open('ds/q_table.json', 'w') as f:
                json.dump(self.agent.q, f, indent=4)
        except:
            print("Could Not Save Q Table")

        return {
            'average_rewards' : average_rewards,
            'total_rewards' : total_rewards,
            'all_scores' : all_scores,
            'average_scores' : average_scores,
            'highest_scores' : highest_scores,
            'durations' : durations,
            'new_states_found_list' : new_states_found_list
        }

    def train_episode(self, render, learn) -> dict:
        complete = False # whether to complete episode

        episode_rewards = 0
        episode_score = 0
        count = 0
        new_states_found = 0
        while not complete:
            count += 1

            # choose action
            action = self.agent.choose_action(self.state)

            # take action
            obs, reward, terminated, _, info = self.env.step(action)

            if reward == -1:
                reward = self.rewards['die']
            elif reward == 1:
                reward = self.rewards['getpoint']
            else:
                reward = self.rewards['stayalive']

            if action == 1:
                reward += self.rewards['flap'] # penalise flapping

            episode_rewards += reward # update rewards for this life

            # get new state and add to q table
            new_state = self.convert_state(obs)
            found_new = self.agent.discover_state(new_state)

            if found_new:
                new_states_found += 1

            # update q table
            if learn:
                self.agent.learn(self.state, action, reward, new_state)

            if render:
                # render environment
                self.env.render()
                time.sleep(1 / 100)  # FPS

            # if bird died
            if terminated:
                # reset environment
                self.state = self.convert_state(self.env.reset()[0])
                episode_score = info['score'] # get score
                complete = True
                continue
            
            # otherwise update state and continue
            self.state = new_state

        # decay exploration
        self.agent.decay_epsilon()

        end_time = time.time() # record end time

        # colate and return episode statistics
        stats = {
            'average_rewards' : episode_rewards / count,
            'total_reward' : episode_rewards,
            'score' : episode_score,
            'duration' : end_time - self.start_time,
            'new_states_found' : new_states_found
        }
        return stats

    def convert_state(self, state) -> tuple:
        """Convert gym state to state that will suit our agent"""
        pipe_x = state[3] # horizintal position of next pipe
        top_pipe_y = state[4] # vertical position of next top pipe
        bottom_pipe_y = state[5] # vertical position of next bottom pipe
        bird_y = state[9] + 0.2 # vertical position of bird normalised 0-1
        #bird_v = state[10] # velocity of bird

        pipe_y_mid = (top_pipe_y + bottom_pipe_y) / 2 # pipe midpoint
        normalized_bird_pipe_y = ((bird_y - pipe_y_mid) * 2) / abs(top_pipe_y - bottom_pipe_y) # vertical distance between bird and pipe midpoint normalised -1 - 1

        return (pipe_x, normalized_bird_pipe_y)
