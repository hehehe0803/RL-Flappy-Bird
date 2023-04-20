import gymnasium
import agent
import time

class AgentTrainer:
    """A class which trains an agent to play flappy bird"""

    def __init__(self, alpha, epsilon, gamma):
        # create the environment
        self.env = gymnasium.make('FlappyBird-v0')

        # set initial state
        self.state = self.convert_state(self.env.reset()[0])

        # create the agent
        self.agent = agent.Agent(alpha, epsilon, gamma, self.state)

    def train(self, episodes, min_target_score) -> dict:
        """Complete training cycles and return statistics"""
        self.min_target_score = min_target_score
        total_rewards = []
        average_rewards = []
        all_scores = []
        average_scores = []
        durations = []

        target_score = 1
        for e in episodes:
            # get score 5 times before increasing difficulty
            if e % 5 == 0:
                target_score += 1

            # train this episode
            stats = self.train_episode(target_score)

            # save episode stats
            average_rewards.append(stats['average_reward'])
            total_rewards.append(stats['total_reward'])
            average_scores.append(stats['average_score'])
            all_scores.append(stats['all_scores'])
            durations.append(stats['duration'])

        return {
            'average_rewards' : average_rewards,
            'total_rewards' : total_rewards,
            'all_scores' : all_scores,
            'average_scores' : average_scores,
            'durations' : durations
        }

    def train_epsiode(self, target_score) -> dict:
        start_time = time.time() # record start time
        complete = False # whether to complete episode

        scores = []
        rewards = []

        this_life_rewards = 0
        while not complete:
            # choose action
            action = self.agent.choose_action(self.state)

            # take action
            obs, reward, terminated, _, info = self.env.step(action)

            score = info['score'] # get score
            this_life_rewards += reward # update rewards for this life

            # add state, action, and reward to history
            self.agent.add_to_history(self.state, action, reward)

            # if bird died
            if terminated:
                self.agent.learn() # learn from mistakes

                # reset environment
                self.state = self.convert_state(self.env.reset()[0])
                scores.append(score)
                rewards.append(this_life_rewards)
                this_life_rewards = 0
                continue
            
            # if target score hit
            if score == target_score:
                self.agent.learn() # learn

                scores.append(score)
                rewards.append(this_life_rewards)
                this_life_rewards = 0
                complete = True # finish episode
                continue
            
            # otherwise update state and continue
            self.state = self.convert_state(obs)

        end_time = time.time() # record end time

        # colate and return episode statistics
        stats = {
            'average_rewards' : sum(rewards) / len(rewards),
            'total_reward' : sum(rewards),
            'average_score' : sum(scores) / len(scores),
            'all_scores' : scores,
            'duration' : end_time - start_time
        }
        return stats

    def convert_state(self, state) -> tuple:
        """Convert gym state to state that will suit our agent"""
        pipe_x = state[3] # horizintal position of next pipe
        top_pipe_y = state[4] # vertical position of next top pipe
        bottom_pipe_y = state[5] # vertical position of next bottom pipe
        bird_y = state[9] # vertical position of bird
        bird_v = state[10] # velocity of bird

        bird_pipe_y_diff = bird_y - bottom_pipe_y # difference between vertical position of bird and bottom pipe
        top_bottom_pipe_y_diff = top_pipe_y - bottom_pipe_y # difference between vertical position of top and bottom pipe

        return (round(pipe_x, 3), round(bird_pipe_y_diff, 3), round(bird_v, 3), round(top_bottom_pipe_y_diff, 3)) # round each to 3dp

