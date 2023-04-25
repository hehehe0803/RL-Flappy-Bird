import trainer
import plotter
import json

# some parameters we can tweak
EPISODES = 10000
EPSILON = 0.15
EPSILON_DECAY = 1e-5
EPSILON_MIN = 1e-4
GAMMA = 1
Q_PATH = 'ds/20000_episodes.json'
RENDER = False
LEARN = True

# maybe penalise based on how many flaps done
REWARDS = {
    'die' : -1000,
    'getpoint' : 1000,
    'stayalive' : 1,
    'flap' : -100
}

Q = {}
# with open(Q_PATH, 'r') as f:
#     # Load the file contents into a dictionary
#     Q = json.load(f)

flappy_trainer = trainer.AgentTrainer(EPSILON, EPSILON_DECAY, EPSILON_MIN, GAMMA, REWARDS, Q)
stats = flappy_trainer.train(EPISODES, RENDER, LEARN)

avr_rewards = stats['average_rewards']
tot_rewards = stats['total_rewards']
avr_scores = stats['average_scores']
all_scores = stats['all_scores']
highest_scores = stats['highest_scores']
new_states_found_list = stats['new_states_found_list']

plotter.plot_results(EPISODES, all_scores, avr_scores, highest_scores)
plotter.plot_new_states(EPISODES, new_states_found_list)




