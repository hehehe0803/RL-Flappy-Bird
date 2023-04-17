import json
import matplotlib.pyplot as plt

def draw_learning_curve(json_file, xlabel='Episodes', ylabel='Rewards', title='Learning Curve'):
    with open(json_file, 'r') as f:
        rewards = json.load(f)

    plt.figure(figsize=(10, 5))
    plt.plot(rewards, label='Rewards')
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.title(title)
    plt.legend()

    plt.show()