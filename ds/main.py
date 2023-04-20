import time
import flappy_bird_gymnasium
import json

import random
import matplotlib.pyplot as plt
import math

# some parameters we can tweak
EPISODES = 100
ALPHA = 0.7
EPSILON = 0.05
GAMMA = 0.95
Q_PATH = 'q_table.json'
RENDER = True


