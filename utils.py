import json
import time

import flappy_bird_gym
import matplotlib.pyplot as plt
import pygame


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

def play():
    env = flappy_bird_gym.make("FlappyBird-v0")

    clock = pygame.time.Clock()
    score = 0

    obs = env.reset()
    while True:
        env.render()

        # Getting action:
        action = 0
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()
            if event.type == pygame.KEYDOWN and (
                    event.key == pygame.K_SPACE or event.key == pygame.K_UP
            ):
                action = 1

        # Processing:
        obs, reward, done, info = env.step(action)

        score += reward
        print(f"Obs: {obs}\n" f"Score: {info}\n")

        clock.tick(15)

        if done:
            env.render()
            time.sleep(0.5)
            break

    env.close()