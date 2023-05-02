import glob
import json
import re
import time

import flappy_bird_gym
import matplotlib.pyplot as plt
import numpy as np
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


def combine_history(pattern: str) -> list:
    filenames = sorted(glob.glob(pattern), key=lambda name: int(re.search(r"\d+", name).group(0)))

    print(filenames)

    combined_scores = []
    for filename in filenames:
        with open(filename, "r") as file:
            scores = json.load(file)
            combined_scores.extend(scores)

    return combined_scores

def plot_combined_scores(combined_scores: list):
    episodes = np.arange(1, len(combined_scores) + 1)
    mean_scores = np.cumsum(combined_scores) / episodes
    max_scores = np.maximum.accumulate(combined_scores)

    plt.figure(figsize=(12, 6))

    plt.scatter(episodes, combined_scores, label='Score', alpha=0.5, s=5)
    plt.plot(episodes, mean_scores, label='Mean Score', color='red')
    plt.plot(episodes, max_scores, label='Max Score', color='green')

    plt.xlabel('Episode')
    plt.ylabel('Score')
    plt.legend()
    plt.title('Scores per Episode')
    plt.grid()

    plt.gca().set_yscale('log')

    plt.show()

def play(n_episodes: int):
    env = flappy_bird_gym.make("FlappyBird-v0")
    scores = list()

    for e in range(n_episodes):
        clock = pygame.time.Clock()

        obs = env.reset()
        while not done:
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

            clock.tick(15)

            if done:
                env.render()
                scores.append(info["score"])
                time.sleep(1)

    env.close()
    return scores