import gym
import numpy as np
import random
import time

#https://stats.stackexchange.com/questions/363076/sarsa-with-linear-function-approximation-weight-overflow

# Get the game we want to play
env = gym.make("MsPacman-ram-v0")

# How many feature
ftotal = 10

# Total feature size
feature_size = env.observation_space.shape[0]

# Total possible actions
action_size = 5

# Total number of games
games_number = 100

# Theta end eligibility
w = np.random.rand(action_size, feature_size)

def getQValue(features, action, theta):
    return np.sum(features*theta[action])

def epsilon_greedy_policy(epsilon, features, theta):

    value = np.random.choice(np.arange(0,2), p=[1-epsilon, epsilon])

    if value == 0:
        qValues = np.array([getQValue(features, a, theta) for a in range(0,action_size)])
        return np.argmax(qValues)
    else:
        return random.choice(range(action_size))


alpha = 1e-7
epsilon = 0.5

for game in range(0, games_number):

    observation = env.reset()
    action = random.randint(0, action_size-1)
    done = False

    print("[*] Evaluating Game {}".format(game))

    # Play the actual game
    while not done:

        env.render()
        newObservation, reward, done, info = env.step(action)

        if (done):
            w = w + alpha*(reward - getQValue(observation, action, w))*observation
            break

        newAction = epsilon_greedy_policy(epsilon, newObservation, w)

        w = w + alpha*(reward + 0.1*getQValue(newObservation, newAction, w) - getQValue(observation, action, w))*observation

        action = newAction
        observation = newObservation


