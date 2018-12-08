# pacmanAgents.py
# ---------------
# Licensing Information:  You are free to use or extend these projects for
# educational purposes provided that (1) you do not distribute or publish
# solutions, (2) you retain this notice, and (3) you provide clear
# attribution to UC Berkeley, including a link to http://ai.berkeley.edu.
#
# Attribution Information: The Pacman AI projects were developed at UC Berkeley.
# The core projects and autograders were primarily created by John DeNero
# (denero@cs.berkeley.edu) and Dan Klein (klein@cs.berkeley.edu).
# Student side autograding was added by Brad Miller, Nick Hay, and
# Pieter Abbeel (pabbeel@cs.berkeley.edu).


from pacman import Directions
from game import Agent
from featureExtractors import *
import random
import game
import util
import numpy as np

class LeftTurnAgent(game.Agent):
    "An agent that turns left at every opportunity"

    def getAction(self, state):
        legal = state.getLegalPacmanActions()
        current = state.getPacmanState().configuration.direction
        if current == Directions.STOP: current = Directions.NORTH
        left = Directions.LEFT[current]
        if left in legal: return left
        if current in legal: return current
        if Directions.RIGHT[current] in legal: return Directions.RIGHT[current]
        if Directions.LEFT[left] in legal: return Directions.LEFT[left]
        return Directions.STOP

class GreedyAgent(Agent):
    def __init__(self, evalFn="scoreEvaluation"):
        self.evaluationFunction = util.lookup(evalFn, globals())
        assert self.evaluationFunction != None

    def getAction(self, state):
        # Generate candidate actions
        legal = state.getLegalPacmanActions()
        if Directions.STOP in legal: legal.remove(Directions.STOP)

        successors = [(state.generateSuccessor(0, action), action) for action in legal]
        scored = [(self.evaluationFunction(state), action) for state, action in successors]
        bestScore = max(scored)[0]
        bestActions = [pair[1] for pair in scored if pair[0] == bestScore]
        return random.choice(bestActions)


class BioAgent(Agent):

    integer_to_action_dict = {0: "Stop",
                              1: "West",
                              2: "East",
                              3: "North",
                              4: "South"}

    action_to_integer_dict = {"Stop": 0,
                              "West": 1,
                              "East": 2,
                              "North": 3,
                              "South": 4}

    actions = ["Stop", "West", "East", "North", "South"]

    def __init__(self, evalFn = "scoreEvaluation", nn_model = None,  extractor='SimpleExtractor'):
        self.nn_model = nn_model
        self.evaluationFunction = util.lookup(evalFn, globals())
        self.featExtractor = util.lookup(extractor, globals())()
        assert self.evaluationFunction is not None


    def getAction(self, state):
        assert self.nn_model is not None

        # Get the legal possible actions and generate the features
        legal = state.getLegalPacmanActions()
        dict_features = []
        for a in BioAgent.actions:
            if a in legal:
                dict_features.append(self.featExtractor.getFeatures(state, a))
            else:
                dict_features.append(util.Counter())

        # Create an array with them
        features = []
        for d in dict_features:
            features.append(d['closest-food'])
            features.append(d['bias'])
            features.append(d['#-of-ghosts-1-step-away'])
            features.append(d['eats-food'])
        features = np.array(features)

        next_action = BioAgent.integer_to_action_dict.get(np.argmax(self.nn_model.predict(features[np.newaxis,...])))
        assert next_action != -1
        if next_action in legal:
            return next_action
        return "Stop"


def scoreEvaluation(state):
    return state.getScore()

class NEATAgent(Agent):

    integer_to_action_dict = {0: "Stop",
                              1: "West",
                              2: "East",
                              3: "North",
                              4: "South"}

    action_to_integer_dict = {"Stop": 0,
                              "West": 1,
                              "East": 2,
                              "North": 3,
                              "South": 4}

    actions = ["Stop", "West", "East", "North", "South"]

    def __init__(self, evalFn = "scoreEvaluation", nn_model = None,  extractor='SimpleExtractor'):
        self.nn_model = nn_model
        self.evaluationFunction = util.lookup(evalFn, globals())
        self.featExtractor = util.lookup(extractor, globals())()
        assert self.evaluationFunction is not None


    def getAction(self, state):
        assert self.nn_model is not None

        # Get the legal possible actions and generate the features
        legal = state.getLegalPacmanActions()
        features = self.featExtractor.getFeaturesGeneral(state)

        # Flatten the array
        features = np.array(features).flatten()

        # Run the network
        next_action = NEATAgent.integer_to_action_dict.get(np.argmax(self.nn_model.activate(features)))

        assert next_action != -1
        if next_action in legal:
            return next_action
        return "Stop"


def scoreEvaluation(state):
    return state.getScore()

