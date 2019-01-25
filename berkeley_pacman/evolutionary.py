from tqdm import tqdm
from deap import base
from deap import creator
from deap import tools

import torch
import torch.nn as nn
import numpy as np

import logging
logging.getLogger().setLevel(logging.INFO)

import random
import multiprocessing

# Berkeley Pac-Man
import pacman
import textDisplay
import graphicsDisplay

from contextlib import contextmanager

@contextmanager
def terminating(thing):
    try:
        yield thing
    finally:
        thing.terminate()


class EvolutionaryModel:

    def __init__(self, args):
        self.cmd_line_args = args
        self.CR = self.cmd_line_args["CR"]
        self.F = self.cmd_line_args["F"]
        self.MU = self.cmd_line_args["MU"]
        self.NGEN = self.cmd_line_args["NGEN"]
        self.DEVICE = self.cmd_line_args["DEVICE"]
        self.NCPU = self.cmd_line_args["NCPU"]
        self.MODE_NO_SCREEN = self.cmd_line_args["MODE_NO_SCREEN"]

        if not self.MODE_NO_SCREEN:
            self.cmd_line_args['display'] = graphicsDisplay.PacmanGraphics()
        else:
            self.cmd_line_args['display'] = textDisplay.NullGraphics()

    def create_model(self, device):
        pass

    def generate_individual(self, individual, model):
        pass

    def evaluate_individual(self, model, individual):
        pass

    def evolve(self):
        pass


class TorchModel(EvolutionaryModel):

    def __init__(self, args):
        EvolutionaryModel.__init__(self, args)

        creator.create("FitnessMax", base.Fitness, weights=(1.0,))
        creator.create("Individual", np.ndarray, fitness=creator.FitnessMax)

        self.model = self.create_model(self.DEVICE)

        self.toolbox = base.Toolbox()
        self.toolbox.register("individual", self.generate_individual, creator.Individual, self.model)
        self.toolbox.register("population", tools.initRepeat, list, self.toolbox.individual)
        self.toolbox.register("select", tools.selRandom, k=3)
        self.toolbox.register("evaluate", self.evaluate_individual, self.model)

        self.pop = self.toolbox.population(n=self.MU)
        self.hof = tools.HallOfFame(1, similar=np.array_equal)

    def create_model(self, device):
        model = nn.Sequential(nn.Linear(35, 70), nn.ReLU(), nn.Linear(70, 5), nn.LogSoftmax(dim=0))
        for parameter in model.parameters():
            parameter.requires_grad = False
        return model

    def generate_individual(self, individual, model):
        weights = []
        for parameter in model.parameters():
            if len(parameter.size()) == 1:
                parameter_dim = parameter.size()[0]
                # weights.append(np.random.rand(parameter_dim) * np.sqrt(1 / (parameter_dim)))
                weights.append(np.random.rand(parameter_dim))
            else:
                parameter_dim_0, parameter_dim_1 = parameter.size()
                weights.append(np.random.rand(parameter_dim_0, parameter_dim_1) * np.sqrt(1 / (parameter_dim_0 + parameter_dim_1)))
                # weights.append(np.random.rand(parameter_dim_0, parameter_dim_1))
        return individual(np.array(weights))

    def evaluate_individual(self, model, individual):

        for parameter, numpy_array in zip(model.parameters(), individual):
            parameter.data = torch.from_numpy(numpy_array)

        pacmanType = pacman.loadAgent("BioAgent", True)
        self.cmd_line_args['pacman'] = pacmanType(nn_model=model)
        games = pacman.runGames(**self.cmd_line_args)
        candidate_fitness = 0
        for game in games:
            candidate_fitness += game.state.getScore()
        score = (candidate_fitness,)
        return score

    def differential_evolution(self, agent, population):
        a, b, c = self.toolbox.select(population)
        y = self.toolbox.clone(agent)
        index = random.randrange(len(agent))

        for i, value in enumerate(agent):
            if i == index or random.random() < self.CR:
                y[i] = a[i] + self.F * (b[i] - c[i])

        #for layer_index, layer_weights in enumerate(y):
        #    index = random.randrange(layer_weights.shape[0])
        #    for i, value in enumerate(layer_weights):
        #        if i == index or random.random() < self.CR:
        #            y[layer_index][i] = a[layer_index][i] + self.F * (b[layer_index][i] - c[layer_index][i])

        y.fitness.values = self.toolbox.evaluate(y)
        if y.fitness > agent.fitness:
            return y
        else:
            return agent

    def evolve(self):

        for generation in tqdm(range(self.NGEN), total=self.NGEN):
            new_agents = []
            for agent in self.pop:
                new_agents.append(self.differential_evolution(agent, self.pop))
            self.pop = new_agents
            self.hof.update(self.pop)
            if (generation % 10) == 0:
                print self.hof[0].fitness.values[0]

        best_individual = self.hof[0]
        for parameter, numpy_array in zip(self.model.parameters(), best_individual):
            parameter.data = torch.from_numpy(numpy_array)