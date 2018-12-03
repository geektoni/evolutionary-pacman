# util
import numpy as np
import sys
from random import Random
from time import time

# Berkeley Pac-Man
import pacman
import textDisplay
import graphicsDisplay

# Keras
from keras.models import Sequential
from keras.layers import Dense

# inspyred
from inspyred import ec
from inspyred.ec import terminators
from inspyred.ec import selectors
from inspyred.ec import variators

# model save/load
import json
from keras.models import model_from_json

# NEAT
import neat

# Command line arguments
cmd_line_args = []

def generate_model():
    model = Sequential()
    model.add(Dense(10, input_shape = (20,), activation = "relu"))
    model.add(Dense(5, activation = "softmax"))
    return model


def eval_genomes(genomes, config):
    for genome_id, genome in genomes:
        genome.fitness = 0
        nn_model = neat.nn.FeedForwardNetwork.create(genome, config)
        pacmanType = pacman.loadAgent("NEATAgent", True)
        cmd_line_args['pacman'] = pacmanType(nn_model=nn_model)
        cmd_line_args['display'] = textDisplay.NullGraphics()
        # cmd_line_args['display'] = graphicsDisplay.PacmanGraphics()
        games = pacman.runGames(**cmd_line_args)
        for game in games:
            genome.fitness += game.state.getScore()


def generate_candidate(random, args):
    nn_model = args.get("nn_model")
    #weights = [np.random.sample(w.shape) for w in nn_model.get_weights()]
    weights = [np.random.uniform(-1, 1, w.shape) for w in nn_model.get_weights()]
    return weights


def evaluate_candidates(candidates, args):
    nn_model = args["nn_model"]
    cmd_line_args = args["cmd_line_args"]
    candidates_fitness = []
    for candidate in candidates:
        nn_model.set_weights(candidate)
        pacmanType = pacman.loadAgent("BioAgent", True)
        cmd_line_args['pacman'] = pacmanType(nn_model = nn_model)
        cmd_line_args['display'] =  textDisplay.NullGraphics()
        #cmd_line_args['display'] = graphicsDisplay.PacmanGraphics()
        games = pacman.runGames(**cmd_line_args)
        candidate_fitness = 0
        for game in games:
            candidate_fitness += game.state.getScore()
        candidates_fitness.append(candidate_fitness)
    print(candidates_fitness)
    return candidates_fitness


if __name__ == '__main__':
    cmd_line_args = pacman.readCommand(sys.argv[1:]) # Get game components based on input

    # Config file
    config_file = "../config-neat"

    # Load configuration.
    config = neat.Config(neat.DefaultGenome, neat.DefaultReproduction,
                         neat.DefaultSpeciesSet, neat.DefaultStagnation,
                         config_file)

    # Create the population, which is the top-level object for a NEAT run.
    p = neat.Population(config)

    # Add a stdout reporter to show progress in the terminal.
    p.add_reporter(neat.StdOutReporter(True))
    stats = neat.StatisticsReporter()
    p.add_reporter(stats)
    p.add_reporter(neat.Checkpointer(5))

    # Run for up to 300 generations.
    pe = neat.ParallelEvaluator(4, eval_genomes)
    winner = p.run(pe.evaluate, 300)

    # Display the winning genome.
    print('\nBest genome:\n{!s}'.format(winner))

    # Show output of the most fit genome against training data.
    print('\nOutput:')
    winner_net = neat.nn.FeedForwardNetwork.create(winner, config)

    pacmanType = pacman.loadAgent("NEATAgent", True)
    cmd_line_args['pacman'] = pacmanType(nn_model = winner_net)
    cmd_line_args['display'] = graphicsDisplay.PacmanGraphics()
    cmd_line_args['numGames'] = 10
    games = pacman.runGames(**cmd_line_args)
    pass
