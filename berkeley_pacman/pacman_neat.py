# util
import sys
import pickle

# Berkeley Pac-Man
import pacman
import textDisplay
import graphicsDisplay

# NEAT
import neat

# Command line arguments
cmd_line_args = []

def eval_genomes(genome, config):
    genome.fitness = 0
    nn_model = neat.nn.FeedForwardNetwork.create(genome, config)
    pacmanType = pacman.loadAgent("NEATAgent", True)
    cmd_line_args['pacman'] = pacmanType(nn_model=nn_model)
    cmd_line_args['display'] = textDisplay.NullGraphics()
    # cmd_line_args['display'] = graphicsDisplay.PacmanGraphics()
    games = pacman.runGames(**cmd_line_args)
    result=0
    for game in games:
        result += game.state.getScore()
    return float(result)/float(len(games))


def eval_genomes_single(genomes, config):
    for id, genome in genomes:
        genome.fitness = 0
        nn_model = neat.nn.FeedForwardNetwork.create(genome, config)
        pacmanType = pacman.loadAgent("NEATAgent", True)
        cmd_line_args['pacman'] = pacmanType(nn_model=nn_model)
        cmd_line_args['display'] = textDisplay.NullGraphics()
        #cmd_line_args['display'] = graphicsDisplay.PacmanGraphics()
        games = pacman.runGames(**cmd_line_args)
        for game in games:
            genome.fitness += game.state.getScore()

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

    # Load from checkpoint
    #if cmd_line_args["checkpoint-neat"] != -1:
    #    p = neat.Checkpointer.restore_checkpoint('neat-checkpoint-'+str(cmd_line_args["checkpoint-neat"]))

    # Add a stdout reporter to show progress in the terminal.
    p.add_reporter(neat.StdOutReporter(True))
    stats = neat.StatisticsReporter()
    p.add_reporter(stats)
    p.add_reporter(neat.Checkpointer(5))

    # Run for up to 300 generations (parallel)
    pe = neat.ParallelEvaluator(4, eval_genomes)
    winner = p.run(pe.evaluate, 1000)
    #winner = p.run(eval_genomes_single, 100)

    # Display the winning genome.
    print('\nBest genome:\n{!s}'.format(winner))

    # Save the winner to disk
    with open('winner-feedforward.out', 'wb') as f:
        pickle.dump(winner, f)

    # Show output of the most fit genome against training data.
    print('\nOutput:')
    winner_net = neat.nn.FeedForwardNetwork.create(winner, config)

    # Load the final agent and run a few games with it
    pacmanType = pacman.loadAgent("NEATAgent", True)
    cmd_line_args['pacman'] = pacmanType(nn_model = winner_net)
    cmd_line_args['display'] = textDisplay.NullGraphics()
    #cmd_line_args['display'] = graphicsDisplay.PacmanGraphics()
    cmd_line_args['numGames'] = 10
    games = pacman.runGames(**cmd_line_args)
    pass
