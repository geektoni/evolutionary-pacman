import sys

# Berkeley Pac-Man
import pacman

import evolutionary
import graphicsDisplay

if __name__ == '__main__':
    cmd_line_args = pacman.readCommand(sys.argv[1:])
    agent = evolutionary.TorchModel(cmd_line_args)
    agent.evolve()
    pacmanType = pacman.loadAgent("BioAgent", True)
    cmd_line_args['pacman'] = pacmanType(nn_model=agent.model)
    cmd_line_args['display'] = graphicsDisplay.PacmanGraphics()
    cmd_line_args['numGames'] = 100
    games = pacman.runGames(**cmd_line_args)
    pass
