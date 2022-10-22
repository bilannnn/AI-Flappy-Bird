import neat
from bird import Bird
from base import Base
from pipe import Pipe
from utils import *


def create_nets_birds_ge(genomes, config):
    nets, birds, ge = [], [], []
    for genome_id, genome in genomes:
        genome.fitness = 0  # start with fitness level of 0
        net = neat.nn.FeedForwardNetwork.create(genome, config)
        nets.append(net)
        birds.append(Bird(230, 350))
        ge.append(genome)
    return nets, birds, ge


def remove_nets_ge_for_died_birds(nets, birds, ge, pipe):
    for bird in birds:
        if pipe.collide(bird, WIN) or (bird.y + bird.img.get_height() - 10 >= FLOOR or bird.y < -50):
            ge[birds.index(bird)].fitness -= 1
            nets.pop(birds.index(bird))
            ge.pop(birds.index(bird))
            birds.pop(birds.index(bird))


def activation(net, bird, pipe):
    # send bird location, top and bottom pipe location and
    return net.activate((bird.y, abs(bird.y - pipe.height), abs(bird.y - pipe.bottom)))


def eval_genomes(genomes, config):
    """
    runs the simulation of the current population of
    birds and sets their fitness based on the distance they
    reach in the game.
    """
    global gen
    gen += 1
    score = 0
    base = Base(FLOOR)
    pipes = [Pipe(WIN_WIDTH)]
    clock = pygame.time.Clock()

    # start by creating lists holding the genome itself, the
    # neural network associated with the genome and the
    # bird object that uses that network to play
    nets, birds, ge = create_nets_birds_ge(genomes, config)

    # start game
    while len(birds) > 0:
        clock.tick(30)

        check_if_quit()

        pipe_id = 0
        if len(birds) > 0:
            if len(pipes) > 1 and birds[0].x > pipes[0].x + pipes[0].PIPE_TOP.get_width():
                pipe_id = 1  # pipe on the screen for neural network input

        for x, bird in enumerate(birds):
            ge[x].fitness += 0.1
            bird.move()
            # determine from network whether to jump or not
            output = activation(nets[birds.index(bird)], bird, pipes[pipe_id])
            if output[0] > 0.5:
                bird.jump()

        base.move()

        for pipe in pipes:
            pipe.move()

            remove_nets_ge_for_died_birds(nets, birds, ge, pipe)

            if not pipe.passed and pipe.x < bird.x:
                pipe.passed = True
                score += 1

                for genome in ge:
                    genome.fitness += 5
                pipes.append(Pipe(WIN_WIDTH))

            if pipe.x + pipe.PIPE_TOP.get_width() < 0:
                pipes.remove(pipe)

        draw_window(WIN, birds, pipes, base, score, gen, pipe_id)


def run(config_file):
    """
    runs the NEAT algorithm to train a neural network to play flappy bird.
    :param config_file: location of config file
    :return: None
    """
    config = neat.config.Config(neat.DefaultGenome, neat.DefaultReproduction,
                                neat.DefaultSpeciesSet, neat.DefaultStagnation,
                                config_file)

    # Create the population, which is the top-level object for a NEAT run.
    p = neat.Population(config)

    # Add a stdout reporter to show progress in the terminal.
    p.add_reporter(neat.StdOutReporter(True))
    stats = neat.StatisticsReporter()
    p.add_reporter(stats)

    # Run for up to 50 generations.
    winner = p.run(eval_genomes, 50)

    # show final stats
    print('\nBest genome:\n{!s}'.format(winner))


if __name__ == '__main__':
    # Determine path to configuration file. This path manipulation is
    # here so that the script will run successfully regardless of the
    # current working directory.
    local_dir = os.path.dirname(__file__)
    config_path = os.path.join(local_dir, 'config-feedforward.txt')
    run(config_path)
