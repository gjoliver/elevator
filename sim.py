from controllers.random import RandomAssigner
from controllers.rr import RoundRobinAssigner
from simulator.evolver import EvolverConfig, Evolver


def sim():
  e = Evolver(EvolverConfig())
  e.evolve()


if __name__ == '__main__':
  sim()
