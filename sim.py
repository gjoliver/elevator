from controllers.random import RandomAssigner
from simulator.elevator import Elevators
from simulator.evolver import Evolver

def sim():
  elevators = Elevators(4)
  controller = RandomAssigner()

  e = Evolver(elevators, controller)
  e.evolve()

  # TODO Report stats.


if __name__ == '__main__':
  sim()
