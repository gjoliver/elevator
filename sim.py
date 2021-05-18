from controllers.rr import RoundRobinAssigner
from controllers.random import RandomAssigner
from simulator.elevator import Elevators
from simulator.evolver import Evolver

def sim():
  elevators = Elevators(4)
  controller = RoundRobinAssigner()

  e = Evolver(elevators, controller, horizon=1000)
  e.evolve()


if __name__ == '__main__':
  sim()
