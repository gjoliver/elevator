from controllers.dedicated import DedicatedAssigner
from controllers.rr import RoundRobinAssigner
from controllers.random import RandomAssigner
from simulator.elevator import Elevators
from simulator.evolver import Evolver

def sim():
  cfg = {
    'elevators': 4,
    'floors': 6,
    'horizon': 1000,
    'controller': DedicatedAssigner,
  }

  elevators = Elevators(cfg)
  controller = cfg['controller'](cfg)

  e = Evolver(elevators, controller, cfg)
  e.evolve()


if __name__ == '__main__':
  sim()
