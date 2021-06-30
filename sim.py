from controllers.rand import RandomAssigner
from controllers.rr import RoundRobinAssigner
from simulator.evolver import EvolverConfig, Evolver


def sim():
  cfg = EvolverConfig()
  e = Evolver(cfg)

  try:
    while e.time() < cfg.horizon:
      # Pause for DEBUGGING
      # input('')

      e.step()
  except KeyboardInterrupt:
    print('sim stopped!')

  print(e.stats())


if __name__ == '__main__':
  sim()
