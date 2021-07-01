from controllers.rand import RandomAssigner
from controllers.rl import RLAssigner
from controllers.rr import RoundRobinAssigner
from simulator.evolver import EvolverConfig, Evolver


NUM_ELEVATORS = 4
NUM_FLOORS = 6

def sim():
  cfg = EvolverConfig(num_elevators=NUM_ELEVATORS,
                      num_floors=NUM_FLOORS,
                      controller=RLAssigner(NUM_ELEVATORS,
                                            NUM_FLOORS))
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
