from controllers.rand import RandomAssigner
from controllers.rl import NumFeatures, RLAssigner
from controllers.rr import RoundRobinAssigner
from simulator.evolver import EvolverConfig, Evolver
from utils.hyper_params import HyperParams


def sim():
  hparams = HyperParams()
  hparams.nn_sizes = [NumFeatures(hparams.num_floors, hparams.num_elevators),
                      30, 30,
                      hparams.num_elevators]
  cfg = EvolverConfig(hparams=hparams,
                      horizon=1000,
                      controller=RLAssigner(hparams))
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
