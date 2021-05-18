import random

from simulator.env import Env
from simulator.rider import Rider

class Evolver(object):
  def __init__(self, elevators, controller, cfg):
    self._cfg = cfg

    self._elevators = elevators
    self._controller = controller
    self._env = Env()

  def evolve(self):
    horizon = self._cfg['horizon']
    try:
      while horizon == 0 or self._env.timer < horizon:
        # Pause for DEBUGGING
        # input('')

        # Random incoming rider.
        rider = Rider(self._env,
                      0,
                      random.choice(range(1, self._cfg['floors'])))

        picked = self._controller.Pick(self._elevators.state(), rider)
        self._elevators.commit(picked, rider)

        self._elevators.step()

        # DEBUGGING
        print(self._elevators)
        print()

        self._env.tick()
    except KeyboardInterrupt:
      print('sim stopped!')

    print(self._env.stats)
