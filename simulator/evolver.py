import random

from controllers.dedicated import DedicatedAssigner
from simulator.elevator import Elevators
from simulator.env import Env
from simulator.rider import Rider


class EvolverConfig(object):
  def __init__(self,
               num_elevators=4,
               num_floors=6,
               horizon=1000,
               controller=None):
    self.num_elevators = num_elevators
    self.num_floors = num_floors
    self.horizon = horizon
    if not controller:
      self.controller = DedicatedAssigner(num_floors)
    else:
      self.controller = controller


class Evolver(object):
  def __init__(self, cfg):
    self._cfg = cfg

    self._elevators = Elevators(num_elevators=cfg.num_elevators,
                                num_floors=cfg.num_floors)
    self._controller = cfg.controller
    self._env = Env()

  def time(self):
    # Get universe time.
    return self._env.timer

  def stats(self):
    return self._env.stats

  def state(self):
    return self._elevators.state()

  def step(self):
    # Random incoming rider.
    rider = Rider(self._env,
                  0,
                  random.choice(range(1, self._cfg.num_floors)))

    picked = self._controller.Pick(self._elevators.state(), rider)
    self._elevators.commit(picked, rider)
    self._elevators.step()

    # DEBUGGING
    # print(self._elevators)

    self._env.tick()
