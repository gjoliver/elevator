import random

from controllers.dedicated import DedicatedAssigner
from simulator.elevator import Elevators
from simulator.env import Env
from simulator.rider import Rider


class EvolverConfig(object):
  def __init__(self, hparams, horizon, controller):
    self.hparams = hparams
    self.horizon = horizon
    self.controller = controller


class Evolver(object):
  def __init__(self, cfg):
    self._cfg = cfg

    self._elevators = Elevators(num_elevators=cfg.hparams.num_elevators,
                                num_floors=cfg.hparams.num_floors)
    self._controller = cfg.controller
    self._env = Env()

  def time(self):
    # Get universe time.
    return self._env.timer

  def stats(self):
    return self._env.stats

  def step(self):
    # New rider
    rider = Rider(
      self._env, 0, random.choice(range(1, self._cfg.hparams.num_floors)))

    state = self._elevators.state()

    action = self._controller.Pick(state, rider)
    # Once we know which elevator is going to pick up the rider,
    # reward becomes simple. For every floor that this rider has
    # to wait, we reward -1 to encourage fast overal pickup time.
    reward = -abs(rider.pickup - state[action]['floor'])

    self._elevators.commit(action, rider)
    self._elevators.step()

    # DEBUGGING
    # print(self._elevators)

    self._env.tick()

    return rider, state, action, reward
