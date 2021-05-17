import random

from simulator.elevator import NUM_FLOORS
from simulator.env import Env
from simulator.rider import Rider

class Evolver(object):
  def __init__(self, elevators, controller, horizon = 0):
    self.horizon = horizon

    self.elevators = elevators
    self.controller = controller
    self.env = Env()

  def evolve(self):
    try:
      while self.horizon == 0 or self.env.timer < self.horizon:
        # Pause for DEBUGGING
        # input('')

        # Random incoming rider.
        rider = Rider(self.env, 0, random.choice(range(1, NUM_FLOORS)))

        picked = self.controller.Pick(self.elevators.state(), rider)
        self.elevators.commit(picked, rider)

        self.elevators.step()

        # DEBUGGING
        # print(self.elevators)

        self.env.tick()
    except KeyboardInterrupt:
      print('sim stopped!')

    print(self.env.stats)
