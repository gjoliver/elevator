import random

from simulator.elevator import NUM_FLOORS
from simulator.rider import Rider

class Evolver(object):
  def __init__(self, elevators, controller):
    self.elevators = elevators
    self.controller = controller

  def evolve(self):
    while True:
      # Pause for testing.
      input('')

      # Floor 1 to 6.
      rider = Rider(0, random.choice(range(1, NUM_FLOORS)))

      state = self.elevators.state()
      picked = self.controller.Pick(state, rider)
      self.elevators.commit(picked, rider)

      print(self.elevators)
