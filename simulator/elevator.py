from enum import Enum
from simulator.rider import Rider

NUM_FLOORS = 6


class S(Enum):
     STOP = 1
     UP   = 2
     DOWN = 3


class Elevator(object):
  def __init__(self):
    self.running = S.STOP
    self.floor = 0
    self.riders = []
    # Committed stops to stop
    self.stops = [False for _ in range(NUM_FLOORS)]

  def commit(self, rider):
    # Commit to pick up the rider.
    self.stops[rider.start] = True
    if self.running == S.STOP:
      self.running = S.DOWN if rider.start <= self.floor else S.UP
    else:
      # Keep current running direction.
      pass

  def step(self):
    pass

  def state(self):
    return {
      'running': self.running,
      'floor': self.floor,
      'riders': self.riders,
      'stops': self.stops,
    }

  def __str__(self):
    s = self.state()
    # Turn rider list into readable string.
    s['riders'] = [str(r) for r in s['riders']]
    return str(s)


class Elevators(object):
  def __init__(self, num_of_elevators):
    self.elevators = [Elevator() for _ in range(num_of_elevators)]

  def state(self):
    return [e.state() for e in self.elevators]

  def commit(self, idx, rider):
    self.elevators[idx].commit(rider)

  def __str__(self):
    return '\n'.join([str(e) for e in self.elevators])
