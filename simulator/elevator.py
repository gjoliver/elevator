from enum import Enum
from simulator.rider import Rider

NUM_FLOORS = 6


class S(Enum):
     STOP = 1
     UP   = 2
     DOWN = 3


class Elevator(object):
  def __init__(self):
    self.state = S.STOP
    self.floor = 1
    self.riders = [Rider(3)]
    self.stops = []

  def commit(floor):
    # TODO, kinda complicated.
    pass

  def cur_state(self):
    return {
      'state': self.state,
      'floor': self.floor,
      'riders': self.riders,
      'stops': self.stops,
    }

  def __str__(self):
    s = self.cur_state()
    # Turn rider list into readable string.
    s['riders'] = [str(r) for r in s['riders']]
    return str(s)


class Elevators(object):
  def __init__(self, num_of_elevators):
    self.elevators = [Elevator() for _ in range(num_of_elevators)]

  def cur_state(self):
    return [e.cur_state() for e in self.elevators]

  def commit(self, idx, floor):
    self.elevators[idx].commit(floor)

  def __str__(self):
    return '\n'.join([str(e) for e in self.elevators])
