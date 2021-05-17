from operator import itemgetter
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
    self.floor = 0    # Current floor.
    self.riders = []  # List of riders in this elevator.
    # Committed stops.
    # Stops are a list of 1 or 2 tuples. The first elements of the tuples
    # are stops that the elevator needs to visit immediately.
    # The second elements, if there are any, are the stops that it will
    # need to visit after their corresponding first stops are satisfied.
    # We use this scheme because drop off stops are only useful if their
    # corresponding pick up stops are satisfied.
    self.stops = []
    # Riders waiting for this elevator at each floor.
    self.waiting = [[] for _ in range(NUM_FLOORS)]

  def _consolidate_stops(self):
    self.stops = list(set(self.stops))
    # Sort by immediate stops.
    self.stops.sort(key=itemgetter(0))

  def commit(self, rider):
    # Commit to pick up the rider.
    self.stops.append((rider.pickup, rider.dropoff))
    self._consolidate_stops()

    # Move the rider to the waiting queue of this elevator for
    # the start floor.
    self.waiting[rider.pickup].append(rider)

  def _update_riders(self):
    new_riders = []
    for r in self.riders:
      # Drop off riders for current floor. Keep the ones that haven't
      # reached their dropoff floors.
      if r.dropoff != self.floor:
        new_riders.append(r)
    # Pick up all the riders waiting at this floor.
    new_riders = new_riders + self.waiting[self.floor]
    # Clear waiting queue at current floor.
    self.waiting[self.floor] = []
    # new_riders are the update list of riders.
    self.riders = new_riders

  def _update_stops(self):
    new_stops = []
    for s in self.stops:
      if s[0] != self.floor:
        new_stops.append(s)
      elif len(s) == 2:
        # Pickup stop satisfied. Queue the dropoff stop as immediate
        # stop to visit.
        new_stops.append((s[1],))
      # else, just drop this dropoff stop that has just been satisfied.
    # Set new stops and consolidate.
    self.stops = new_stops
    self._consolidate_stops()

  def _go(self):
    # Travel to next floor.
    if self.running == S.STOP:
      pass
    elif self.running == S.UP:
      assert self.floor < NUM_FLOORS - 1, 'already at top'
      self.floor += 1
    elif self.running == S.DOWN:
      assert self.floor > 0, 'already at ground floor'
      self.floor -= 1
    else:
      assert False, 'what?'

  def _next_stop_below(self):
    """next committed stop below current floor.
       -1 if no commited floor below.
    """
    # stops are sorted in increasing order.
    below = -1
    for stop in self.stops:
      if stop[0] < self.floor:
        # Higher stops below current floor will overwrite lower stops.
        below = stop[0]
    return below

  def _next_stop_above(self):
    """next committed stop aboev current floor.
       -1 if no commited floor above.
    """
    above = -1
    for stop in reversed(self.stops):
      if stop[0] > self.floor:
        # Lower stops above current floor will overwrite higer stops.
        above = stop[0]
    return above

  def _update_running(self):
    if self.running == S.DOWN:
      # Lower stops have priority.
      if self._next_stop_below() >= 0:
        self.running = S.DOWN
        return
      elif self._next_stop_above() >= 0:
        self.running = S.UP
        return
    elif self.running == S.UP:
      # Higher stops have priority.
      if self._next_stop_above() >= 0:
        self.running = S.UP
        return
      elif self._next_stop_below() >= 0:
        self.running = S.DOWN
        return
    elif self.running == S.STOP:
      # Go to whichever stop is closer.
      stop_below = self._next_stop_below()
      stop_above = self._next_stop_above()
      if stop_below >= 0 and stop_above < 0:
        self.running = S.DOWN
        return
      elif stop_below < 0 and stop_above >= 0:
        self.running = S.UP
        return
      elif stop_below >= 0 and stop_above >= 0:
        self.running = (
          S.DOWN
          if abs(stop_below - self.floor) <= abs(stop_above - self.floor)
          else S.UP)
        return
    # Otherwise, rest at current floor.
    self.running = S.STOP

  def step(self):
    self._go()
    self._update_riders()
    self._update_stops()
    self._update_running()

  def state(self):
    return {
      'running': self.running,
      'floor': self.floor,
      'riders': self.riders,
      'waiting': self.waiting,
      'stops': self.stops,
    }

  def __str__(self):
    s = self.state()
    # Turn rider list into readable string.
    s['riders'] = [str(r) for r in s['riders']]
    s['waiting'] = [[str(r) for r in w] for w in s['waiting']]
    return str(s)


class Elevators(object):
  def __init__(self, num_of_elevators):
    self.elevators = [Elevator() for _ in range(num_of_elevators)]

  def state(self):
    return [e.state() for e in self.elevators]

  def step(self):
    for e in self.elevators:
      e.step()

  def commit(self, idx, rider):
    self.elevators[idx].commit(rider)

  def __str__(self):
    return '\n'.join([str(e) for e in self.elevators])
