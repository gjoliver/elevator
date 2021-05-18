import math


class DedicatedAssigner(object):
  def __init__(self, cfg):
    self._cfg = cfg

  def Pick(self, state, rider):
    num_elev = len(state)
    num_floors = self._cfg['floors']

    # Basically, we are breaking all the floors into num_elev buckets.
    # Each bucket is of size (num_floors / num_elev).
    bucket = math.floor(num_floors / num_elev)
    # Except for the first elevator, which will handle
    # (num_floors / num_elev + num_floors % num_elev) floors.
    r = num_floors % num_elev
    if rider.dropoff < r:
      return 0

    return math.floor((rider.dropoff - r) / bucket)
