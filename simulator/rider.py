class Rider(object):
  def __init__(self, env, pickup, dropoff):
    self._env = env
    self._start_time = env.timer

    self.pickup = pickup
    self.dropoff = dropoff

  def submit_stats(self):
    stats = self._env.stats
    stats.num_riders += 1
    floors_served = abs(self.dropoff - self.pickup)
    stats.num_floors_served += floors_served
    ticks = self._env.timer - self._start_time
    stats.num_excessive_ticks += ticks - floors_served

  def __del__(self):
    self.submit_stats()

  def __str__(self):
    return f'{self.pickup}->{self.dropoff}'
