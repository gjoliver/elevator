class Stats(object):
  num_riders = 0
  num_floors_served = 0
  num_excessive_ticks = 0

  def __str__(self):
    efficiency = self.num_floors_served / (
      self.num_floors_served + self.num_excessive_ticks)
    return f'''
num_riders: {self.num_riders}
num_floors_served: {self.num_floors_served}
num_excessive_ticks: {self.num_excessive_ticks}
efficiency: {efficiency}
'''


class Env(object):
  def __init__(self):
    self.timer = 0
    self.stats = Stats()

  def tick(self):
    self.timer += 1
