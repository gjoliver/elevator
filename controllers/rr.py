class RoundRobinAssigner(object):
  def __init__(self, cfg):
    del cfg  # not used
    self.choice = 0

  def Pick(self, state, rider):
    choice = self.choice
    self.choice = (self.choice + 1) % len(state)
    return choice
