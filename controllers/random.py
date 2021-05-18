import random

class RandomAssigner(object):
  def __init__(self, cfg):
    del cfg  # not used
    pass

  def Pick(self, state, rider):
    del rider  # not used.
    return random.choice(range(len(state)))
