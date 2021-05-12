import random

class RandomAssigner(object):
  def __init__(self):
    pass

  def PickElevator(self, cur_state):
    return random.choice(len(cur_state))
