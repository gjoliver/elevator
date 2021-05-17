class Rider(object):
  def __init__(self, pickup, dropoff):
    self.pickup = pickup
    self.dropoff = dropoff

  def __str__(self):
    return f'{self.pickup}->{self.dropoff}'
