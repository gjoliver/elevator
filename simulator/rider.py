class Rider(object):
  def __init__(self, start, dest):
    self.start = start
    self.dest = dest

  def __str__(self):
    return f'{self.start}->{self.dest}'
