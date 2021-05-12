class Rider(object):
  def __init__(self, dest):
    self.dest = dest

  def __str__(self):
    return 'd%d' % self.dest
