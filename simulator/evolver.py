class Evolver(object):
  def __init__(self, elevators, controller):
    self.elevators = elevators
    self.controller = controller

  def evolve(self):
    while True:
      # TODO
      # Generate random guests.
      # Ask controller which elevator should commit.
      # Control elevator.
      # Go to next step
      input('')

      cur_state = self.elevators.cur_state()
      picked = self.controller.PickElevator(cur_state)
      self.elevators.commit(picked, rider.dest)

      print(self.elevators)
