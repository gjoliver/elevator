# Hyper parameters used throughout the program.
# Defined with defaults.
class HyperParams(object):
  def __init__(self,
               nn_sizes=None,
               num_elevators=4,
               num_floors=6,
               training_iterations=10,
               batch_size=128,
               gamma=0.9,
               learning_rate=0.001):
    self.nn_sizes = nn_sizes
    self.num_elevators = num_elevators
    self.num_floors = num_floors
    self.training_iterations = training_iterations
    self.batch_size = batch_size
    self.gamma = gamma
    self.learning_rate = learning_rate
