import unittest

from simulator.elevator import S, Elevator
from simulator.rider import Rider

class TestElevator(unittest.TestCase):
  def test_elevator(self):
    e = Elevator()
    e.commit(Rider(0, 3))
    self.assertEqual(e.running, S.DOWN)
    self.assertEqual(e.stops, [True, False, False, False, False, False])


if __name__ == '__main__':
  unittest.main()
