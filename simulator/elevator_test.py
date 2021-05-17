import unittest

from simulator.elevator import S, Elevator
from simulator.rider import Rider


class TestElevator(unittest.TestCase):
  def test_elevator(self):
    e = Elevator()
    e.commit(Rider(0, 3))
    self.assertEqual(e.stops, [(0, 3)])

    e.commit(Rider(0, 3)) # Dup
    self.assertEqual(e.stops, [(0, 3)])  # De-dupped.

    e.commit(Rider(0, 5))
    self.assertEqual(e.stops, [(0, 3), (0, 5)])

    e.commit(Rider(3, 0))
    self.assertEqual(e.stops, [(0, 3), (0, 5), (3, 0)])

  def test_stop_below(self):
    e = Elevator()
    e.stops = [(0,), (0, 5), (2,)]
    e.floor = 3
    self.assertEqual(e._next_stop_below(), 2)

    e.stops = [(2,), (2, 5), (5,)]
    e.floor = 0
    self.assertEqual(e._next_stop_below(), -1)

  def test_stop_above(self):
    e = Elevator()
    e.stops = [(0,), (0, 5), (2,)]
    e.floor = 1
    self.assertEqual(e._next_stop_above(), 2)

    e.stops = [(2,), (2, 5), (5,)]
    e.floor = 5
    self.assertEqual(e._next_stop_above(), -1)


if __name__ == '__main__':
  unittest.main()
