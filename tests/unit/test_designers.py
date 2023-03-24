import unittest
import tests.unit.fakes as fakes
from exauq.core.designers import SingleLevelAdaptiveSampler

class TestSingleLevelAdaptiveSampler(unittest.TestCase):
    def test_designer_str(self):
        """Test that the string description of an instance of
        SingleLevelAdaptiveSampling designer is derived from its constituent
        parts."""

        simulator = fakes.OneDimSimulator(0, 1)
        emulator = fakes.DumbEmulator()
        designer = SingleLevelAdaptiveSampler(emulator, simulator)

        expected = f"SingleLevelAdaptiveSampling designer for simulator {str(simulator)}, " \
                   f"using emulator {str(emulator)}"
        self.assertEqual(expected, str(designer))

    def test_designer_repr(self):
        """Test that the string representation of an instance of
        SingleLevelAdaptiveSampling designer is derived from its constituent
        parts."""

        simulator = fakes.OneDimSimulator(0, 1)
        emulator = fakes.DumbEmulator()
        designer = SingleLevelAdaptiveSampler(emulator, simulator)

        expected = f"SingleLevelAdaptiveSampling(simulator={repr(simulator)}, " \
                   f"emulator={repr(emulator)})"
        self.assertEqual(expected, repr(designer))


if __name__ == "__main__":
    unittest.main()
