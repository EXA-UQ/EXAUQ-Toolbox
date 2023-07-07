import unittest

import tests.unit.fakes as fakes
from exauq.core.designers import RandomSamplerDesigner, SingleLevelAdaptiveSampler
from exauq.core.modelling import Input, TrainingDatum
from tests.utilities.utilities import exact


class TestRandomSamplerDesigner(unittest.TestCase):
    def test_size_type_error(self):
        """Test that a TypeError is raised if something other than an int is provided
        as the size."""

        designer = RandomSamplerDesigner()
        with self.assertRaisesRegex(
            TypeError, exact("Argument 'size' must be of type 'int'.")
        ):
            designer.new_design_points(2.3)

    def test_new_design_points_return_list_length(self):
        """Test that a list of the required size is returned."""

        designer = RandomSamplerDesigner()
        for size in range(0, 3):
            design_points = designer.new_design_points(size)
            self.assertIsInstance(design_points, list)
            self.assertEqual(size, len(design_points))


class TestSingleLevelAdaptiveSampler(unittest.TestCase):
    initial_design = [Input(0.2), Input(0.55)]

    def test_str(self):
        """Test that the string description of an instance of
        SingleLevelAdaptiveSampler designer is derived from its constituent
        parts."""

        designer = SingleLevelAdaptiveSampler(self.initial_design)

        expected = f"SingleLevelAdaptiveSampler designer with initial design {str(self.initial_design)}"
        self.assertEqual(expected, str(designer))

    def test_repr(self):
        """Test that the string representation of an instance of
        SingleLevelAdaptiveSampler designer is derived from its constituent
        parts."""

        designer = SingleLevelAdaptiveSampler(self.initial_design)

        expected = (
            f"SingleLevelAdaptiveSampler(initial_design={repr(self.initial_design)})"
        )
        self.assertEqual(expected, repr(designer))

    def test_train(self):
        """Test that training an emulator with the SLAS designer returns an
        emulator of the same type."""

        designer = SingleLevelAdaptiveSampler(self.initial_design)
        simulator = fakes.OneDimSimulator(0, 1)
        emulator = fakes.DumbEmulator()

        trained_emulator = designer.train(emulator, simulator)

        self.assertIsInstance(trained_emulator, type(emulator))

    def test_train_fits_with_initial_design(self):
        """Test that the emulator returned by the SLAS designer has been trained
        on an initial design."""

        simulator = fakes.OneDimSimulator(0, 1)
        emulator = fakes.DumbEmulator()
        initial_design = [Input(0.2), Input(0.55)]

        designer = SingleLevelAdaptiveSampler(initial_design)

        trained_emulator = designer.train(emulator, simulator)
        expected = [TrainingDatum(Input(0.2), 0.2), TrainingDatum(Input(0.55), 0.55)]
        self.assertEqual(expected, trained_emulator.training_data[0:2])

    def test_train_returns_new_emulator(self):
        """Test that training an emulator returns a new emulator object,
        leaving the original unchanged."""

        simulator = fakes.OneDimSimulator(0, 1)
        emulator = fakes.DumbEmulator()
        designer = SingleLevelAdaptiveSampler(self.initial_design)

        trained_emulator = designer.train(emulator, simulator)
        self.assertNotEqual(emulator, trained_emulator)
        self.assertIsNone(emulator.training_data)


if __name__ == "__main__":
    unittest.main()
