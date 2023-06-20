import unittest
import tests.unit.fakes as fakes
from tests.utilities.utilities import exact
from exauq.core.designers import SingleLevelAdaptiveSampler
from exauq.core.modelling import Input, TrainingDatum


class TestSingleLevelAdaptiveSampler(unittest.TestCase):
    def setUp(self) -> None:
        self.datum = TrainingDatum(Input(0), 1)
        self.initial_data = [self.datum]

    def test_training_data_wrong_type_error(self):
        """Test that a TypeError is raised if the SLAS designer is initialised with
        something other than a collection of training data."""

        msg = (
            f"{SingleLevelAdaptiveSampler.__name__} must be initialised with a "
            "(finite) collection of TrainingDatum"
        )

        # Mock a stream of unsizeable data (note it doesn't implement __len__ so
        # doesn't define a collection).
        def unsizeable_data():
            for i in range(1000):
                yield self.datum

        for data in [None, 1, TrainingDatum(Input(0), 1), ['foo'], unsizeable_data()]:
            with self.subTest(data=data):
                with self.assertRaisesRegex(TypeError, exact(msg)):
                    SingleLevelAdaptiveSampler(data)

    def test_training_data_empty_error(self):
        """Test that a ValueError is raised if the SLAS designer is initialised with
        an empty list of training data."""

        msg = "'initial_data' must be nonempty"
        with self.assertRaisesRegex(ValueError, exact(msg)):
            SingleLevelAdaptiveSampler([])

    def test_training_data(self):
        """Test that a SingleLevelAdaptiveSampler can be initialised from different
        collections of TrainingDatum."""

        for data in [[self.datum], tuple([self.datum])]:
            with self.subTest(data=data):
                SingleLevelAdaptiveSampler(data)

    def test_str(self):
        """Test that the string description of an instance of
        SingleLevelAdaptiveSampler designer is derived from its constituent
        parts."""

        designer = SingleLevelAdaptiveSampler(self.initial_data)

        expected = (
            "SingleLevelAdaptiveSampler designer with initial data "
            f"{str(self.initial_data)}"
        )
        self.assertEqual(expected, str(designer))

    def test_repr(self):
        """Test that the string representation of an instance of
        SingleLevelAdaptiveSampler designer is derived from its constituent
        parts."""

        designer = SingleLevelAdaptiveSampler(self.initial_data)

        expected = f"SingleLevelAdaptiveSampler(initial_data={repr(self.initial_data)})"
        self.assertEqual(expected, repr(designer))

    def test_train(self):
        """Test that training an emulator with the SLAS designer returns an
        emulator of the same type."""

        designer = SingleLevelAdaptiveSampler(self.initial_data)
        emulator = fakes.DumbEmulator()

        trained_emulator = designer.train(emulator)

        self.assertIsInstance(trained_emulator, type(emulator))

    def test_train_fits_with_initial_design(self):
        """Test that the emulator returned by the SLAS designer has been trained
        on initial data."""

        emulator = fakes.DumbEmulator()
        initial_design = [
            TrainingDatum(Input(0.2), 0.2),
            TrainingDatum(Input(0.55), 0.55),
        ]
        designer = SingleLevelAdaptiveSampler(initial_design)

        trained_emulator = designer.train(emulator)

        self.assertEqual(initial_design, trained_emulator.training_data)

    def test_train_returns_new_emulator(self):
        """Test that training an emulator returns a new emulator object,
        leaving the original unchanged."""

        emulator = fakes.DumbEmulator()
        designer = SingleLevelAdaptiveSampler(self.initial_data)

        trained_emulator = designer.train(emulator)

        self.assertNotEqual(emulator, trained_emulator)
        self.assertIsNone(emulator.training_data)


if __name__ == "__main__":
    unittest.main()
