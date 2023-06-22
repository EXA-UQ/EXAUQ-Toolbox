import unittest
import tests.unit.fakes as fakes
from numbers import Real
from tests.utilities.utilities import exact
from exauq.core.designers import SingleLevelAdaptiveSampler
from exauq.core.modelling import Input, TrainingDatum


class TestSingleLevelAdaptiveSampler(unittest.TestCase):
    def setUp(self) -> None:
        self.datum = TrainingDatum(Input(0), 1)
        self.initial_data = [self.datum]
        self.designer = SingleLevelAdaptiveSampler(self.initial_data)
        self.emulator = fakes.DumbEmulator()

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

        expected = (
            "SingleLevelAdaptiveSampler designer with initial data "
            f"{str(self.initial_data)}"
        )
        self.assertEqual(expected, str(self.designer))

    def test_repr(self):
        """Test that the string representation of an instance of
        SingleLevelAdaptiveSampler designer is derived from its constituent
        parts."""

        expected = f"SingleLevelAdaptiveSampler(initial_data={repr(self.initial_data)})"
        self.assertEqual(expected, repr(self.designer))

    def test_train(self):
        """Test that training an emulator with the SLAS designer returns an
        emulator of the same type."""

        trained_emulator = self.designer.train(self.emulator)

        self.assertIsInstance(trained_emulator, type(self.emulator))

    def test_train_fits_with_initial_design(self):
        """Test that the emulator returned by the SLAS designer has been trained
        on initial data."""

        initial_design = [
            TrainingDatum(Input(0.2), 0.2),
            TrainingDatum(Input(0.55), 0.55),
        ]
        designer = SingleLevelAdaptiveSampler(initial_design)

        trained_emulator = designer.train(self.emulator)

        self.assertEqual(initial_design, trained_emulator.training_data)

    def test_train_returns_new_emulator(self):
        """Test that training an emulator returns a new emulator object,
        leaving the original unchanged."""

        trained_emulator = self.designer.train(self.emulator)

        self.assertNotEqual(self.emulator, trained_emulator)
        self.assertIsNone(self.emulator.training_data)

    def test_new_design_batch_default(self):
        """Test that a list with a single Input is returned for a default batch."""

        batch = self.designer.new_design_batch(self.emulator)

        self.assertIsInstance(batch, list)
        self.assertEqual(1, len(batch))
        self.assertIsInstance(batch[0], Input)

    def test_new_design_batch_number(self):
        """Test that a list of the correct number of inputs is returned when batch
        number is specified."""

        size = 2

        batch = self.designer.new_design_batch(self.emulator, size=size)

        self.assertIsInstance(batch, list)
        self.assertEqual(size, len(batch))
        for _input in batch:
            self.assertIsInstance(_input, Input)

    def test_new_design_batch_calculates_esloo_errors(self):
        """Test that one ES-LOO error is computed for each initial design point from the
        supplied emulator, and that ES-LOO errors are real numbers."""

        self.emulator.fit([TrainingDatum(Input(0), 1), TrainingDatum(Input(1), 1)])

        self.assertIsNone(self.designer.esloo_errors)

        self.designer.new_design_batch(self.emulator)

        self.assertEqual(
            len(self.emulator.training_data),
            len(self.designer.esloo_errors)
        )
        for error in self.designer.esloo_errors:
            self.assertIsInstance(error, Real)


if __name__ == "__main__":
    unittest.main()
