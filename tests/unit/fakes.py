"""Contains fakes used to support unit tests
"""
import typing
from exauq.core.modelling import(
    Experiment,
    TrainingDatum,
    AbstractEmulator,
    AbstractSimulator
)

# The tolerance used for determining if two floating point numbers are equal.
TOLERANCE_PLACES: float = 7
TOLERANCE: float = 10 ** TOLERANCE_PLACES


class DumbEmulator(AbstractEmulator):
    """A concrete emulator for emulating 1-dimensional simulators.

    This emulator predicts zero at inputs on which it hasn't been fitted, while
    predicting observations on which it has been fitted correctly.
    

    Attributes
    ----------
    training_data: list[TrainingDatum] or None
        Defines the pairs of experiments and observations on which the emulator
        has been trained. Each `TrainingDatum` should have a 1-dim
        `Experiment`.
    """
    def __init__(self):
        super()
        self._training_data: typing.Optional[list[TrainingDatum]] = None

    @property
    def training_data(self) -> typing.Optional[list[TrainingDatum]]:
        """Get the data on which the emulator has been trained."""
        return super().training_data

    def fit(self, data: list[TrainingDatum]) -> None:
        """Fits the emulator on the given data.

        Any prior training data are discarded, so that each
        call to `fit` effectively fits the emulator again from scratch.

        Parameters
        ----------
        data : list[TrainingDatum]
            Defines the pairs of experiments and observations on which to train
            the emulator. Each `TrainingDatum` should have a 1-dim `Experiment`.
        """
        self._training_data = data
    
    def predict(self, x: Experiment) -> float:
        """Estimate the simulator output for a given input.

        The emulator will predict the correct simulator output for `x` which
        feature in the training data. For new `x`, zero will be returned.

        Parameters
        ----------
        x : Experiment
            An input to the simulator.

        Returns
        -------
        float
            The value predicted by the emulator, which is an estimate of the
            simulator's output at `x`.
        """
        for datum in self._training_data:
            if abs(datum.experiment.value - x.value) < 1e-10:
                return datum.observation

        return 0


class OneDimSimulator(AbstractSimulator):
    """A basic simulator defined on a one-dimensional domain.

    This simulator simply defines the identity function ``f(x) = x`` and is
    defined on a closed interval [a, b].
    
    Parameters
    ----------
    lower_limit: float
        The lower limit of the domain on which the simulator is defined.
    upper_limit: float
        The upper limit of the domain on which the simulator is defined.

    Attributes
    ----------
    domain: OneDimDomain
        The domain on which the simulator is defined.
    """ 
    def __init__(self, lower_limit: float, upper_limit: float):
        self.lower_limit: float = lower_limit
        self.upper_limit: float = upper_limit

    def compute(self, x: Experiment) -> float:
        """Evaluate the identity function at the given point.

        Parameters
        ----------
        x : Experiment
            The input at which to evaluate the simulator.

        Returns
        -------
        float
            The value of the input `x`.
        """
        return x.value
