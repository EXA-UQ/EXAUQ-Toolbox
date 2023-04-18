"""Contains fakes used to support unit tests
"""
import typing
from exauq.core.modelling import(
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
    training_data: list[tuple[float, float]]
        A list of pairs of `float`s ``(x,y)`` on which the emulator has been
        fitted. Here, ``x`` is a simulator input and ``y`` is the corresponding
        simulator output.
    """
    def __init__(self):
        super()
        self._training_data: typing.Optional[list[tuple[float, float]]] = None

    @property
    def training_data(self) -> typing.Optional[list[tuple[float, float]]]:
        return super().training_data

    def fit(self, data: list[tuple[float, float]]) -> None:
        """Fits the emulator on the given data.

        Any prior training data are discarded, so that each
        call to `fit` effectively fits the emulator again from scratch.

        Parameters
        ----------
        data : list[tuple[float, float]]
            A list of pairs of `float`s ``(x,y)`` on which to fit the
            emulator. Here, ``x`` is a simulator input and ``y`` is the
            corresponding simulator output.
        """
        self._training_data = data
    
    def predict(self, x: float) -> float:
        """Estimate the simulator output for a given input.

        The emulator will predict the correct simulator output for `x` which
        feature in the training data. For new `x`, zero will be returned.

        Parameters
        ----------
        x : float
            An input to the simulator.

        Returns
        -------
        float
            The value predicted by the emulator, which is an estimate of the
            simulator's output at `x`.
        """
        for input, observation in self._training_data:
            if abs(input - x) < 1e-10:
                return observation

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

    def compute(self, x: float) -> float:
        """Evaluate the identity function at the given point.

        Parameters
        ----------
        x : float
            The input at which to evaluate the simulator.

        Returns
        -------
        float
            The given input, `x`.
        """
        return x
