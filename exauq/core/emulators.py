from collections.abc import Sequence
import math
from mogp_emulator import GaussianProcess
import numpy as np
from exauq.core.modelling import (
    TrainingDatum,
    AbstractEmulator)
from exauq.utilities.mogp_fitting import fit_GP_MAP


class MogpEmulator(AbstractEmulator):
    """
    An emulator wrapping a ``GaussianProcess`` object from the mogp-emulator
    package.

    This class allows mogp-emulator ``GaussianProcess`` objects to be used with
    the designers defined in the EXAUQ-Toolbox, ensuring the interface required
    by the designers is present. Keyword arguments supplied to the
    `MogpEmulator` are passed onto the ``GaussianProcess``
    initialiser to create the underlying (i.e. wrapped) ``GaussianProcess``
    object. Note that any ``inputs`` or ``targets`` supplied are ignored: the
    underlying ``GaussianProcess`` will initially be constructed with no
    training data.
    
    The underlying ``GaussianProcess` object can be obtained through the
    `gp` property. Note that the `fit` method, used to train the emulator, will
    modify the underlying ``GaussianProcess``.

    Parameters
    ----------
    **kwargs : dict, optional
        Any permitted keyword arguments that can be used to create a
        mogp-emulator ``GaussianProcess`` object. See the mogp-emulator
        documentation for details. If ``inputs`` or ``targets`` are supplied as
        keyword arguments then these will be ignored.
    
    Attributes
    ----------
    gp : mogp_emulator.GaussianProcess
        (Read-only) The underlying mogp-emulator ``GaussianProcess`` object
        constructed by this class.
    training_data: list[TrainingDatum] or None
        (Read-only) Defines the pairs of inputs and simulator outputs on which
        the emulator has been trained.
    
    Raises
    ------
    RuntimeError
        If 
    """
    def __init__(self, **kwargs):
        super()
        self._gp_kwargs = self._remove_entries(kwargs, 'inputs', 'targets')
        self._gp = self._make_gp(**self._gp_kwargs)
        self._training_data = TrainingDatum.list_from_arrays(
            self._gp.inputs, self._gp.targets
            )
    
    @staticmethod
    def _remove_entries(_dict: dict, *args) -> dict:
        """Return a dict with the specified keys removed.
        """

        return {k: v for (k, v) in _dict.items() if k not in args}

    @staticmethod
    def _make_gp(**kwargs) -> GaussianProcess:
        """Create an mogp GaussianProcess from given kwargs, raising a
        RuntimeError if this fails.
        """

        try:
            return GaussianProcess([], [], **kwargs)

        except BaseException:
            msg = (
                "Could not construct mogp-emulator GaussianProcess during "
                "initialisation of MogpEmulator"
                )
            raise RuntimeError(msg)

    @property
    def gp(self) -> GaussianProcess:
        """(Read-only) Get the underlying mogp GaussianProcess for this
        emulator."""
        return self._gp

    @property
    def training_data(self) -> list[TrainingDatum]:
        """(Read-only) Get the data on which the emulator has been trained."""
        return self._training_data
    
    def fit(
            self,
            training_data: list[TrainingDatum],
            hyperparameter_bounds : Sequence[tuple[float, float]] = None
            ) -> None:
        """Train the emulator, including estimation of hyperparameters.

        This method will train the underlying ``GaussianProcess``, as stored in
        the `gp` property, using the supplied training data. Hyperparameters are
        estimated as part of this training, by maximising the log-posterior. If
        bounds are supplied for the hyperparameters, then the estimated
        hyperparameters will respect these bounds (the underlying maximisation
        is constrained by the bounds).

        Parameters
        ----------
        training_data: list[TrainingDatum]
            The pairs of inputs and simulator outputs on which the emulator
            should been trained.
        hyperparameter_bounds : sequence of tuple[float, float], optional
            (Default: None) A sequence of bounds to apply to hyperparameters
            during estimation, of the form ``(lower_bound, upper_bound)``. All
            but the last tuple should represent bounds for the correlation
            length parameters, while the last tuple should represent bounds for
            the covariance.
        """
        
        if training_data is None or training_data == []:
            return

        inputs = np.array([datum.input.value for datum in training_data])
        targets = np.array([datum.output for datum in training_data])
        bounds = (
            self._compute_raw_param_bounds(hyperparameter_bounds)
            if hyperparameter_bounds is not None
            else None
            )
        self._gp = fit_GP_MAP(
            GaussianProcess(inputs, targets, **self._gp_kwargs), bounds=bounds
            )
        self._training_data = training_data
        
        return None

    @staticmethod
    def _compute_raw_param_bounds(
        bounds: Sequence[tuple[float, float]]
        ) -> tuple[tuple[float, ...]]:
        """Compute raw parameter bounds from bounds on correlation length
        parameters and covariance.
        
        For the definitions of the transformations from raw values, see:
        
        https://mogp-emulator.readthedocs.io/en/latest/implementation/GPParams.html#mogp_emulator.GPParams.GPParams
        """
        
        # Note: we need to swap the order of the bounds for correlation, because
        # _raw_from_corr is a decreasing function (i.e. min of raw corresponds
        # to max of correlation and vice-versa).
        raw_bounds = [
            (MogpEmulator._raw_from_corr(bnd[1]), MogpEmulator._raw_from_corr(bnd[0]))
            for bnd in bounds[:-1]
        ] + [
            (MogpEmulator._raw_from_cov(bounds[-1][0]), MogpEmulator._raw_from_cov(bounds[-1][1]))
        ]
        return tuple(raw_bounds)

    @staticmethod
    def _raw_from_corr(corr: float) -> float:
        """Compute a raw parameter from a correlation length parameter.

        See https://mogp-emulator.readthedocs.io/en/latest/implementation/GPParams.html#mogp_emulator.GPParams.GPParams
        """

        return -2 * math.log(corr)
    
    @staticmethod
    def _raw_from_cov(cov: float) -> float:
        """Compute a raw parameter from a covariance parameter.

        See https://mogp-emulator.readthedocs.io/en/latest/implementation/GPParams.html#mogp_emulator.GPParams.GPParams
        """

        return math.log(cov)
