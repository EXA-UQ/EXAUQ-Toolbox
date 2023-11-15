"""Provides emulators for simulators."""
from __future__ import annotations

import dataclasses
import math
from collections.abc import Sequence
from numbers import Real
from typing import Optional, Union

import numpy as np
from mogp_emulator import GaussianProcess
from mogp_emulator.GPParams import GPParams

from exauq.core.modelling import AbstractEmulator, Input, Prediction, TrainingDatum
from exauq.core.numerics import equal_within_tolerance
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
        If keyword arguments are supplied upon initialisation that aren't
        supported by the initialiser of ``GaussianProcess`` from the
        mogp-emulator package.
    """

    def __init__(self, **kwargs):
        self._gp_kwargs = self._remove_entries(kwargs, "inputs", "targets")
        self._gp = self._make_gp(**self._gp_kwargs)
        self._training_data = TrainingDatum.list_from_arrays(
            self._gp.inputs, self._gp.targets
        )
        self.fit_hyperparameters = None

    @staticmethod
    def _remove_entries(_dict: dict, *args) -> dict:
        """Return a dict with the specified keys removed."""

        return {k: v for (k, v) in _dict.items() if k not in args}

    @staticmethod
    def _make_gp(**kwargs) -> GaussianProcess:
        """Create an mogp GaussianProcess from given kwargs, raising a
        RuntimeError if this fails.
        """

        try:
            return GaussianProcess([], [], **kwargs)

        except Exception:
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
        hyperparameter_bounds: Sequence[tuple[Optional[float], Optional[float]]] = None,
        hyperparameters=None,
    ) -> None:
        """Train the emulator, including estimation of hyperparameters.

        This method will train the underlying ``GaussianProcess``, as stored in
        the `gp` property, using the supplied training data. Hyperparameters are
        estimated as part of this training, by maximising the log-posterior.

        If bounds are supplied for the hyperparameters, then the estimated
        hyperparameters will respect these bounds (the underlying maximisation
        is constrained by the bounds). A bound that is set to ``None`` is
        treated as unconstrained. Upper bounds must be ``None`` or a positive
        number.

        Parameters
        ----------
        training_data : list[TrainingDatum]
            The pairs of inputs and simulator outputs on which the emulator
            should be trained.
        hyperparameter_bounds : sequence of tuple[Optional[float], Optional[float]], optional
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
        if hyperparameters is None:
            self._fit_gp_with_estimation(
                inputs, targets, hyperparameter_bounds=hyperparameter_bounds
            )
        elif self._gp_kwargs["nugget"] == "fit" and hyperparameters.nugget is None:
            raise ValueError(
                "The underlying MOGP GaussianProcess was created with 'nugget'='fit', "
                "but the nugget supplied during fitting is "
                f"{hyperparameters.nugget}, when it should instead be a float."
            )
        else:
            self._fit_gp_with_hyperparameters(inputs, targets, hyperparameters)

        self.fit_hyperparameters = MogpHyperparameters.from_mogp_gp(self._gp)
        self._training_data = training_data

        return None

    def _fit_gp_with_estimation(
        self,
        inputs,
        targets,
        hyperparameter_bounds: Optional[
            Sequence[tuple[Optional[float], Optional[float]]]
        ] = None,
    ) -> None:
        bounds = (
            self._compute_raw_param_bounds(hyperparameter_bounds)
            if hyperparameter_bounds is not None
            else None
        )
        self._gp = fit_GP_MAP(
            GaussianProcess(inputs, targets, **self._gp_kwargs), bounds=bounds
        )
        return None

    def _fit_gp_with_hyperparameters(
        self, inputs, targets, hyperparameters: MogpHyperparameters
    ) -> None:
        kwargs = self._gp_kwargs
        if hyperparameters.nugget is not None:
            kwargs["nugget"] = hyperparameters.nugget

        self._gp = GaussianProcess(inputs, targets, **kwargs)
        self._gp.fit(hyperparameters.to_mogp_gp_params(use_nugget=kwargs["nugget"]))
        return None

    # TODO: use static methods from MogpHyperparameters instead?
    @staticmethod
    def _compute_raw_param_bounds(
        bounds: Sequence[tuple[Optional[float], Optional[float]]]
    ) -> tuple[tuple[Optional[float], Optional[float]], ...]:
        """Compute raw parameter bounds from bounds on correlation length
        parameters and covariance.

        Raises a ValueError if one of the upper bounds is a non-positive number.

        For the definitions of the transformations from raw values, see:

        https://mogp-emulator.readthedocs.io/en/latest/implementation/GPParams.html#mogp_emulator.GPParams.GPParams
        """

        for _, upper in bounds:
            if upper is not None and upper <= 0:
                raise ValueError("Upper bounds must be positive numbers")

        # Note: we need to swap the order of the bounds for correlation, because
        # _raw_from_corr is a decreasing function (i.e. min of raw corresponds
        # to max of correlation and vice-versa).
        raw_bounds = [
            (MogpEmulator._raw_from_corr(bnd[1]), MogpEmulator._raw_from_corr(bnd[0]))
            for bnd in bounds[:-1]
        ] + [
            (
                MogpEmulator._raw_from_cov(bounds[-1][0]),
                MogpEmulator._raw_from_cov(bounds[-1][1]),
            )
        ]
        return tuple(raw_bounds)

    @staticmethod
    def _raw_from_corr(corr: Optional[float]) -> Optional[float]:
        """Compute a raw parameter from a correlation length parameter.

        See https://mogp-emulator.readthedocs.io/en/latest/implementation/GPParams.html#mogp_emulator.GPParams.GPParams
        """

        if corr is None or corr <= 0:
            return None

        return -2 * math.log(corr)

    @staticmethod
    def _raw_from_cov(cov: Optional[float]) -> Optional[float]:
        """Compute a raw parameter from a covariance parameter.

        See https://mogp-emulator.readthedocs.io/en/latest/implementation/GPParams.html#mogp_emulator.GPParams.GPParams
        """
        if cov is None or cov <= 0:
            return None

        return math.log(cov)

    def predict(self, x: Input) -> Prediction:
        """Make a prediction of a simulator output for a given input.

        Parameters
        ----------
        x : Input
            A simulator input.

        Returns
        -------
        Prediction
            The emulator's prediction of the simulator output from the given the input.

        Raises
        ------
        RuntimeError
            If this emulator has not been trained on any data before making the
            prediction.
        """

        if not isinstance(x, Input):
            raise TypeError(f"Expected 'x' to be of type Input, but received {type(x)}.")

        if len(self.training_data) == 0:
            raise RuntimeError(
                "Cannot make prediction because emulator has not been trained on any data."
            )

        if not len(x) == (expected_dim := self._get_input_dim()):
            raise ValueError(
                f"Expected 'x' to be an Input with {expected_dim} coordinates, but "
                f"it has {len(x)} instead."
            )

        return self._to_prediction(self.gp.predict(np.array(x)))

    def _get_input_dim(self) -> Optional[int]:
        """Get the dimension of the inputs in the training data. Note: assumes that
        each input in the training data has the same dimension."""
        try:
            return len(self.training_data[0].input)
        except IndexError:
            return None

    @staticmethod
    def _to_prediction(mogp_predict_result) -> Prediction:
        """Convert an MOGP ``PredictResult`` to a ``Prediction`` object.

        See https://mogp-emulator.readthedocs.io/en/latest/implementation/GaussianProcess.html#the-predictresult-class
        """
        return Prediction(
            estimate=mogp_predict_result.mean[0], variance=mogp_predict_result.unc[0]
        )


@dataclasses.dataclass()
class MogpHyperparameters:
    corr: Optional[np.array] = None
    cov: Optional[float] = None
    nugget: Optional[float] = None

    @classmethod
    def from_mogp_gp(cls, gp: GaussianProcess) -> MogpHyperparameters:
        return cls(
            corr=gp.theta.corr,
            cov=gp.theta.cov,
            nugget=gp.theta.nugget,
        )

    def __eq__(self, other: MogpHyperparameters) -> bool:
        try:
            nuggets_equal = (
                self.nugget is None and other.nugget is None
            ) or equal_within_tolerance(self.nugget, other.nugget)
        except TypeError:
            return False

        return all(
            [
                nuggets_equal,
                equal_within_tolerance(self.corr, other.corr),
                equal_within_tolerance(self.cov, other.cov),
            ]
        )

    def to_mogp_gp_params(self, use_nugget: Union[float, str] = "fit") -> GPParams:
        raw_params = [self.transform_corr(x) for x in self.corr] + [
            self.transform_cov(self.cov)
        ]

        if self.nugget is not None:
            params = GPParams(n_corr=len(self.corr), nugget=self.nugget)
        else:
            params = GPParams(n_corr=len(self.corr), nugget=use_nugget)

        params.set_data(np.array(raw_params, dtype=float))
        return params

    @staticmethod
    def transform_corr(corr: Real) -> float:
        """Compute a raw parameter from a correlation length parameter.

        See https://mogp-emulator.readthedocs.io/en/latest/implementation/GPParams.html#mogp_emulator.GPParams.GPParams
        """

        # N.B. This catches single-element Numpy arrays as well, for which coercion to
        # scalars is deprecated.
        if not isinstance(corr, Real):
            raise TypeError(
                f"Expected 'corr' to be a real number, but received {type(corr)}."
            )
        try:
            return -2 * math.log(corr)
        except ValueError:
            raise ValueError(
                f"'corr' must be a positive real number, but received {corr}."
            ) from None

    @staticmethod
    def transform_cov(cov: Real) -> float:
        """Compute a raw parameter from a covariance parameter.

        See https://mogp-emulator.readthedocs.io/en/latest/implementation/GPParams.html#mogp_emulator.GPParams.GPParams
        """

        # N.B. This catches single-element Numpy arrays as well, for which coercion to
        # scalars is deprecated.
        if not isinstance(cov, Real):
            raise TypeError(
                f"Expected 'cov' to be a real number, but received {type(cov)}."
            )

        try:
            return math.log(cov)
        except ValueError:
            raise ValueError(
                f"'cov' must be a positive real number, but received {cov}."
            ) from None

    @staticmethod
    def _raw_from_nugget(nugget: Optional[float]) -> Optional[float]:
        if nugget is None:
            return None

        return math.log(nugget)
