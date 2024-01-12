"""Provides emulators for simulators."""
from __future__ import annotations

import dataclasses
import functools
import math
from collections.abc import Collection, Sequence
from numbers import Real
from typing import Literal, Optional, Union

import numpy as np
from mogp_emulator import GaussianProcess
from mogp_emulator.GPParams import GPParams

from exauq.core.modelling import (
    AbstractGaussianProcess,
    AbstractHyperparameters,
    Input,
    OptionalFloatPairs,
    Prediction,
    TrainingDatum,
)
from exauq.core.numerics import equal_within_tolerance
from exauq.utilities.mogp_fitting import fit_GP_MAP
from exauq.utilities.validation import check_real


class MogpEmulator(AbstractGaussianProcess):
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
    training_data: tuple[TrainingDatum]
        (Read-only) Defines the pairs of inputs and simulator outputs on which
        the emulator has been trained.
    fit_hyperparameters : MogpHyperparameters or None
        (Read-only) The hyperparameters of the underlying fitted Gaussian
        process model, or ``None`` if the model has not been fit to data.

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

        # Add the default nugget type if not provided explicitly
        if "nugget" not in self._gp_kwargs:
            self._gp_kwargs["nugget"] = self._gp.nugget_type

        self._training_data = tuple(
            TrainingDatum.list_from_arrays(self._gp.inputs, self._gp.targets)
        )
        self._fit_hyperparameters = None

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
        """(Read-only) The underlying mogp GaussianProcess for this
        emulator."""

        return self._gp

    @property
    def training_data(self) -> tuple[TrainingDatum]:
        """(Read-only) The data on which the emulator has been trained."""

        return self._training_data

    @property
    def fit_hyperparameters(self) -> Optional[MogpHyperparameters]:
        """(Read-only) The hyperparameters of the underlying fitted Gaussian
        process model, or ``None`` if the model has not been fitted to data."""

        return self._fit_hyperparameters

    def fit(
        self,
        training_data: Collection[TrainingDatum],
        hyperparameters: Optional[MogpHyperparameters] = None,
        hyperparameter_bounds: Optional[Sequence[OptionalFloatPairs]] = None,
    ) -> None:
        """Fit the emulator to data.

        This method trains the underlying ``GaussianProcess``, as stored in
        the `gp` property, using the supplied training data. By default,
        hyperparameters are estimated as part of this training, by maximising the
        log-posterior. Alternatively, a collection of hyperparamters can be supplied to
        use directly as the fitted values. (If the nugget is not supplied as part of these
        values, then it will be calculated according to the 'nugget' argument used in the
        construction of the underlying ``GaussianProcess``.)

        If bounds are supplied for the hyperparameters, then fitting with hyperparameter
        estimation will respect these bounds (i.e. the underlying log-posterior
        maximisation will be constrained by the bounds). A bound that is set to ``None``
        is treated as unconstrained; additionally, upper bounds must be ``None`` or a
        positive number. Note that the bounds are ignored if fitting with specific
        hyperparameters.

        Parameters
        ----------
        training_data : collection of TrainingDatum
            The pairs of inputs and simulator outputs on which the emulator
            should be trained. Should be a finite collection of such pairs.
        hyperparameters : MogpHyperparameters, optional
            (Default: None) Hyperparameters to use directly in fitting the Gaussian
            process. If ``None`` then the hyperparameters will be estimated as part of
            fitting to data.
        hyperparameter_bounds : sequence of tuple[Optional[float], Optional[float]], optional
            (Default: None) A sequence of bounds to apply to hyperparameters
            during estimation, of the form ``(lower_bound, upper_bound)``. All
            but the last tuple should represent bounds for the correlation
            length parameters, while the last tuple should represent bounds for
            the covariance.

        Raises
        ------

        ValueError
            If `hyperparameters` is provided with nugget being ``None`` but `self.gp`
            was created with nugget fitting method 'fit'.
        """

        training_data = self._parse_training_data(training_data)
        if not training_data:
            return None

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

        self._fit_hyperparameters = MogpHyperparameters.from_mogp_gp_params(
            self._gp.theta
        )
        self._training_data = training_data

        return None

    @staticmethod
    def _parse_training_data(training_data) -> tuple[TrainingDatum]:
        if training_data is None:
            return tuple()

        try:
            _ = len(training_data)  # to catch infinite iterators
            if not all(isinstance(x, TrainingDatum) for x in training_data):
                raise TypeError

            return tuple(training_data)

        except TypeError:
            raise TypeError(
                f"Expected a finite collection of TrainingDatum, but received {type(training_data)}."
            )

    def _fit_gp_with_estimation(
        self,
        inputs: np.ndarray,
        targets: np.ndarray,
        hyperparameter_bounds: Optional[
            Sequence[tuple[Optional[float], Optional[float]]]
        ] = None,
    ) -> None:
        """Fit the underlying GaussianProcess object to data with hyperparameter
        estimation, optionally with bounds applied to the hyperparameter estimation."""

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
        self,
        inputs: np.ndarray,
        targets: np.ndarray,
        hyperparameters: MogpHyperparameters,
    ) -> None:
        """Fit the underlying GaussianProcess object to data using specific
        hyperparameters."""

        kwargs = self._gp_kwargs
        nugget_type = "fixed"
        _hyperparameters = hyperparameters

        # Fit using supplied nugget hyperparameter if available...
        if hyperparameters.nugget is not None:
            kwargs["nugget"] = hyperparameters.nugget

        # ... Otherwise use the nugget given at GP construction, if a real number...
        elif isinstance(kwargs["nugget"], Real):
            _hyperparameters = MogpHyperparameters(
                hyperparameters.corr, hyperparameters.cov, kwargs["nugget"]
            )

        # ... Otherwise use the nugget calculation method given at GP construction.
        else:
            nugget_type = kwargs["nugget"]

        self._gp = GaussianProcess(inputs, targets, **kwargs)
        self._gp.fit(_hyperparameters.to_mogp_gp_params(nugget_type=nugget_type))

        return None

    @staticmethod
    def _compute_raw_param_bounds(
        bounds: Sequence[OptionalFloatPairs],
    ) -> tuple[OptionalFloatPairs, ...]:
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
            (
                MogpHyperparameters.transform_corr(bnd[1]),
                MogpHyperparameters.transform_corr(bnd[0]),
            )
            for bnd in bounds[:-1]
        ] + [
            (
                MogpHyperparameters.transform_cov(bounds[-1][0]),
                MogpHyperparameters.transform_cov(bounds[-1][1]),
            )
        ]
        return tuple(raw_bounds)

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


def _validate_nonnegative_real_domain(arg_name: str):
    """A decorator to be applied to functions with a single real-valued argument called
    `arg_name`. The decorator adds validation that the argument is a real number >= 0."""

    def decorator(func):
        @functools.wraps(func)
        def wrapped(arg: Real):
            # N.B. Not using try-except here because that would allow single-element Numpy
            # arrays to pass through with deprecation warning.
            if not isinstance(arg, Real):
                raise TypeError(
                    f"Expected '{arg_name}' to be a real number, but received {type(arg)}."
                )

            if arg < 0:
                raise ValueError(f"'{arg_name}' cannot be < 0, but received {arg}.")

            return func(arg)

        return wrapped

    return decorator


@dataclasses.dataclass(frozen=True)
class MogpHyperparameters(AbstractHyperparameters):
    """Hyperparameters for use in fitting Gaussian processes via `MogpEmulator`.

    This provides a simplified interface to parameters used in
    ``mogp_emulator.GaussianProcess`` objects and is comparable to the
    ``mogp_emulator.GPParams.GPParams`` class. The correlation length scale parameters,
    covariance and nugget described below are on the 'transformed' (linear) scale rather
    than the log scale; cf.
    https://mogp-emulator.readthedocs.io/en/latest/implementation/GPParams.html#mogp_emulator.GPParams.GPParams

    Equality of `MogpHyperparameters` objects is tested hyperparameter-wise up to the
    default numerical precision defined in ``exauq.core.numerics.FLOAT_TOLERANCE``
    (see ``exauq.core.numerics.equal_within_tolerance``).

    Parameters
    ----------
    corr : sequence or Numpy array of numbers.Real
        The correlation length scale parameters. The length of the sequence or array
        should equal the number of input coordinates for an emulator and each scale
        parameter should be a positive.
    cov : numbers.Real
        The covariance, which should be positive.
    nugget : numbers.Real, optional
        (Default: None) A nugget, which should be non-negative if provided.

    Attributes
    ----------
    corr : sequence or Numpy array of numbers.Real
        (Read-only) The correlation length scale parameters.
    cov : numbers.Real
        (Read-only) The covariance.
    nugget : numbers.Real, optional
        (Read only, default: None) The nugget, or ``None`` if not supplied.
    """

    corr: Union[Sequence[Real], np.ndarray[Real]]
    cov: Real
    nugget: Optional[Real] = None

    def __post_init__(self):
        if not isinstance(self.corr, (Sequence, np.ndarray)):
            raise TypeError(
                f"Expected 'corr' to be a sequence or array, but received {type(self.corr)}."
            )

        nonpositive_corrs = [x for x in self.corr if not isinstance(x, Real) or x <= 0]
        if nonpositive_corrs:
            nonpositive_element = nonpositive_corrs[0]
            raise ValueError(
                "Expected 'corr' to be a sequence or array of positive real numbers, "
                f"but found element {nonpositive_element} of type {type(nonpositive_element)}."
            )

        check_real(
            self.cov,
            TypeError(
                f"Expected 'cov' to be a real number, but received {type(self.cov)}."
            ),
        )
        if self.cov <= 0:
            raise ValueError(
                f"Expected 'cov' to be a positive real number, but received {self.cov}."
            )

        if self.nugget is not None:
            if not isinstance(self.nugget, Real):
                raise TypeError(
                    f"Expected 'nugget' to be a real number, but received {type(self.nugget)}."
                )

            if self.nugget < 0:
                raise ValueError(
                    f"Expected 'nugget' to be a positive real number, but received {self.nugget}."
                )

    @classmethod
    def from_mogp_gp_params(cls, params: GPParams) -> MogpHyperparameters:
        """Create an instance of MogpHyperparameters from an
        ``mogp_emulator.GPParams.GPParams`` object.

        Parameters
        ----------
        params : mogp_emulator.GPParams.GPParams
            A parameters object from mogp-emulator.

        Returns
        -------
        MogpHyperparameters
            The hyperparameters extracted from the given `params`.
        """

        if not isinstance(params, GPParams):
            raise TypeError(
                "Expected 'params' to be of type mogp_emulator.GPParams.GPParams, but "
                f"received {type(params)}."
            )

        if params.corr is None and params.cov is None:
            raise ValueError(
                "Cannot create hyperparameters with correlations and covariance equal to "
                "None in 'params'."
            )

        return cls(
            corr=params.corr,
            cov=params.cov,
            nugget=params.nugget,
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

    def to_mogp_gp_params(
        self, nugget_type: Literal["fixed", "fit", "adaptive", "pivot"] = "fixed"
    ) -> GPParams:
        """Convert this object to an instance of ``mogp_emulator.GPParams.GPParams``.

        The correlation length scales and covariance hyperparameters are copied to the
        returned mogp-emulator parameters object. How the nugget is set in the output
        depends on the value of the `nugget_type`, which describes the nugget fitting
        method recorded in the output object:

        * If `nugget_type` is one of {'fixed', 'fit'} then `self.nugget` must be defined
          as a real number. The nugget will be copied over to the output and the fitting
          method will be as specified in `nugget_type`. (In this case, 'fixed' represents
          the case where the nugget has been assigned a specific value, whereas 'fit'
          represents the case where the value has been estimated when fitting the
          emulator.)

        * If `nugget_type` is one of {'adaptive', 'pivot'} then `self.nugget` is ignored
          and will not be copied over to the output. The fitting method recorded in the
          output object will be as specified in `nugget_type`, representing the case
          where the nugget is computed in some way different to hyperparameter estimation.

        See https://mogp-emulator.readthedocs.io/en/latest/implementation/GPParams.html#mogp_emulator.GPParams.GPParams.nugget_type).
        for details of the `nugget_type` attribute in ``mogp_emulator.GPParams.GPParams``
        objects and
        https://mogp-emulator.readthedocs.io/en/latest/implementation/GaussianProcess.html#mogp_emulator.GaussianProcess.GaussianProcess
        for a discussion about what the different nugget fitting methods mean.

        Parameters
        ----------
        nugget_type : one of {"fixed", "fit", "adaptive", "pivot"}
            (Default: 'fixed') The type of nugget to be specified in construction of the
            returned ``mogp_emulator.GPParams.GPParams`` object. See above for discussion
            on valid values.

        Returns
        -------
        mogp_emulator.GPParams.GPParams
            An mogp-emulator parameters object with the hyperparameters given in this
            object.

        See Also
        --------
        See https://mogp-emulator.readthedocs.io/en/latest/implementation/GPParams.html#mogp_emulator.GPParams.GPParams.nugget_type).
        for details of the `nugget_type` attribute in ``mogp_emulator.GPParams.GPParams``
        objects and
        https://mogp-emulator.readthedocs.io/en/latest/implementation/GaussianProcess.html#mogp_emulator.GaussianProcess.GaussianProcess
        for a discussion about what the different nugget fitting methods mean.
        """

        if not isinstance(nugget_type, str):
            raise TypeError(
                "Expected 'nugget_type' to be of type str, but got "
                f"{type(nugget_type)}."
            )

        nugget_types = ["fixed", "fit", "adaptive", "pivot"]
        if nugget_type not in nugget_types:
            nugget_types_str = ", ".join(f"'{nt}'" for nt in nugget_types)
            raise ValueError(
                f"'nugget_type' must be one of {{{nugget_types_str}}}, "
                f"but got '{nugget_type}'."
            )

        if nugget_type in ["fixed", "fit"] and self.nugget is None:
            raise ValueError(
                f"Cannot set nugget fitting method to 'nugget_type = {nugget_type}' "
                "when this object's nugget is None."
            )

        transformed_params = [self.transform_corr(x) for x in self.corr] + [
            self.transform_cov(self.cov)
        ]

        if nugget_type == "fixed":
            params = GPParams(n_corr=len(self.corr), nugget=self.nugget)
        elif nugget_type == "fit":
            transformed_params.append(self.transform_nugget(self.nugget))
            params = GPParams(n_corr=len(self.corr), nugget="fit")
        else:
            params = GPParams(n_corr=len(self.corr), nugget=nugget_type)

        params.set_data(np.array(transformed_params, dtype=float))

        return params

    @staticmethod
    @_validate_nonnegative_real_domain("corr")
    def transform_corr(corr: Real) -> float:
        """Transform a correlation length scale parameter to a negative log scale.

        This maps the parameter to `-2 * log(corr)`; cf.
        https://mogp-emulator.readthedocs.io/en/latest/implementation/GPParams.html#mogp_emulator.GPParams.GPParams
        """
        if corr == 0:
            return math.inf

        return -2 * math.log(corr)

    @staticmethod
    @_validate_nonnegative_real_domain("cov")
    def transform_cov(cov: Real) -> float:
        """Transform a covariance parameter to the log scale.

        This maps the parameter to `log(cov)`; cf.
        https://mogp-emulator.readthedocs.io/en/latest/implementation/GPParams.html#mogp_emulator.GPParams.GPParams
        """
        if cov == 0:
            return -math.inf

        return math.log(cov)

    @staticmethod
    @_validate_nonnegative_real_domain("nugget")
    def transform_nugget(nugget: Real) -> float:
        """Transform a nugget parameter to the log scale.

        This maps the parameter to `log(nugget)`; cf.
        https://mogp-emulator.readthedocs.io/en/latest/implementation/GPParams.html#mogp_emulator.GPParams.GPParams
        """
        if nugget == 0:
            return -math.inf

        return math.log(nugget)
