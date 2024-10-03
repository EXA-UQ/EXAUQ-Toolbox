"""Provides emulators for simulators."""

from __future__ import annotations

import dataclasses
import itertools
from collections.abc import Collection, Sequence
from numbers import Real
from typing import Any, Literal, Optional

import mogp_emulator as mogp
import numpy as np
from mogp_emulator import GaussianProcess
from mogp_emulator.GPParams import GPParams
from numpy.typing import NDArray

from exauq.core.modelling import (
    AbstractGaussianProcess,
    GaussianProcessHyperparameters,
    GaussianProcessPrediction,
    Input,
    OptionalFloatPairs,
    TrainingDatum,
)
from exauq.utilities.mogp_fitting import fit_GP_MAP
from exauq.utilities.decorators import suppress_print, redirect_print_to_log


class MogpEmulator(AbstractGaussianProcess):
    """
    An emulator wrapping a ``GaussianProcess`` object from the mogp-emulator
    package.

    This class allows mogp-emulator ``GaussianProcess`` objects to be used with the
    designers defined in the EXAUQ-Toolbox, ensuring the interface required by the
    designers is present. Keyword arguments supplied to the `MogpEmulator` are passed onto
    the ``GaussianProcess`` initialiser to create the underlying (i.e. wrapped)
    ``GaussianProcess`` object. Note that any ``inputs`` or ``targets`` supplied are
    ignored: the underlying ``GaussianProcess`` will initially be constructed with no
    training data. Additionally, the only ``kernel``s currently supported are
    'SquaredExponential' (the default), 'Matern52' and 'ProductMat52'; these should be
    specified as strings during initialisation.

    The underlying ``GaussianProcess` object can be obtained through the
    `gp` property. Note that the `fit` method, used to train the emulator, will
    modify the underlying ``GaussianProcess``.

    Parameters
    ----------
    **kwargs : dict, optional
        Any permitted keyword arguments that can be used to create a mogp-emulator
        ``GaussianProcess`` object. See the mogp-emulator documentation for details. If
        ``inputs`` or ``targets`` are supplied as keyword arguments then these will be
        ignored. ``kernel``, if supplied, should be a string of one of the currently
        supported kernels (see above).

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
    ValueError
        If the kernel supplied is not one of the supported kernel functions.
    RuntimeError
        If keyword arguments are supplied upon initialisation that aren't
        supported by the initialiser of ``GaussianProcess`` from the
        mogp-emulator package.
    """

    _kernel_funcs = {
        "Matern52": mogp.Kernel.Matern52().kernel_f,
        "SquaredExponential": mogp.Kernel.SquaredExponential().kernel_f,
        "ProductMat52": mogp.Kernel.ProductMat52().kernel_f,
    }

    @suppress_print
    def __init__(self, **kwargs):
        self._gp_kwargs = self._remove_entries(kwargs, "inputs", "targets")
        self._validate_kernel(self._gp_kwargs)
        self._kernel = (
            self._kernel_funcs[self._gp_kwargs["kernel"]]
            if "kernel" in self._gp_kwargs
            else self._kernel_funcs["SquaredExponential"]
        )
        self._gp = self._make_gp(**self._gp_kwargs)

        # Add the default nugget type if not provided explicitly
        if "nugget" not in self._gp_kwargs:
            self._gp_kwargs["nugget"] = self._gp.nugget_type

        self._training_data = tuple(
            TrainingDatum.list_from_arrays(self._gp.inputs, self._gp.targets)
        )
        self._fit_hyperparameters = None

        # Correlation length scale parameters on a negative log scale
        self._corr_transformed = None

    @staticmethod
    def _remove_entries(_dict: dict, *args) -> dict:
        """Return a dict with the specified keys removed."""

        return {k: v for (k, v) in _dict.items() if k not in args}

    @classmethod
    def _validate_kernel(cls, kwargs):
        try:
            kernel = kwargs["kernel"]
        except KeyError:
            return None

        if kernel not in cls._kernel_funcs:
            raise ValueError(
                f"Could not initialise MogpEmulator with kernel = {kernel}: not a "
                "supported kernel function."
            )
        else:
            return None

    @staticmethod
    @suppress_print
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

    @suppress_print
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
            the process variance.

        Raises
        ------
        ValueError
            If `hyperparameters` is provided with nugget being ``None`` but `self.gp`
            was created with nugget fitting method 'fit'.
        """

        training_data = self._parse_training_data(training_data)
        if not training_data:
            return None

        if not (
            hyperparameters is None or isinstance(hyperparameters, MogpHyperparameters)
        ):
            raise TypeError(
                "Expected 'hyperparameters' to be None or of type "
                f"{MogpHyperparameters.__name__}, but received {type(hyperparameters)}."
            )

        self._validate_hyperparameter_bounds(hyperparameter_bounds)

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
        self._corr_transformed = self._gp.theta.corr_raw
        self._training_data = training_data

        return None

    @staticmethod
    def _parse_training_data(training_data: Any) -> tuple[TrainingDatum]:
        """Check that a collection of training data has been provided and return as a
        tuple if so."""

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

    def _validate_hyperparameter_bounds(
        self, hyperparameter_bounds: Sequence[OptionalFloatPairs]
    ) -> None:
        """Validate that each pair of bounds is ordered correctly in a sequence of
        hyperparameter bounds."""

        if hyperparameter_bounds is None:
            return None
        else:
            for i, bound in enumerate(hyperparameter_bounds):
                try:
                    self._validate_bound_pair(bound)
                except (TypeError, ValueError) as e:
                    raise e.__class__(
                        f"Invalid bound {bound} at index {i} of 'hyperparameter_bounds': {e}"
                    )
            return None

    @staticmethod
    def _validate_bound_pair(bounds: Sequence):
        """Check that a sequence of bounds defines a pair of bounds that are ordered
        correctly."""

        try:
            if not len(bounds) == 2:
                raise ValueError(
                    "Expected 'bounds' to be a sequence of length 2, but "
                    f"got length {len(bounds)}."
                )
        except TypeError:
            raise TypeError(
                f"Expected 'bounds' to be a sequence, but received {type(bounds)}."
            )

        if not all(bound is None or isinstance(bound, Real) for bound in bounds):
            raise TypeError(
                f"Expected each bound in {bounds} to be None or of type {Real}."
            )

        lower, upper = bounds
        if lower is not None and upper is not None and upper < lower:
            raise ValueError(
                "Lower bound must be less than or equal to upper bound, but received "
                f"lower bound = {lower} and upper bound = {upper}."
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
                hyperparameters.corr_length_scales,
                hyperparameters.process_var,
                kwargs["nugget"],
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
        parameters and process variance.

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
                MogpEmulator._transform_corr(bnd[1]),
                MogpEmulator._transform_corr(bnd[0]),
            )
            for bnd in bounds[:-1]
        ] + [
            (
                MogpEmulator._transform_cov(bounds[-1][0]),
                MogpEmulator._transform_cov(bounds[-1][1]),
            )
        ]
        return tuple(raw_bounds)

    @staticmethod
    def _transform_corr(corr: Optional[Real]) -> Optional[Real]:
        return MogpHyperparameters.transform_corr(corr) if corr is not None else None

    @staticmethod
    def _transform_cov(cov: Optional[Real]) -> Optional[Real]:
        return MogpHyperparameters.transform_cov(cov) if cov is not None else None

    def correlation(self, inputs1: Sequence[Input], inputs2: Sequence[Input]) -> NDArray:
        """Compute the correlation matrix for two sequences of simulator inputs.

        If ``corr_matrix`` is the Numpy array output by this method, then its shape is
        such that (in pseudocode) ``corr_matrix[i, j] = kernel(inputs1[i], inputs2[j])``,
        where ``kernel`` is the kernel function for the underlying Gaussian process. The
        only exception to this is when either of the sequence of inputs is empty, in which
        case an empty array is returned.

        In order to calculate the correlation between nonempty sequences of inputs, this
        emulator's ``fit_hyperparameters`` needs to not be ``None``, i.e. the emulator
        needs to have been trained on data.

        Parameters
        ----------
        inputs1, inputs2 : Sequence[Input]
            Sequences of simulator inputs.

        Returns
        -------
        numpy.ndarray
            The correlation matrix for the two sequences of inputs, as an array of shape
            ``(len(inputs1), len(inputs2))``.

        Raises
        ------
        AssertionError
            If this emulator has not yet been trained on data.
        ValueError
            If the dimension of any of the supplied simulator inputs doesn't match the
            dimension of training data inputs for this emulator.
        """
        try:
            if len(inputs1) == 0 or len(inputs2) == 0:
                return np.array([])
        except TypeError:
            # Raised if inputs1 or inputs2 not iterable
            raise TypeError(
                "Expected 'inputs1' and 'inputs2' to be sequences of Input objects, "
                f"but received {type(inputs1)} and {type(inputs2)} instead."
            )

        if not (
            all(isinstance(x, Input) for x in inputs1)
            and all(isinstance(x, Input) for x in inputs2)
        ):
            raise TypeError(
                "Expected 'inputs1' and 'inputs2' to only contain Input objects."
            )

        try:
            return self._kernel(inputs1, inputs2, self._corr_transformed)
        except AssertionError:
            # mogp-emulator arg validation errors typically seem to be raised as
            # AssertionErrors - these get bubbled up through self._kernel
            assert self._corr_transformed is not None, (
                f"Cannot calculate correlations for this instance of {self.__class__} "
                "because it hasn't yet been trained on data."
            )

        except ValueError:
            expected_dim = len(self.training_data[0].input)
            wrong_dims = list(
                len(x)
                for x in itertools.chain(inputs1, inputs2)
                if len(x) != expected_dim
            )
            if wrong_dims:
                raise ValueError(
                    f"Expected inputs to have dimension equal to {expected_dim}, but "
                    f"received input of dimension {wrong_dims[0]}."
                ) from None

    def covariance_matrix(self, inputs: Sequence[Input]) -> NDArray:
        """Compute the covariance matrix for a sequence of simulator inputs.

        In pseudocode, the covariance matrix for a given collection `inputs` of simulator
        inputs is defined in terms of the correlation matrix as ``sigma^2 *
        correlation(inputs, training_inputs)``, where ``sigma^2`` is the process variance
        for this Gaussian process (which was determined or supplied during training) and
        ``training_inputs`` are the simulator inputs used in training. The only exceptions
        to this are when the supplied `inputs` is empty or if this emulator hasn't been
        trained on data: in these cases an empty array is returned.

        Parameters
        ----------
        inputs : Sequence[Input]
            A sequence of simulator inputs.

        Returns
        -------
        numpy.ndarray
            The covariance matrix for the sequence of inputs, as an array of shape
            ``(len(inputs), n)`` where ``n`` is the number of training data points for
            this Gaussian process.

        Raises
        ------
        ValueError
            If the dimension of any of the supplied simulator inputs doesn't match the
            dimension of training data inputs for this emulator.
        """
        try:
            return super().covariance_matrix(inputs)
        except TypeError:
            if not isinstance(inputs, Sequence):
                raise TypeError(
                    "Expected 'inputs' to be a sequence of Input objects, but received "
                    f"{type(inputs)} instead."
                ) from None
            else:
                raise TypeError(
                    "Expected 'inputs' to only contain Input objects."
                ) from None

    def predict(self, x: Input) -> GaussianProcessPrediction:
        """Make a prediction of a simulator output for a given input.

        Parameters
        ----------
        x : Input
            A simulator input.

        Returns
        -------
        GaussianProcessPrediction
            The Gaussian process's prediction of the simulator output from the given
            input.

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
    def _to_prediction(mogp_predict_result) -> GaussianProcessPrediction:
        """Convert an MOGP ``PredictResult`` to a ``GaussianProcessPrediction`` object.

        See https://mogp-emulator.readthedocs.io/en/latest/implementation/GaussianProcess.html#the-predictresult-class
        """
        return GaussianProcessPrediction(
            estimate=mogp_predict_result.mean[0], variance=mogp_predict_result.unc[0]
        )


@dataclasses.dataclass(frozen=True)
class MogpHyperparameters(GaussianProcessHyperparameters):
    """Hyperparameters for use in fitting Gaussian processes via `MogpEmulator`.

    This provides a simplified interface to parameters used in
    ``mogp_emulator.GaussianProcess`` objects and is comparable to the
    ``mogp_emulator.GPParams.GPParams`` class. The correlation length scale parameters,
    process variance and nugget described below are on the 'transformed' (linear) scale
    rather than the log scale; cf.
    https://mogp-emulator.readthedocs.io/en/latest/implementation/GPParams.html#mogp_emulator.GPParams.GPParams

    Equality of `MogpHyperparameters` objects is tested hyperparameter-wise up to the
    default numerical precision defined in ``exauq.core.numerics.FLOAT_TOLERANCE``
    (see ``exauq.core.numerics.equal_within_tolerance``).

    Parameters
    ----------
    corr_length_scales : sequence or Numpy array of numbers.Real
        The correlation length scale parameters. The length of the sequence or array
        should equal the number of input coordinates for an emulator and each scale
        parameter should be a positive.
    process_var: numbers.Real
        The process variance, which should be positive.
    nugget : numbers.Real, optional
        (Default: None) A nugget, which should be non-negative if provided.

    Attributes
    ----------
    corr_length_scales : sequence or Numpy array of numbers.Real
        (Read-only) The correlation length scale parameters.
    process_var: numbers.Real
        (Read-only) The process variance.
    nugget : numbers.Real, optional
        (Read only, default: None) The nugget, or ``None`` if not supplied.
    """

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
                "Cannot create hyperparameters with correlation length scales and process "
                "variance equal to None in 'params'."
            )

        return cls(
            corr_length_scales=params.corr,
            process_var=params.cov,
            nugget=params.nugget,
        )

    def __eq__(self, other) -> bool:
        return isinstance(other, self.__class__) and super().__eq__(other)

    def to_mogp_gp_params(
        self, nugget_type: Literal["fixed", "fit", "adaptive", "pivot"] = "fixed"
    ) -> GPParams:
        """Convert this object to an instance of ``mogp_emulator.GPParams.GPParams``.

        The correlation length scales and process variance hyperparameters are copied to
        the returned mogp-emulator parameters object. How the nugget is set in the output
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

        transformed_params = [self.transform_corr(x) for x in self.corr_length_scales] + [
            self.transform_cov(self.process_var)
        ]

        if nugget_type == "fixed":
            params = GPParams(n_corr=len(self.corr_length_scales), nugget=self.nugget)
        elif nugget_type == "fit":
            transformed_params.append(self.transform_nugget(self.nugget))
            params = GPParams(n_corr=len(self.corr_length_scales), nugget="fit")
        else:
            params = GPParams(n_corr=len(self.corr_length_scales), nugget=nugget_type)

        params.set_data(np.array(transformed_params, dtype=float))

        return params
