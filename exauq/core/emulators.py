"""
Provides the emulators for the simulators building upon the `mogp` package
and adapting to work with the implemented designers from ``exauq.core.designers``.


[MogpEmulator][exauq.core.emulators.MogpEmulator]
---------------------------------------------------------------------------------------
[`correlation`][exauq.core.emulators.MogpEmulator.correlation]
Compute correlation matrix for Input Sequences.

[`covariance_matrix`][exauq.core.emulators.MogpEmulator.covariance_matrix]
Compute covariance matrix for Input Sequences.

[`fit`][exauq.core.emulators.MogpEmulator.fit]
Fit emulator to the data.

[`fit_hyperparameters`][exauq.core.emulators.MogpEmulator.fit_hyperparameters]
**(Read-Only)** Hyperparameters of current fitted GP.

[`gp`][exauq.core.emulators.MogpEmulator.gp]
**(Read-Only)** Underlying GP for this emulator.

[`predict`][exauq.core.emulators.MogpEmulator.predict]
Make prediction for simulator output given Input.

[`training_data`][exauq.core.emulators.MogpEmulator.training_data]
**(Read-only)** The data on which the emulator has been trained.


[MogpHyperparameters][exauq.core.emulators.MogpHyperparameters]
---------------------------------------------------------------------------------------
[`from_mogp_gp_params`][exauq.core.emulators.MogpHyperparameters.from_mogp_gp_params]
Create instance of `MogpHyperparameters`.

[`to_mogp_gp_params`][exauq.core.emulators.MogpHyperparameters.to_mogp_gp_params]
Convert to an instance of ``mogp_emulator.GPParams.GPParams``.


"""

from __future__ import annotations

import dataclasses
import itertools
from collections.abc import Collection, Sequence
from numbers import Real
from typing import Any, Literal, Optional, Tuple
from warnings import warn

import mogp_emulator as mogp
import numpy as np
import pymc as pm
from mogp_emulator import GaussianProcess
from mogp_emulator.GPParams import GPParams
from numpy.typing import NDArray
from pytensor.tensor import dot, eye
from pytensor.tensor import sum as pt_sum
from pytensor.tensor.slinalg import cholesky, solve_triangular

from exauq.core.modelling import (
    AbstractGaussianProcess,
    AbstractHyperparameters,
    GaussianProcessHyperparameters,
    GaussianProcessPrediction,
    Input,
    MLTrainingData,
    MultiLevel,
    OptionalFloatPairs,
    TrainingData,
    TrainingDatum,
)
from exauq.core.numerics import equal_within_tolerance
from exauq.utilities.decorators import suppress_print
from exauq.utilities.mogp_fitting import fit_GP_MAP


class MogpEmulator(AbstractGaussianProcess[TrainingData]):
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

    The underlying ``GaussianProcess`` object can be obtained through the
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
    kinv : numpy.ndarray
        (Read-only) The inverse of the covariance matrix of the training data,
        or an empty NumPy array if the model has not been fitted to data.

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

        # Inverse of covariance matrix
        self._kinv = np.array([])

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

    @property
    def kinv(self) -> NDArray:
        """(Read-only) The inverse of the covariance matrix of the training data,
        or an empty NumPy array if the model has not been fitted to data."""

        return self._kinv

    @suppress_print
    def fit(
        self,
        training_data: TrainingData,
        hyperparameters: Optional[MogpHyperparameters] = None,
        hyperparameter_bounds: Optional[Sequence[OptionalFloatPairs]] = None,
    ) -> None:
        """Fit the emulator to data.

        This method trains the underlying ``GaussianProcess``, as stored in
        the `gp` property, using the supplied training data. By default,
        hyperparameters are estimated as part of this training, by maximising the
        log-posterior. Alternatively, a collection of hyperparameters can be supplied to
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
        training_data :
            The pairs of inputs and simulator outputs on which the emulator
            should be trained. Should be a finite collection of such pairs.
        hyperparameters :
            Hyperparameters to use directly in fitting the Gaussian
            process. If ``None`` then the hyperparameters will be estimated as part of
            fitting to data.
        hyperparameter_bounds :
            A sequence of bounds to apply to hyperparameters
            during estimation, of the form ``(lower_bound, upper_bound)``. All
            but the last tuple should represent bounds for the correlation
            length parameters, while the last tuple should represent bounds for
            the process variance.

        Raises
        ------
        ValueError
            If `training_data` is provided with duplicate inputs: all inputs must be unique.

            If `hyperparameters` is provided with nugget being ``None`` but `self.gp`
            was created with nugget fitting method 'fit'.
        """

        training_data = self._parse_training_data(training_data)
        if not training_data:
            return None

        self._validate_training_data_unique(training_data)

        if not (
            hyperparameters is None or isinstance(hyperparameters, MogpHyperparameters)
        ):
            raise TypeError(
                "Expected 'hyperparameters' to be None or of type "
                f"{MogpHyperparameters.__name__}, but received {type(hyperparameters)} instead."
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

        if len(training_data) < self._gp.n_params:
            warn(
                f"Fewer training points ({len(training_data)}) than hyperparameters ({self._gp.n_params}) being "
                f"estimated. Estimates may be unreliable."
            )

        self._corr_transformed = self._gp.theta.corr_raw
        self._training_data = training_data
        self._kinv = self._compute_kinv()

        return None

    @staticmethod
    def _validate_training_data_unique(training_data: tuple[TrainingDatum]):
        """Check whether the given collection of TrainingDatum are unique,
        raising a ValueError if not."""

        inputs = [data_point.input for data_point in training_data]

        for input1, input2 in itertools.combinations(inputs, 2):
            if equal_within_tolerance(input1, input2):
                raise ValueError(
                    f"Points {np.round(input1, 9)} and {np.round(input2, 9)}"
                    " in 'TrainingDatum' are not unique within tolerance."
                )

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
                f"Expected 'training_data' to be of type finite collection of TrainingDatum, "
                f"but received {type(training_data)} instead."
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
                f"Expected 'bounds' to be of type sequence, but received {type(bounds)} instead."
            )

        if not all(bound is None or isinstance(bound, Real) for bound in bounds):
            raise TypeError(
                f"Expected each 'bound' in {bounds} to be None or of type {Real} "
                "but one or more elements were of an unexpected type."
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

        <https://mogp-emulator.readthedocs.io/en/latest/implementation/GPParams.html#mogp_emulator.GPParams.GPParams>
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
        inputs1, inputs2 :
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
                "Expected 'inputs1' and 'inputs2' to be of type sequences of Input objects, "
                f"but received {type(inputs1)} and {type(inputs2)} instead."
            )

        if not (
            all(isinstance(x, Input) for x in inputs1)
            and all(isinstance(x, Input) for x in inputs2)
        ):
            raise TypeError(
                "Expected all 'inputs1' and 'inputs2' to be of type Input objects, "
                "but one or more elements were of an unexpected type."
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
        inputs :
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
                    "Expected 'inputs' to be of type sequence of Input objects, but received "
                    f"{type(inputs)} instead."
                ) from None
            else:
                raise TypeError(
                    "Expected all elements of 'inputs' to be of type Input objects, "
                    "but one or more elements were of an unexpected type."
                ) from None

    def predict(self, x: Input) -> GaussianProcessPrediction:
        """Make a prediction of a simulator output for a given input.

        Parameters
        ----------
        x :
            A simulator input.

        Returns
        -------
        GaussianProcessPrediction
            The Gaussian process' prediction of the simulator output from the given
            input.

        Raises
        ------
        RuntimeError
            If this emulator has not been trained on any data before making the
            prediction.
        """

        if not isinstance(x, Input):
            raise TypeError(
                f"Expected 'x' to be of type Input, but received {type(x)} instead."
            )

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
    [mogp_docs.GPParams.GPParams](https://mogp-emulator.readthedocs.io/en/latest/implementation/GPParams.html#mogp_emulator.GPParams.GPParams).

    Equality of `MogpHyperparameters` objects is tested hyperparameter-wise up to the
    default numerical precision defined in ``exauq.core.numerics.FLOAT_TOLERANCE``
    (see ``exauq.core.numerics.equal_within_tolerance``).

    Parameters
    ----------
    corr_length_scales : sequence or Numpy array of Real
        The correlation length scale parameters. The length of the sequence or array
        should equal the number of input coordinates for an emulator and each scale
        parameter should be a positive.
    process_var: numbers.Real
        The process variance, which should be positive.
    nugget : numbers.Real, optional
         A nugget, which should be non-negative if provided.

    Attributes
    ----------
    corr_length_scales : sequence or Numpy array of Real
        (Read-only) The correlation length scale parameters.
    process_var: numbers.Real
        (Read-only) The process variance.
    nugget : numbers.Real, optional
        (Read only) The nugget, or ``None`` if not supplied.

    See Also
    ---------
    [equal_within_tolerance][exauq.core.numerics.equal_within_tolerance] :
    Numerical tolerance check.
    """

    @classmethod
    def from_mogp_gp_params(cls, params: GPParams) -> MogpHyperparameters:
        """Create an instance of MogpHyperparameters from an
        ``mogp_emulator.GPParams.GPParams`` object.

        Parameters
        ----------
        params :
            A parameters object from mogp-emulator.

        Returns
        -------
        MogpHyperparameters
            The hyperparameters extracted from the given `params`.
        """

        if not isinstance(params, GPParams):
            raise TypeError(
                "Expected 'params' to be of type mogp_emulator.GPParams.GPParams, but "
                f"received {type(params)} instead."
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


        Parameters
        ----------
        nugget_type : one of {"fixed", "fit", "adaptive", "pivot"}
            The type of nugget to be specified in construction of the
            returned ``mogp_emulator.GPParams.GPParams`` object. See above for discussion
            on valid values.

        Returns
        -------
        mogp_emulator.GPParams.GPParams
            An mogp-emulator parameters object with the hyperparameters given in this
            object.

        See Also
        --------
        See [mogp-emulator/nugget_type](https://mogp-emulator.readthedocs.io/en/latest/implementation/GPParams.html#mogp_emulator.GPParams.GPParams.nugget_type)
        for details of the `nugget_type` attribute in ``mogp_emulator.GPParams.GPParams``
        objects and
        [mogp-emulator/nugget_fitting_methods](https://mogp-emulator.readthedocs.io/en/latest/implementation/GaussianProcess.html#mogp_emulator.GaussianProcess.GaussianProcess)
        for a discussion about what the different nugget fitting methods mean.

        """

        if not isinstance(nugget_type, str):
            raise TypeError(
                "Expected 'nugget_type' to be of type str, but received "
                f"{type(nugget_type)} instead."
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


class DeepDishGPHyperparameters(AbstractHyperparameters):
    """
    Class to manage hyperparameters for multi-level Gaussian Process models.

    This class handles creation, inheritance, and management of hyperparameters
    across multiple levels according to the specified rules:

    1. Defaults apply when no hyperparameter is provided.
    2. Single-level specifications (without level suffix) define Level 1 parameters.
    3. Higher levels inherit from lower levels unless explicitly overridden.
    4. Parameters follow the naming convention {parameter}_L{level} (e.g., ls1_L1)
    """

    def __init__(self, model_context=None, input_dims=2, levels=3):
        """
        Initialize the hyperparameters manager.

        Parameters:
        -----------
        model_context : pm.Model, optional
            PyMC model context where hyperparameters will be defined.
            Can be set later with set_model_context().
        input_dims : int
            Number of input dimensions (determines number of length scales)
        levels : int
            Number of levels in the multi-level GP model
        """
        self.model = model_context
        self.input_dims = input_dims
        self.levels = levels
        self.param_specs = {}  # Specifications for parameters
        self.params = {}  # Created parameter objects
        self.initialized = False

    def set_model_context(self, model_context):
        """
        Set the PyMC model context after initialization.

        Parameters:
        -----------
        model_context : pm.Model
            PyMC model context where hyperparameters will be defined
        """
        self.model = model_context
        # Reset initialization state and params since we have a new context
        self.initialized = False
        self.params = {}
        return self

    def set_prior(self, param_name, dist_type, **dist_kwargs):
        """
        Set a prior distribution for a parameter.

        Parameters:
        -----------
        param_name : str
            Name of the parameter, with or without level suffix.
            Examples: "ls1", "ls1_L2", "sig_L3"
        dist_type : str
            Name of the PyMC distribution (e.g., "Gamma", "Normal")
        **dist_kwargs :
            Keyword arguments for the distribution constructor
        """
        # Handle case where level is not specified (applies to Level 1)
        if "_L" not in param_name:
            param_name = f"{param_name}_L1"

        self.param_specs[param_name] = {"dist_type": dist_type, "params": dist_kwargs}

        # Clear params to ensure reinitialization
        self.params = {}
        self.initialized = False
        return self

    def apply_defaults(self):
        """Apply default prior distributions for parameters not explicitly set"""
        # Define defaults for Level 1
        for i in range(1, self.input_dims + 1):
            if f"ls{i}_L1" not in self.param_specs:
                self.param_specs[f"ls{i}_L1"] = {
                    "dist_type": "Gamma",
                    "params": {"alpha": 2, "beta": 4},
                }

        if "sig_L1" not in self.param_specs:
            self.param_specs["sig_L1"] = {
                "dist_type": "Gamma",
                "params": {"alpha": 8, "beta": 2},
            }

        if "nug_L1" not in self.param_specs:
            self.param_specs["nug_L1"] = {
                "dist_type": "Gamma",
                "params": {"alpha": 2, "beta": 4},
            }

        if "beta_L1" not in self.param_specs:
            self.param_specs["beta_L1"] = {
                "dist_type": "Normal",
                "params": {"mu": 0, "sigma": 10},
            }

        # Set up inheritance for higher levels
        base_params = ["sig", "nug", "beta"]
        base_params.extend([f"ls{i}" for i in range(1, self.input_dims + 1)])

        for level in range(2, self.levels + 1):
            for base_param in base_params:
                param_name = f"{base_param}_L{level}"
                if param_name not in self.param_specs:
                    # Inherit from the previous level
                    prev_level_param = f"{base_param}_L{level - 1}"
                    self.param_specs[param_name] = {"inherit_from": prev_level_param}

    def initialize(self):
        """
        Initialize all hyperparameters within the model context.

        Raises:
        -------
        ValueError
            If model_context has not been set
        """
        if self.initialized:
            return

        if self.model is None:
            raise ValueError(
                "Model context must be set before initialization. Use set_model_context()."
            )

        # Apply defaults for parameters not explicitly set
        self.apply_defaults()

        # Create actual parameter objects
        created_params = {}
        with self.model:
            # First pass: create all parameters that don't inherit
            for param_name, spec in self.param_specs.items():
                if "inherit_from" not in spec:
                    dist_class = getattr(pm, spec["dist_type"])
                    created_params[param_name] = dist_class(param_name, **spec["params"])

        # Second pass: resolve inheritances
        self.params = created_params.copy()
        for param_name, spec in self.param_specs.items():
            if "inherit_from" in spec:
                inherit_from = spec["inherit_from"]
                if inherit_from in self.params:
                    self.params[param_name] = self.params[inherit_from]

        self.initialized = True

    def get(self, param_name, level=None):
        """
        Get a hyperparameter.

        Parameters:
        -----------
        param_name : str
            Base name of the parameter (e.g., "ls1", "sig")
        level : int, optional
            Level to get the parameter for. If None, assumes param_name
            already includes the level suffix.

        Returns:
        --------
        pm.Distribution
            The requested hyperparameter

        Raises:
        -------
        ValueError
            If model_context has not been set or parameter initialization failed
        """
        if self.model is None:
            raise ValueError(
                "Model context must be set before getting parameters. Use set_model_context()."
            )

        if not self.initialized:
            self.initialize()

        if level is not None:
            param_name = f"{param_name}_L{level}"

        param = self.params.get(param_name)
        if param is None:
            raise ValueError(
                f"Parameter {param_name} not found. Check parameter name and level."
            )

        return param

    def get_lengthscales(self, level):
        """
        Get all length scale parameters for a specific level as a list.

        Parameters:
        -----------
        level : int
            The level to get length scales for

        Returns:
        --------
        list
            List of length scale parameters for the specified level
        """
        return [self.get(f"ls{i}", level=level) for i in range(1, self.input_dims + 1)]

    def get_all_for_level(self, level):
        """
        Get all hyperparameters for a specific level.

        Parameters:
        -----------
        level : int
            The level to get parameters for

        Returns:
        --------
        dict
            Dictionary of parameter names to parameter objects
        """
        if self.model is None:
            raise ValueError(
                "Model context must be set before getting parameters. Use set_model_context()."
            )

        if not self.initialized:
            self.initialize()

        level_params = {}
        for param_name in self.params:
            if f"_L{level}" in param_name:
                base_name = param_name.split(f"_L{level}")[0]
                level_params[base_name] = self.params[param_name]

        return level_params

    def get_signal_variance(self, level):
        """Get signal variance parameter for the specified level"""
        return self.get("sig", level)

    def get_nugget(self, level):
        """Get nugget parameter for the specified level"""
        return self.get("nug", level)

    def get_mean_constant(self, level):
        """Get mean constant parameter for the specified level"""
        return self.get("beta", level)


class PosteriorCovariance(pm.gp.cov.Covariance):
    def __init__(self, prior_cov, X_train, NL, noise_sigma):
        """
        Posterior covariance function for multi-level Gaussian Process.

        Parameters
        ----------
        prior_cov: Instance of a PyMC covariance function (e.g., SquaredExponential)
        X_train: Training inputs (N x D)
        NL: Current level in the multi-level structure
        noise_sigma: Observation noise standard deviation
        """
        input_dim = X_train[0].shape[1]
        super(PosteriorCovariance, self).__init__(input_dim)
        self.prior_cov = prior_cov
        self.X_train = X_train
        self.noise_sigma = noise_sigma
        self.NL = NL

    def full(self, X, Xs=None):
        """Compute the posterior covariance matrix"""
        # Default Xs to X if not provided
        if Xs is None:
            Xs = X

        # Compute prior covariances
        K_xx = self.prior_cov(self.X_train[self.NL - 1], self.X_train[self.NL - 1])
        K_xs = self.prior_cov(self.X_train[self.NL - 1], X)
        K_ss = self.prior_cov(X)

        # Add noise to training covariance
        input_dim = K_xx.shape[0]

        noise_matrix = eye(input_dim) * (self.noise_sigma**2)
        L_xx = cholesky(K_xx + noise_matrix)
        L_inv = solve_triangular(L_xx, eye(L_xx.shape[0]), lower=True)
        K_xx_inv = dot(L_inv.T, L_inv)

        # Compute posterior covariance
        K_post = K_ss - dot(dot(K_xs.T, K_xx_inv), K_xs)
        return K_post

    def diag(self, X):
        """Diagonal elements of posterior covariance (predictive variance)"""
        K_ss_diag = self.prior_cov.diag(X)
        K_xs = self.prior_cov(self.X_train[self.NL - 1], X)
        K_xx = self.prior_cov(self.X_train[self.NL - 1])
        input_dim = K_xx.shape[0]

        noise_matrix = eye(input_dim) * (self.noise_sigma**2)

        L_xx = cholesky(K_xx + noise_matrix)
        L_inv = solve_triangular(L_xx, eye(input_dim), lower=True)
        K_xx_inv = dot(L_inv.T, L_inv)

        # Compute diagonal elements of posterior covariance
        diag_K_post = K_ss_diag - pt_sum(dot(K_xs.T, K_xx_inv) * K_xs.T, axis=0)

        return diag_K_post


class PosteriorMean(pm.gp.mean.Mean):
    def __init__(self, prior_mean, prior_cov, X_train, Y_train, NL, noise_sigma):
        """
        Posterior mean function for multi-level Gaussian Process.

        Parameters
        ----------
        prior_mean: Instance of a PyMC mean function
        prior_cov: Instance of a PyMC covariance function
        X_train: Training inputs (N x D)
        Y_train: Training outputs
        NL: Current level in the multi-level structure
        noise_sigma: Observation noise standard deviation
        """
        super(PosteriorMean, self).__init__()
        self.prior_mean = prior_mean
        self.prior_cov = prior_cov
        self.X_train = X_train
        self.Y_train = Y_train
        self.noise_sigma = noise_sigma
        self.NL = NL

    def __call__(self, X):
        """Compute the posterior mean"""
        K_xx = self.prior_cov(self.X_train[self.NL - 1], self.X_train[self.NL - 1])
        K_xs = self.prior_cov(self.X_train[self.NL - 1], X)

        # Add noise to training covariance
        input_dim = K_xx.shape[0]

        noise_matrix = eye(input_dim) * (self.noise_sigma**2)
        L_xx = cholesky(K_xx + noise_matrix)
        L_inv = solve_triangular(L_xx, eye(L_xx.shape[0]), lower=True)
        K_xx_inv = dot(L_inv.T, L_inv)

        M_post = self.prior_mean(X) + dot(
            dot(K_xs.T, K_xx_inv),
            (self.Y_train[self.NL - 1] - self.prior_mean(self.X_train[self.NL - 1])),
        )
        return M_post


class DeepDishGP(AbstractGaussianProcess[MLTrainingData]):
    """
    Multi-level Deep GP emulator using PyMC.

    This class implements a multi-level Gaussian Process where each level
    builds upon the posterior distribution of the previous level.
    The number of input dimensions and levels are determined automatically
    from the training data.
    """

    def __init__(self):
        """
        Initialize a DeepDishGP.

        The number of input dimensions and levels will be determined
        automatically from the training data when fit() is called.
        """
        self._input_dims = None
        self._levels = None
        self._training_data = None
        self._fit_hyperparameters = None
        self._model = None
        self._trace = None
        self._kinv_value = None
        self._cov_funcs = None
        self._mean_funcs = None
        self._all_cov_funcs = None
        self._all_mean_funcs = None

    @property
    def training_data(self) -> tuple[TrainingDatum]:
        """The data on which the emulator has been trained."""
        return self._training_data

    @property
    def fit_hyperparameters(self) -> Optional[GaussianProcessHyperparameters]:
        """The hyperparameters of the fit for this emulator."""
        return self._fit_hyperparameters

    @property
    def kinv(self):
        """The inverse of the covariance matrix of the training data."""
        if self._kinv_value is None and self._training_data is not None:
            self._kinv_value = self._compute_kinv()
        return self._kinv_value

    def _organize_training_data(
        self, training_data: MLTrainingData
    ) -> Tuple[list, list, list]:
        """
        Organize training data into arrays for PyMC model and determine
        dimensionality and levels.

        Parameters
        ----------
        training_data : MLTrainingData
            Multi-level training data

        Returns
        -------
        tuple
            Organized X, y, and combined data for each level
        """
        if not isinstance(training_data, MultiLevel):
            raise TypeError(
                f"Expected 'training_data' to be of type {MultiLevel.__name__}, "
                f"but received {type(training_data)} instead."
            )

        # Determine the number of levels from the data
        available_levels = sorted(training_data.levels)

        # Validate levels form a continuous sequence starting from 1
        expected_levels = list(range(1, max(available_levels) + 1))
        if available_levels != expected_levels:
            missing = set(expected_levels) - set(available_levels)
            raise ValueError(
                f"Missing training data for levels: {missing}. "
                f"Required continuous sequence from 1 to {max(available_levels)}"
            )

        # Set levels from training data
        self._levels = max(available_levels)

        # Convert training data to arrays for each level
        X_arrays = []
        y_arrays = []

        # Determine input dimensionality from first data point
        first_level_data = training_data[1]
        if not first_level_data:
            raise ValueError("Training data must contain at least one point at level 1")

        self._input_dims = len(first_level_data[0].input)

        for level in range(1, self._levels + 1):
            level_data = training_data[level]

            # Check for duplicate inputs
            inputs = [datum.input for datum in level_data]
            if len(inputs) != len(set(map(str, inputs))):
                raise ValueError(
                    f"Duplicate inputs found in training data for level {level}"
                )

            # Validate input dimensions consistent across all points
            for datum in level_data:
                if len(datum.input) != self._input_dims:
                    raise ValueError(
                        f"Inconsistent input dimensions. Expected {self._input_dims} but "
                        f"found {len(datum.input)} at level {level}"
                    )

            # Extract inputs and outputs
            X_level = np.array([[coord for coord in datum.input] for datum in level_data])
            y_level = np.array([datum.output for datum in level_data])

            X_arrays.append(X_level)
            y_arrays.append(y_level)

        # Store the training data
        self._training_data = MultiLevel(
            {level: tuple(data) for level, data in training_data.items()}
        )

        return X_arrays, y_arrays, list(zip(X_arrays, y_arrays))

    def fit(
        self,
        training_data: MLTrainingData,
        hyperparameters: Optional[DeepDishGPHyperparameters] = None,
        hyperparameter_bounds: Optional[Sequence[OptionalFloatPairs]] = None,
    ) -> None:
        """
        Fit the DeepDishGP to data.

        Parameters
        ----------
        training_data : MLTrainingData
            Multi-level training data
        hyperparameters : Optional[GaussianProcessHyperparameters]
            Hyperparameters to use (if None, they will be estimated)
        hyperparameter_bounds : Optional[Sequence[OptionalFloatPairs]]
            Bounds for hyperparameter estimation
        """
        X_arrays, y_arrays, combined_data = self._organize_training_data(training_data)

        # Create hyperparameters
        if hyperparameters is None:
            # Initialize hyperparameters with default priors
            hparams = self._create_default_hyperparameters(hyperparameter_bounds)
        else:
            hparams = hyperparameters

        # Create and sample from the model
        with pm.Model() as model:
            # Set model context for hyperparameters
            hparams.set_model_context(model)
            hparams.initialize()  # Ensure all hyperparameters are initialized

            # Store all covariance and mean functions for later prediction
            self._cov_funcs = []
            self._mean_funcs = []

            # Level 1 GP (Base level)
            length_scales_L1 = hparams.get_lengthscales(1)
            sig_L1 = hparams.get_signal_variance(1)
            nug_L1 = hparams.get_nugget(1)
            beta_L1 = hparams.get_mean_constant(1)

            cov1 = sig_L1**2 * pm.gp.cov.ExpQuad(self._input_dims, ls=length_scales_L1)
            mean1 = pm.gp.mean.Constant(beta_L1)
            gp1 = pm.gp.Marginal(mean_func=mean1, cov_func=cov1)
            y_obs1 = gp1.marginal_likelihood(
                "y_obs1", X=X_arrays[0], y=y_arrays[0], noise=nug_L1
            )

            self._cov_funcs.append(cov1)
            self._mean_funcs.append(mean1)

            # Build up each level in sequence with correct posteriors
            prev_covs = [cov1]  # Track all previous covariance functions
            prev_means = [mean1]  # Track all previous mean functions

            for level in range(2, self._levels + 1):
                # Get level-specific hyperparameters
                length_scales = hparams.get_lengthscales(level)
                sig = hparams.get_signal_variance(level)
                nug = hparams.get_nugget(level)
                beta = hparams.get_mean_constant(level)

                # Create level-specific prior cov and mean
                cov_prior = sig**2 * pm.gp.cov.ExpQuad(self._input_dims, ls=length_scales)
                mean_prior = pm.gp.mean.Constant(beta)

                # Store for later use
                self._cov_funcs.append(cov_prior)
                self._mean_funcs.append(mean_prior)

                # Build up posterior functions from previous levels
                level_cov = cov_prior
                level_mean = mean_prior

                # For each previous level, create posterior for current level
                for prev_level in range(1, level):
                    idx = prev_level - 1  # 0-indexed arrays

                    # Create posterior from previous level
                    level_cov = PosteriorCovariance(
                        level_cov, X_arrays, prev_level + 1, 1e-4
                    )
                    level_mean = PosteriorMean(
                        level_mean,
                        prev_covs[idx],
                        X_arrays,
                        y_arrays,
                        prev_level + 1,
                        1e-4,
                    )

                # Add to tracking lists
                prev_covs.append(level_cov)
                prev_means.append(level_mean)

                # Create GP for this level
                gp_level = pm.gp.Marginal(mean_func=level_mean, cov_func=level_cov)
                y_obs_level = gp_level.marginal_likelihood(
                    f"y_obs{level}",
                    X=X_arrays[level - 1],
                    y=y_arrays[level - 1],
                    noise=nug,
                )

            # Sample
            trace = pm.sample(
                1000,
                tune=1000,
                return_inferencedata=True,
                target_accept=0.95,
                progressbar=True,
            )

        # Store model components for prediction
        self._model = model
        self._trace = trace
        self._all_cov_funcs = prev_covs
        self._all_mean_funcs = prev_means

        # Store hyperparameters
        self._fit_hyperparameters = self._extract_hyperparameters_from_trace(
            trace, hparams
        )

        # Reset kinv since model has changed
        self._kinv_value = None

    def _create_default_hyperparameters(self, bounds=None):
        """
        Create default hyperparameters for the model.

        Parameters
        ----------
        bounds : Optional[Sequence[OptionalFloatPairs]]
            Bounds for hyperparameter estimation

        Returns
        -------
        DeepDishGPHyperparameters
            Configured hyperparameters object
        """

        if self._input_dims is None or self._levels is None:
            raise ValueError(
                "Cannot create hyperparameters before fitting. Input dimensions and levels unknown."
            )

        hparams = DeepDishGPHyperparameters(
            input_dims=self._input_dims, levels=self._levels
        )

        # Set default priors for all length scales
        for i in range(1, self._input_dims + 1):
            hparams.set_prior(f"ls{i}", "Gamma", alpha=2, beta=4)

        # Set other default hyperparameters
        hparams.set_prior("sig", "Gamma", alpha=8, beta=2).set_prior(
            "nug", "Gamma", alpha=2, beta=4
        ).set_prior("beta", "Normal", mu=0, sigma=10)

        # Apply bounds if provided
        if bounds is not None:
            if len(bounds) != self._input_dims + 1:
                raise ValueError(
                    f"Expected {self._input_dims + 1} bounds (length scales + process variance), "
                    f"but received {len(bounds)}"
                )

            # TODO: Implement bounds application to priors

        return hparams

    def _extract_hyperparameters_from_trace(self, trace, hparams):
        """
        Extract hyperparameters from the sampling trace.

        Parameters
        ----------
        trace : PyMC inference data
            Trace from sampling
        hparams : DeepDishGPHyperparameters
            Hyperparameters object used for sampling

        Returns
        -------
        GaussianProcessHyperparameters
            Fitted hyperparameters
        """
        # Extract posterior means for key parameters
        ls_values = []

        for i in range(1, self._input_dims + 1):
            param_name = f"ls{i}_L1"
            if param_name in trace.posterior:
                ls_values.append(float(trace.posterior[param_name].mean().values))
            else:
                raise ValueError(f"Required parameter {param_name} not found in trace")

        sig_value = float(trace.posterior["sig_L1"].mean().values)
        nug_value = float(trace.posterior["nug_L1"].mean().values)

        # Create GaussianProcessHyperparameters
        return GaussianProcessHyperparameters(
            corr_length_scales=ls_values, process_var=sig_value, nugget=nug_value
        )

    def predict(self, x: Input) -> GaussianProcessPrediction:
        """
        Make a prediction for the given input.

        Parameters
        ----------
        x : Input
            Input point to make prediction at

        Returns
        -------
        GaussianProcessPrediction
            The prediction with mean and variance
        """
        if self._trace is None or self._input_dims is None or self._levels is None:
            raise ValueError("Model has not been fitted yet. Call fit() first.")

        if not isinstance(x, Input):
            raise TypeError(f"Expected 'x' to be of type Input, but received {type(x)}")

        if len(x) != self._input_dims:
            raise ValueError(
                f"Expected input of dimension {self._input_dims}, but received {len(x)}"
            )

        # Convert input to numpy array
        x_array = np.array([coord for coord in x]).reshape(1, -1)

        # Extract data from training
        X_arrays, y_arrays = self._get_training_arrays()

        # Extract mean hyperparameters from the trace for level 1 (as defaults)
        trace = self._trace
        ls_values_L1 = [
            float(trace.posterior[f"ls{i}_L1"].mean().values)
            for i in range(1, self._input_dims + 1)
        ]
        sig_value_L1 = float(trace.posterior["sig_L1"].mean().values)
        beta_value_L1 = float(trace.posterior["beta_L1"].mean().values)

        # Initialize predictions for each level
        level_means = []
        level_vars = []

        # Make predictions for each level
        for level in range(1, self._levels + 1):
            # Get level-specific hyperparameters, falling back to level 1 if not available
            try:
                ls_values = [
                    float(trace.posterior[f"ls{i}_L{level}"].mean().values)
                    for i in range(1, self._input_dims + 1)
                ]
            except (KeyError, AttributeError):
                ls_values = ls_values_L1

            try:
                sig_value = float(trace.posterior[f"sig_L{level}"].mean().values)
            except (KeyError, AttributeError):
                sig_value = sig_value_L1

            try:
                beta_value = float(trace.posterior[f"beta_L{level}"].mean().values)
            except (KeyError, AttributeError):
                beta_value = beta_value_L1

            # Create the level-specific covariance and mean functions
            cov_level = sig_value**2 * pm.gp.cov.ExpQuad(self._input_dims, ls=ls_values)
            mean_level = pm.gp.mean.Constant(beta_value)

            # For levels higher than 1, apply posterior transformations
            if level > 1:
                # Apply transformations for each previous level
                for prev_level in range(1, level):
                    X_prev = X_arrays[prev_level - 1]
                    y_prev = y_arrays[prev_level - 1]

                    # Extract noise parameter for previous level
                    try:
                        nug_prev = float(
                            trace.posterior[f"nug_L{prev_level}"].mean().values
                        )
                    except (KeyError, AttributeError):
                        nug_prev = float(trace.posterior["nug_L1"].mean().values)

                    # Creating posterior functions in numpy for stability
                    # Compute Gram matrix
                    K_xx = self._compute_gram_matrix(X_prev, X_prev, ls_values, sig_value)
                    K_xs = self._compute_gram_matrix(
                        X_prev, x_array, ls_values, sig_value
                    )
                    k_ss = sig_value  # Self-covariance

                    # Add noise
                    K_xx += np.eye(len(X_prev)) * nug_prev

                    # Compute Cholesky decomposition
                    try:
                        L = np.linalg.cholesky(K_xx)
                    except np.linalg.LinAlgError:
                        # Add regularization
                        K_xx += np.eye(len(X_prev)) * 1e-4
                        L = np.linalg.cholesky(K_xx)

                    # Compute posterior mean
                    alpha = np.linalg.solve(L, y_prev - beta_value)
                    alpha = np.linalg.solve(L.T, alpha)
                    level_mean = beta_value + np.dot(K_xs.T, alpha)

                    # Compute posterior variance
                    v = np.linalg.solve(L, K_xs)
                    level_var = k_ss - np.dot(v.T, v)

                    # Ensure positive variance
                    level_var = max(float(level_var), 1e-8)

                    level_means.append(float(level_mean))
                    level_vars.append(level_var)
            else:
                # For level 1, use standard GP prediction
                K_xx = self._compute_gram_matrix(
                    X_arrays[0], X_arrays[0], ls_values, sig_value
                )
                K_xs = self._compute_gram_matrix(
                    X_arrays[0], x_array, ls_values, sig_value
                )
                k_ss = sig_value

                try:
                    nug = float(trace.posterior["nug_L1"].mean().values)
                except (KeyError, AttributeError):
                    nug = 1e-6

                K_xx += np.eye(len(X_arrays[0])) * nug

                try:
                    L = np.linalg.cholesky(K_xx)
                except np.linalg.LinAlgError:
                    K_xx += np.eye(len(X_arrays[0])) * 1e-4
                    L = np.linalg.cholesky(K_xx)

                alpha = np.linalg.solve(L, y_arrays[0] - beta_value)
                alpha = np.linalg.solve(L.T, alpha)
                level_mean = beta_value + np.dot(K_xs.T, alpha)

                v = np.linalg.solve(L, K_xs)
                level_var = k_ss - np.dot(v.T, v)
                level_var = max(float(level_var), 1e-8)

                level_means.append(float(level_mean))
                level_vars.append(level_var)

        # Combine predictions from all levels
        # For multi-level models, the final prediction is the sum of all levels
        final_mean = sum(level_means)
        final_var = sum(level_vars)  # Assuming independence between levels

        return GaussianProcessPrediction(final_mean, final_var)

    def _compute_gram_matrix(self, X1, X2, lengthscales, sigma):
        """Helper to compute covariance matrices using squared exponential kernel"""
        # Compute pairwise squared distances
        sq_dist = np.zeros((X1.shape[0], X2.shape[0]))

        for i in range(X1.shape[0]):
            for j in range(X2.shape[0]):
                for d in range(self._input_dims):
                    sq_dist[i, j] += ((X1[i, d] - X2[j, d]) / lengthscales[d]) ** 2

        # Apply squared exponential kernel
        return sigma * np.exp(-0.5 * sq_dist)

    def _get_training_arrays(self):
        """Extract training arrays from stored training data."""
        X_arrays = []
        y_arrays = []

        for level in range(1, self._levels + 1):
            level_data = self._training_data[level]
            X_level = np.array([[coord for coord in datum.input] for datum in level_data])
            y_level = np.array([datum.output for datum in level_data])

            X_arrays.append(X_level)
            y_arrays.append(y_level)

        return X_arrays, y_arrays

    def correlation(
        self, inputs1: Sequence[Input], inputs2: Sequence[Input]
    ) -> np.ndarray:
        """
        Compute the correlation matrix between two sets of inputs.

        Parameters
        ----------
        inputs1, inputs2 : Sequence[Input]
            Sequences of simulator inputs

        Returns
        -------
        numpy.ndarray
            Correlation matrix of shape (len(inputs1), len(inputs2))
        """
        if not self._fit_hyperparameters:
            return np.array([])

        if not inputs1 or not inputs2:
            return np.array([])

        # Extract length scales
        corr_length_scales = self._fit_hyperparameters.corr_length_scales

        # Convert inputs to numpy arrays
        X1 = np.array([[coord for coord in x] for x in inputs1])
        X2 = np.array([[coord for coord in x] for x in inputs2])

        # Compute squared distances with appropriate scaling
        n1, n2 = len(inputs1), len(inputs2)
        K = np.zeros((n1, n2))

        for i in range(n1):
            for j in range(n2):
                # Compute squared distance
                dist_sq = 0
                for d in range(self._input_dims):
                    dist_sq += ((X1[i, d] - X2[j, d]) / corr_length_scales[d]) ** 2

                # Apply squared exponential kernel
                K[i, j] = np.exp(-0.5 * dist_sq)

        return K
