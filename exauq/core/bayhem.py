"""
Bayesian Hierarchical Emulator for Multi-level simulations (BayHEM): a multi-level
Gaussian process emulator, fitted with PyMC, in which each level's GP is conditioned on
the posterior of the level below it.

This module requires the optional ``bayhem`` extra (``pip install exauq[bayhem]``),
which is only available on Python 3.12 and above.

[BayHEMGP][exauq.core.bayhem.BayHEMGP]
---------------------------------------------------------------------------------------
[`fit`][exauq.core.bayhem.BayHEMGP.fit]
Fit the multi-level emulator to `MultiLevel` training data by MCMC sampling or MAP.

[`predict`][exauq.core.bayhem.BayHEMGP.predict]
Make a prediction for the top-level simulator output given an Input.

[`fit_hyperparameters`][exauq.core.bayhem.BayHEMGP.fit_hyperparameters]
**(Read-Only)** Fitted hyperparameters for each level.

[`training_data`][exauq.core.bayhem.BayHEMGP.training_data]
**(Read-only)** The data on which the emulator has been trained.


[BayHEMGPHyperparameters][exauq.core.bayhem.BayHEMGPHyperparameters]
---------------------------------------------------------------------------------------
[`set_prior`][exauq.core.bayhem.BayHEMGPHyperparameters.set_prior]
Set the PyMC prior distribution for a hyperparameter at a given level.

"""

from __future__ import annotations

from collections.abc import Sequence
from typing import Optional, Tuple
from warnings import warn

import numpy as np
from numpy.linalg import cholesky as np_cholesky
from numpy.linalg import solve as np_solve
from numpy.typing import NDArray

try:
    import pymc as pm
    from pytensor.tensor import dot, eye
    from pytensor.tensor import sum as pt_sum
    from pytensor.tensor.linalg import cholesky, solve_triangular
except ImportError as e:  # pragma: no cover
    raise ImportError(
        "exauq.core.bayhem requires PyMC, which is an optional dependency: install it "
        "with `pip install exauq[bayhem]` (Python 3.12 or later)."
    ) from e

from exauq.core.modelling import (
    AbstractGaussianProcess,
    AbstractHyperparameters,
    GaussianProcessHyperparameters,
    GaussianProcessPrediction,
    Input,
    MLTrainingData,
    MultiLevel,
    OptionalFloatPairs,
)


class BayHEMGPHyperparameters(AbstractHyperparameters):
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

        Parameters
        ----------
        model_context : pm.Model, optional
            PyMC model context where hyperparameters will be defined.
            Can be set later with set_model_context().
        input_dims : int
            Number of input dimensions (determines number of length scales)
        levels : int
            Number of levels in the multi-level GP model
        """
        self._model = model_context
        self._input_dims = input_dims
        self._levels = levels
        self._param_specs = {}  # Specifications for parameters
        self._params = {}  # Created parameter objects
        self._initialized = False

    @property
    def levels(self) -> int:
        """(Read-only) Number of levels in the multi-level GP model."""
        return self._levels

    @property
    def input_dims(self) -> int:
        """(Read-only) Number of input dimensions."""
        return self._input_dims

    def set_model_context(self, model_context):
        """
        Set the PyMC model context after initialization.

        Parameters
        ----------
        model_context : pm.Model
            PyMC model context where hyperparameters will be defined
        """
        self._model = model_context
        # Reset initialization state and params since we have a new context
        self._initialized = False
        self._params = {}
        return self

    def set_prior(self, param_name, dist_type, **dist_kwargs):
        """
        Set a prior distribution for a parameter.

        Parameters
        ----------
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

        self._param_specs[param_name] = {"dist_type": dist_type, "params": dist_kwargs}

        # Clear params to ensure reinitialization
        self._params = {}
        self._initialized = False
        return self

    def apply_defaults(self):
        """Apply default prior distributions for parameters not explicitly set"""
        # Define defaults for Level 1
        for i in range(1, self._input_dims + 1):
            if f"ls{i}_L1" not in self._param_specs:
                self._param_specs[f"ls{i}_L1"] = {
                    "dist_type": "Gamma",
                    "params": {"alpha": 2, "beta": 4},
                }

        if "sig_L1" not in self._param_specs:
            self._param_specs["sig_L1"] = {
                "dist_type": "Gamma",
                "params": {"alpha": 8, "beta": 2},
            }

        if "nug_L1" not in self._param_specs:
            self._param_specs["nug_L1"] = {
                "dist_type": "Gamma",
                "params": {"alpha": 2, "beta": 4},
            }

        if "beta_L1" not in self._param_specs:
            self._param_specs["beta_L1"] = {
                "dist_type": "Normal",
                "params": {"mu": 0, "sigma": 10},
            }

        # Set up inheritance for higher levels
        base_params = ["sig", "nug", "beta"]
        base_params.extend([f"ls{i}" for i in range(1, self._input_dims + 1)])

        for level in range(2, self._levels + 1):
            for base_param in base_params:
                param_name = f"{base_param}_L{level}"
                if param_name not in self._param_specs:
                    # Inherit from the previous level
                    prev_level_param = f"{base_param}_L{level - 1}"
                    self._param_specs[param_name] = {"inherit_from": prev_level_param}

    def initialize(self):
        """
        Initialize all hyperparameters within the model context.

        Raises
        ------
        ValueError
            If model_context has not been set
        """
        if self._initialized:
            return

        if self._model is None:
            raise ValueError(
                "Model context must be set before initialization. Use set_model_context()."
            )

        # Apply defaults for parameters not explicitly set
        self.apply_defaults()

        # Create actual parameter objects
        created_params = {}
        with self._model:
            # First pass: create all parameters that don't inherit
            for param_name, spec in self._param_specs.items():
                if "inherit_from" not in spec:
                    dist_class = getattr(pm, spec["dist_type"])
                    created_params[param_name] = dist_class(param_name, **spec["params"])

        # Second pass: resolve inheritances
        self._params = created_params.copy()
        for param_name, spec in self._param_specs.items():
            if "inherit_from" in spec:
                inherit_from = spec["inherit_from"]
                if inherit_from in self._params:
                    self._params[param_name] = self._params[inherit_from]

        self._initialized = True

    def get(self, param_name, level=None):
        """
        Get a hyperparameter.

        Parameters
        ----------
        param_name : str
            Base name of the parameter (e.g., "ls1", "sig")
        level : int, optional
            Level to get the parameter for. If None, assumes param_name
            already includes the level suffix.

        Returns
        -------
        pm.Distribution
            The requested hyperparameter

        Raises
        ------
        ValueError
            If model_context has not been set or parameter initialization failed
        """
        if self._model is None:
            raise ValueError(
                "Model context must be set before getting parameters. Use set_model_context()."
            )

        if not self._initialized:
            self.initialize()

        if level is not None:
            param_name = f"{param_name}_L{level}"

        param = self._params.get(param_name)
        if param is None:
            raise ValueError(
                f"Parameter {param_name} not found. Check parameter name and level."
            )

        return param

    def get_lengthscales(self, level):
        """
        Get all length scale parameters for a specific level as a list.

        Parameters
        ----------
        level : int
            The level to get length scales for

        Returns
        -------
        list
            List of length scale parameters for the specified level
        """
        return [self.get(f"ls{i}", level=level) for i in range(1, self._input_dims + 1)]

    def get_all_for_level(self, level):
        """
        Get all hyperparameters for a specific level.

        Parameters
        ----------
        level : int
            The level to get parameters for

        Returns
        -------
        dict
            Dictionary of parameter names to parameter objects
        """
        if self._model is None:
            raise ValueError(
                "Model context must be set before getting parameters. Use set_model_context()."
            )

        if not self._initialized:
            self.initialize()

        level_params = {}
        for param_name in self._params:
            if f"_L{level}" in param_name:
                base_name = param_name.split(f"_L{level}")[0]
                level_params[base_name] = self._params[param_name]

        return level_params

    def get_process_sd(self, level):
        """Get process SD parameter, sigma, for the specified level"""
        return self.get("sig", level)

    def get_nugget(self, level):
        """Get nugget parameter for the specified level"""
        return self.get("nug", level)

    def get_mean_constant(self, level):
        """Get mean constant parameter for the specified level"""
        return self.get("beta", level)


class PosteriorCovariance(pm.gp.cov.Covariance):
    def __init__(self, prior_cov, X_train, level, nugget):
        """
        Posterior covariance function for multi-level Gaussian Process.

        Parameters
        ----------
        prior_cov: Instance of a PyMC covariance function (e.g., SquaredExponential)
        X_train: Training inputs (N x D)
        level: Current level in the multi-level structure (1-indexed)
        nugget: Nugget standard deviation
        """
        input_dim = X_train[0].shape[1]
        super(PosteriorCovariance, self).__init__(input_dim)
        self.prior_cov = prior_cov
        self.X_train = X_train
        self.nugget = nugget
        self.level = level

    def full(self, X, Xs=None):
        """Compute the posterior covariance matrix"""
        # Default Xs to X if not provided
        if Xs is None:
            Xs = X

        # Compute prior covariances
        K_xx = self.prior_cov(self.X_train[self.level - 1], self.X_train[self.level - 1])
        K_xs = self.prior_cov(self.X_train[self.level - 1], X)
        K_ss = self.prior_cov(X)

        # Add noise to training covariance
        input_dim = K_xx.shape[0]

        noise_matrix = eye(input_dim) * self.nugget**2
        L_xx = cholesky(K_xx + noise_matrix)
        L_inv = solve_triangular(L_xx, eye(L_xx.shape[0]), lower=True)
        K_xx_inv = dot(L_inv.T, L_inv)

        # Compute posterior covariance
        K_post = K_ss - dot(dot(K_xs.T, K_xx_inv), K_xs)
        return K_post

    def diag(self, X):
        """Diagonal elements of posterior covariance (predictive variance)"""
        K_ss_diag = self.prior_cov.diag(X)
        K_xs = self.prior_cov(self.X_train[self.level - 1], X)
        K_xx = self.prior_cov(self.X_train[self.level - 1])
        input_dim = K_xx.shape[0]

        noise_matrix = eye(input_dim) * self.nugget**2

        L_xx = cholesky(K_xx + noise_matrix)
        L_inv = solve_triangular(L_xx, eye(input_dim), lower=True)
        K_xx_inv = dot(L_inv.T, L_inv)

        # Compute diagonal elements of posterior covariance
        diag_K_post = K_ss_diag - pt_sum(dot(K_xs.T, K_xx_inv) * K_xs.T, axis=0)

        return diag_K_post


class PosteriorMean(pm.gp.mean.Mean):
    def __init__(self, prior_mean, prior_cov, X_train, Y_train, level, nugget):
        """
        Posterior mean function for multi-level Gaussian Process.

        Parameters
        ----------
        prior_mean: Instance of a PyMC mean function
        prior_cov: Instance of a PyMC covariance function
        X_train: Training inputs (N x D)
        Y_train: Training outputs
        level: Current level in the multi-level structure (1-indexed)
        nugget: Nugget standard deviation
        """
        super(PosteriorMean, self).__init__()
        self.prior_mean = prior_mean
        self.prior_cov = prior_cov
        self.X_train = X_train
        self.Y_train = Y_train
        self.nugget = nugget
        self.level = level

    def __call__(self, X):
        """Compute the posterior mean"""
        K_xx = self.prior_cov(self.X_train[self.level - 1], self.X_train[self.level - 1])
        K_xs = self.prior_cov(self.X_train[self.level - 1], X)

        # Add noise to training covariance
        input_dim = K_xx.shape[0]

        noise_matrix = eye(input_dim) * self.nugget**2
        L_xx = cholesky(K_xx + noise_matrix)
        L_inv = solve_triangular(L_xx, eye(L_xx.shape[0]), lower=True)
        K_xx_inv = dot(L_inv.T, L_inv)

        M_post = self.prior_mean(X) + dot(
            dot(K_xs.T, K_xx_inv),
            (
                self.Y_train[self.level - 1]
                - self.prior_mean(self.X_train[self.level - 1])
            ),
        )
        return M_post


class BayHEMGP(AbstractGaussianProcess[MLTrainingData]):
    """
    Bayesian Hierarchical multi-level GP emulator using PyMC.

    This class implements a multi-level Gaussian Process where each level
    builds upon the posterior distribution of the previous level.
    The number of input dimensions and levels are determined automatically
    from the training data.
    """

    def __init__(self):
        """
        Initialize a BayHEMGP.

        The number of input dimensions and levels will be determined
        automatically from the training data when fit() is called.
        """
        self._input_dims = None
        self._levels = None
        self._training_data = None
        self._fit_hyperparameters = None
        self._model = None
        self._trace = None
        self._MAP = None

    @property
    def training_data(self) -> Optional[MLTrainingData]:
        """The data on which the emulator has been trained, or None if not fitted."""
        return self._training_data

    @property
    def fit_hyperparameters(
        self,
    ) -> Optional[MultiLevel[GaussianProcessHyperparameters]]:
        """The fitted hyperparameters for each level (posterior means when sampling,
        MAP estimates otherwise), or ``None`` if the emulator has not been fitted."""
        return self._fit_hyperparameters

    @property
    def kinv(self) -> NDArray:
        """Not implemented for BayHEMGP; see issue #452."""
        raise NotImplementedError(
            "'kinv' is not implemented for BayHEMGP; see "
            "https://github.com/EXA-UQ/EXAUQ-Toolbox/issues/452"
        )

    def covariance_matrix(self, inputs: Sequence[Input]) -> NDArray:
        """Not implemented for BayHEMGP; see issue #452."""
        raise NotImplementedError(
            "'covariance_matrix' is not implemented for BayHEMGP; see "
            "https://github.com/EXA-UQ/EXAUQ-Toolbox/issues/452"
        )

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

            if not level_data:
                raise ValueError(
                    f"Training data must contain at least one point at level {level}"
                )

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
        hyperparameters: Optional[BayHEMGPHyperparameters] = None,
        hyperparameter_bounds: Optional[Sequence[OptionalFloatPairs]] = None,
        MAP: bool = False,
        draws: int = 1000,
        tune: int = 1000,
        sample_kwargs: Optional[dict] = None,
    ) -> None:
        """
        Fit the BayHEMGP to data.

        Parameters
        ----------
        training_data : MLTrainingData
            Multi-level training data
        hyperparameters : Optional[BayHEMGPHyperparameters]
            Hyperparameters to use. If None, default priors will be used.
        hyperparameter_bounds : Optional[Sequence[OptionalFloatPairs]]
            Bounds for hyperparameter estimation. Not yet implemented.
        MAP : bool
            If True, use MAP estimation. If False (default), use full MCMC sampling.
        draws : int
            Number of posterior samples to draw (default 1000).
        tune : int
            Number of warmup/tuning samples (default 1000).
        sample_kwargs : Optional[dict]
            Extra keyword arguments passed to ``pymc.sample`` (e.g. ``chains``,
            ``cores``, ``random_seed``, ``progressbar``). Ignored when ``MAP`` is True.
        """
        X_arrays, y_arrays, combined_data = self._organize_training_data(training_data)

        # Create hyperparameters
        if hyperparameters is None:
            # Initialize hyperparameters with default priors
            hparams = self._create_default_hyperparameters(hyperparameter_bounds)
        else:
            hparams = hyperparameters

        # Check that the hyperparameters are consistent with the training data
        if hparams.levels != len(self.training_data):
            raise ValueError(
                f"Expected {len(self.training_data)} levels in hyperparameters, "
                f"but received {hparams.levels}."
            )

        n_inputs = len(self.training_data[1][0].input)
        if hparams.input_dims != n_inputs:
            raise ValueError(
                f"Expected {n_inputs} input dimensions in hyperparameters, "
                f"but received {hparams.input_dims}."
            )

        # Create and sample from the model
        with pm.Model() as model:
            # Set model context for hyperparameters
            hparams.set_model_context(model)
            hparams.initialize()  # Ensure all hyperparameters are initialized

            # Level 1 GP (Base level)
            length_scales_L1 = hparams.get_lengthscales(1)
            sig_L1 = hparams.get_process_sd(1)
            nug_L1 = hparams.get_nugget(1)
            beta_L1 = hparams.get_mean_constant(1)

            cov1 = sig_L1**2 * pm.gp.cov.ExpQuad(self._input_dims, ls=length_scales_L1)
            mean1 = pm.gp.mean.Constant(beta_L1)
            gp1 = pm.gp.Marginal(mean_func=mean1, cov_func=cov1)
            y_obs1 = gp1.marginal_likelihood(
                "y_obs1", X=X_arrays[0], y=y_arrays[0], sigma=nug_L1
            )

            # Build up each level in sequence with correct posteriors
            prev_covs = [cov1]  # Track all previous covariance functions

            for level in range(2, self._levels + 1):
                # Get level-specific hyperparameters
                length_scales = hparams.get_lengthscales(level)
                sig = hparams.get_process_sd(level)
                nug = hparams.get_nugget(level)
                beta = hparams.get_mean_constant(level)

                # Create level-specific prior cov and mean
                cov_prior = sig**2 * pm.gp.cov.ExpQuad(self._input_dims, ls=length_scales)
                mean_prior = pm.gp.mean.Constant(beta)

                # Build up posterior functions from previous levels
                level_cov = cov_prior
                level_mean = mean_prior

                # For each previous level, create posterior for current level
                for prev_level in range(1, level):
                    idx = prev_level - 1  # 0-indexed arrays

                    # Get nugget from corresponding level
                    prev_nug = hparams.get_nugget(prev_level)

                    # Create posterior from previous level
                    level_cov = PosteriorCovariance(
                        level_cov,
                        X_arrays,
                        prev_level,
                        prev_nug,
                    )
                    level_mean = PosteriorMean(
                        level_mean,
                        prev_covs[idx],
                        X_arrays,
                        y_arrays,
                        prev_level,
                        prev_nug,
                    )

                prev_covs.append(level_cov)

                # Create GP for this level
                gp_level = pm.gp.Marginal(mean_func=level_mean, cov_func=level_cov)
                y_obs_level = gp_level.marginal_likelihood(
                    f"y_obs{level}",
                    X=X_arrays[level - 1],
                    y=y_arrays[level - 1],
                    sigma=nug,
                )

            # Sample (clearing any previous fit first)
            self._MAP = None
            self._trace = None
            if MAP is True:
                self._MAP = pm.find_MAP()
                self._fit_hyperparameters = MultiLevel(
                    {
                        i: self._extract_hyperparameters_from_MAP(level=i)
                        for i in range(1, self._levels + 1)
                    }
                )

            else:
                trace = pm.sample(
                    **{
                        "draws": draws,
                        "tune": tune,
                        "return_inferencedata": True,
                        "target_accept": 0.95,
                        **(sample_kwargs or {}),
                    }
                )
                self._trace = trace
                self._fit_hyperparameters = MultiLevel(
                    {
                        i: self._extract_hyperparameters_from_trace(trace, level=i)
                        for i in range(1, self._levels + 1)
                    }
                )

        self._model = model

    def _create_default_hyperparameters(self, bounds=None):
        """
        Create default hyperparameters for the model.

        Parameters
        ----------
        bounds : Optional[Sequence[OptionalFloatPairs]]
            Bounds for hyperparameter estimation

        Returns
        -------
        BayHEMGPHyperparameters
            Configured hyperparameters object
        """

        if self._input_dims is None or self._levels is None:
            raise ValueError(
                "Cannot create hyperparameters before fitting. Input dimensions and levels unknown."
            )

        hparams = BayHEMGPHyperparameters(
            input_dims=self._input_dims, levels=self._levels
        )

        # Set default priors for all length scales
        for i in range(1, self._input_dims + 1):
            hparams.set_prior(f"ls{i}", "Gamma", alpha=2, beta=4)

        # Set other default hyperparameters
        hparams.set_prior("sig", "Gamma", alpha=8, beta=2).set_prior(
            "nug", "Gamma", alpha=2, beta=4
        ).set_prior("beta", "Normal", mu=0, sigma=10)

        # Bounds are accepted for API compatibility but not yet implemented
        if bounds is not None:
            warn(
                "hyperparameter_bounds is not yet implemented for BayHEMGP and will be ignored. "
                "Consider using custom priors via BayHEMGPHyperparameters to constrain parameters."
            )

        return hparams

    def _extract_hyperparameters_from_trace(self, trace, level=1):
        """
        Extract hyperparameters from the sampling trace.

        Parameters
        ----------
        trace : PyMC inference data
            Trace from sampling
        level: which level to extract the hyperparameters from

        Returns
        -------
        GaussianProcessHyperparameters
            Fitted hyperparameters
        """
        # Extract posterior means for key parameters
        ls_values = []

        for i in range(1, self._input_dims + 1):
            param_name = f"ls{i}_L{level}"
            if param_name in trace.posterior:
                ls_values.append(float(trace.posterior[param_name].mean().values))
            else:
                for prev_level in range(level - 1, 0, -1):
                    param_name = f"ls{i}_L{prev_level}"
                    if param_name in trace.posterior:
                        ls_values.append(float(trace.posterior[param_name].mean().values))
                        break
                else:
                    raise ValueError(
                        f"Required parameter {param_name} not found in trace"
                    )

        param_name = f"sig_L{level}"
        if param_name in trace.posterior:
            sig_value = float(trace.posterior[param_name].mean().values)
        else:
            for prev_level in range(level - 1, 0, -1):
                param_name = f"sig_L{prev_level}"
                if param_name in trace.posterior:
                    sig_value = float(trace.posterior[param_name].mean().values)
                    break
            else:
                raise ValueError(f"Required parameter {param_name} not found in trace")

        param_name = f"nug_L{level}"
        if param_name in trace.posterior:
            nug_value = float(trace.posterior[param_name].mean().values)
        else:
            for prev_level in range(level - 1, 0, -1):
                param_name = f"nug_L{prev_level}"
                if param_name in trace.posterior:
                    nug_value = float(trace.posterior[param_name].mean().values)
                    break
            else:
                raise ValueError(f"Required parameter {param_name} not found in trace")

        # Create GaussianProcessHyperparameters
        return GaussianProcessHyperparameters(
            corr_length_scales=ls_values, process_var=sig_value**2, nugget=nug_value
        )

    def _extract_hyperparameters_from_MAP(self, level=1):
        """
        Extract hyperparameters from find_MAP output.

        Parameters
        ----------
        MAP : PyMC output
            Maximum a posteriori for model
        level: which level to extract the hyperparameters from

        Returns
        -------
        GaussianProcessHyperparameters
            Fitted hyperparameters
        """
        # Extract MAP for each parameter
        ls_values = []

        for i in range(1, self._input_dims + 1):
            param_name = f"ls{i}_L{level}"
            if param_name in self._MAP:
                ls_values.append(float(self._MAP[param_name]))
            else:
                for prev_level in range(level - 1, 0, -1):
                    param_name = f"ls{i}_L{prev_level}"
                    if param_name in self._MAP:
                        ls_values.append(float(self._MAP[param_name]))
                        break
                else:
                    raise ValueError(
                        f"Required parameter {param_name} not found in model"
                    )

        param_name = f"sig_L{level}"
        if param_name in self._MAP:
            sig_value = float(self._MAP[param_name])
        else:
            for prev_level in range(level - 1, 0, -1):
                param_name = f"sig_L{prev_level}"
                if param_name in self._MAP:
                    sig_value = float(self._MAP[param_name])
                    break
            else:
                raise ValueError(f"Required parameter {param_name} not found in model")

        param_name = f"nug_L{level}"
        if param_name in self._MAP:
            nug_value = float(self._MAP[param_name])
        else:
            for prev_level in range(level - 1, 0, -1):
                param_name = f"nug_L{prev_level}"
                if param_name in self._MAP:
                    nug_value = float(self._MAP[param_name])
                    break
            else:
                raise ValueError(f"Required parameter {param_name} not found in model")

        # Create GaussianProcessHyperparameters
        return GaussianProcessHyperparameters(
            corr_length_scales=ls_values, process_var=sig_value**2, nugget=nug_value
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
            The predicted mean (estimate) and variance for the input
        """
        if self._trace is None and self._MAP is None:
            raise ValueError("Model has not been fitted yet. Call fit() first.")

        if not isinstance(x, Input):
            raise TypeError(f"Expected 'x' to be of type Input, but received {type(x)}")

        if len(x) != self._input_dims:
            raise ValueError(
                f"Expected input of dimension {self._input_dims}, but received {len(x)}"
            )

        # Convert input to numpy array (single point)
        x_pred = np.array([coord for coord in x]).reshape(1, -1)

        # Extract training data
        X_arrays, y_arrays = self._get_training_arrays()

        # Helper to get beta for a level (from _fit_hyperparameters or raw MAP/trace)
        def get_beta(level_idx):
            """Get mean constant for a level, with fallback to 0."""
            # Try to get from MAP first
            if self._MAP:
                param_name = f"beta_L{level_idx}"
                if param_name in self._MAP:
                    return float(self._MAP[param_name])
                # Fall back through previous levels
                for prev in range(level_idx - 1, 0, -1):
                    param_name = f"beta_L{prev}"
                    if param_name in self._MAP:
                        return float(self._MAP[param_name])
                return 0.0
            else:
                # Trace mode
                param_name = f"beta_L{level_idx}"
                if param_name in self._trace.posterior:
                    return float(self._trace.posterior[param_name].mean().values)
                # Fall back through previous levels
                for prev in range(level_idx - 1, 0, -1):
                    param_name = f"beta_L{prev}"
                    if param_name in self._trace.posterior:
                        return float(self._trace.posterior[param_name].mean().values)
                return 0.0

        # Level 1: Build base prior functions
        hp_L1 = self._fit_hyperparameters[1]
        ls_L1 = hp_L1.corr_length_scales
        sig2_L1 = hp_L1.process_var
        beta_L1 = get_beta(1)

        prior_cov_L1 = lambda X1, X2, ls=ls_L1, s2=sig2_L1: sq_exp_cov(X1, X2, ls, s2)
        prior_mean_L1 = beta_L1

        # Track covariance and mean functions for each level (for use in higher levels)
        all_cov_funcs = [prior_cov_L1]
        all_mean_funcs = [prior_mean_L1]

        # Build posterior functions for levels 2+
        for level in range(2, self._levels + 1):
            # Get this level's hyperparameters
            hp = self._fit_hyperparameters[level]
            ls = hp.corr_length_scales
            sig2 = hp.process_var
            beta = get_beta(level)

            # Create this level's prior functions (with default args to capture values)
            level_prior_cov = lambda X1, X2, ls=ls, s2=sig2: sq_exp_cov(X1, X2, ls, s2)
            level_prior_mean = beta

            # Start with this level's prior, then condition on all previous levels
            level_cov = level_prior_cov
            level_mean = level_prior_mean

            for prev_level in range(1, level):
                prev_nug = self._fit_hyperparameters[prev_level].nugget

                # Wrap with posterior conditioning from prev_level's data
                level_cov = PosteriorCovarianceNumeric(
                    level_cov, X_arrays, prev_nug, level=prev_level
                )
                level_mean = PosteriorMeanNumeric(
                    level_mean,
                    all_cov_funcs[prev_level - 1],
                    X_arrays,
                    y_arrays,
                    prev_nug,
                    level=prev_level,
                )

            # Store for use by higher levels
            all_cov_funcs.append(level_cov)
            all_mean_funcs.append(level_mean)

        # Final prediction uses the top level's posterior
        top_level = self._levels
        top_nug = self._fit_hyperparameters[top_level].nugget

        # Get the accumulated posterior from the loop (or L1 prior if only 1 level)
        if self._levels == 1:
            final_cov = prior_cov_L1
            final_mean = prior_mean_L1
        else:
            final_cov = all_cov_funcs[-1]
            final_mean = all_mean_funcs[-1]

        post_mean_fn = PosteriorMeanNumeric(
            final_mean, final_cov, X_arrays, y_arrays, top_nug, level=top_level
        )
        post_cov_fn = PosteriorCovarianceNumeric(
            final_cov, X_arrays, top_nug, level=top_level
        )

        # Predict for new data
        mu = post_mean_fn(X_new=x_pred)
        Sigma = post_cov_fn(X=x_pred)

        # Extract scalar values for single-point prediction
        estimate = float(mu[0])
        variance = float(Sigma[0, 0])

        return GaussianProcessPrediction(estimate=estimate, variance=variance)

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
        Compute the correlation matrix between two sets of inputs, using the squared
        exponential kernel with the top level's fitted correlation length scales.

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

        # ponytail: top level's length scales only; per-level access is issue #452
        corr_length_scales = self._fit_hyperparameters[self._levels].corr_length_scales
        X1 = np.array([[coord for coord in x] for x in inputs1])
        X2 = np.array([[coord for coord in x] for x in inputs2])
        return sq_exp_cov(X1, X2, corr_length_scales, 1.0)


def sq_exp_cov(X1, X2, lengthscales, sigma2):
    X1 = np.atleast_2d(X1)
    X2 = np.atleast_2d(X2)
    ls = np.asarray(lengthscales, dtype=float)
    X1s = X1 / ls
    X2s = X2 / ls
    sq = (
        np.sum(X1s**2, axis=1)[:, None]
        + np.sum(X2s**2, axis=1)[None, :]
        - 2.0 * (X1s @ X2s.T)
    )
    return float(sigma2) * np.exp(-0.5 * sq)


# --- Posterior covariance numeric (now exposes posterior_cross_cov) ---
class PosteriorCovarianceNumeric:
    def __init__(self, prior_cov, X_arrays, nugget, level=1):
        """
        prior_cov: either
           - a numeric kernel function k(X1, X2)  OR
           - a PosteriorCovarianceNumeric instance representing previous-level posterior
        X_arrays: list of training arrays per level
        nugget: scalar noise std for THIS level (added to K_tt for this level)
        level: 1-based level index
        """
        self.prior_cov = prior_cov
        self.X_arrays = X_arrays
        self.nugget = float(nugget)
        self.level = int(level)

    def _prior_function(self):
        """
        Return a callable prior(X1,X2) used at *this* level.
        If prior_cov is a previous-level PosteriorCovarianceNumeric, the prior function
        is that previous object's posterior_cross_cov.
        If prior_cov is a kernel callable, we use it directly.
        """
        if isinstance(self.prior_cov, PosteriorCovarianceNumeric):
            return (
                self.prior_cov.posterior_cross_cov
            )  # callable(X1,X2) -> K_post_prev(X1,X2)
        else:
            return self.prior_cov  # callable kernel

    def posterior_cross_cov(self, X1, X2):
        """
        Return posterior covariance at THIS level between arbitrary sets X1 and X2:
          K_post(X1,X2) = K_prior(X1,X2)
                          - K_prior(X1, X_train) @ inv(K_prior(X_train,X_train) + nugget^2 I) @ K_prior(X_train, X2)
        where K_prior is the 'prior function' for this level: either base kernel OR previous level posterior.
        """
        X1 = np.atleast_2d(X1)
        X2 = np.atleast_2d(X2)
        prior = self._prior_function()
        X_train = np.atleast_2d(self.X_arrays[self.level - 1])

        # Prior covariances
        K_ss = prior(X1, X2)
        K_1x = prior(X1, X_train)
        K_xx = prior(X_train, X_train)
        K_x2 = prior(X_train, X2)

        # Add nugget for this level
        K_xx = K_xx + np.eye(K_xx.shape[0]) * (self.nugget**2)

        L = np_cholesky(K_xx)
        Linv = np_solve(L, np.eye(L.shape[0]))
        K_tt_inv = Linv.T @ Linv
        return K_ss - K_1x @ (K_tt_inv @ K_x2)

    def __call__(self, X):
        """
        Return posterior covariance K_post(X, X) (square matrix).
        """
        return self.posterior_cross_cov(X, X)


class PosteriorMeanNumeric:
    def __init__(self, prior_mean, prior_cov, X_arrays, y_arrays, nugget, level=1):
        """
        prior_mean: either a numeric constant, a callable mean(X) or a PosteriorMeanNumeric (previous level)
        prior_cov: either a numeric kernel or a PosteriorCovarianceNumeric (previous-level posterior)
        X_arrays / y_arrays: lists of training arrays per level
        nugget: scalar for THIS level
        level: 1-based
        """
        self.prior_mean = prior_mean
        self.prior_cov = prior_cov
        self.X_arrays = X_arrays
        self.y_arrays = y_arrays
        self.nugget = float(nugget)
        self.level = int(level)

    def _eval_prior_mean(self, X):
        X = np.atleast_2d(X)
        if isinstance(self.prior_mean, PosteriorMeanNumeric):
            return self.prior_mean(X)
        elif callable(self.prior_mean):
            return np.atleast_1d(self.prior_mean(X))
        else:
            return np.full(X.shape[0], float(self.prior_mean))

    def __call__(self, X_new):
        X_new = np.atleast_2d(X_new)
        X_train = np.atleast_2d(self.X_arrays[self.level - 1])
        y_train = np.atleast_1d(self.y_arrays[self.level - 1])

        # Evaluate mean function
        mu_train = self._eval_prior_mean(X_train)
        mu_new = self._eval_prior_mean(X_new)

        # Obtain prior covariance
        if isinstance(self.prior_cov, PosteriorCovarianceNumeric):
            prior = self.prior_cov.posterior_cross_cov
        else:
            prior = self.prior_cov

        # Find posterior mean
        K_xx = prior(X_train, X_train) + np.eye(X_train.shape[0]) * (self.nugget**2)
        K_xs = prior(X_train, X_new)
        L = np_cholesky(K_xx)
        Linv = np_solve(L, np.eye(L.shape[0]))
        K_xx_inv = Linv.T @ Linv
        M_post = mu_new + K_xs.T @ (K_xx_inv @ (y_train - mu_train))
        return M_post
