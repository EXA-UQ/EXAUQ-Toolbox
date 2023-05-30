# Code in the 'fit' method of 'MogpEmulator' has been adapted from
# the function _fit_single_GP_MAP in the fitting.py module of mogp-emulator
# as at the following version (accessed 2023-04-28):
#
# https://github.com/alan-turing-institute/mogp-emulator/blob/72dc73a49dbab621ef5748546127da990fd81e4a/mogp_emulator/fitting.py
# 
# which is made available under the following licence:
#
#   MIT License
#  
#   Copyright (c) 2019 The Alan Turing Institute
#  
#   Permission is hereby granted, free of charge, to any person obtaining a copy
#   of this software and associated documentation files (the "Software"), to deal
#   in the Software without restriction, including without limitation the rights
#   to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
#   copies of the Software, and to permit persons to whom the Software is
#   furnished to do so, subject to the following conditions:
#  
#   The above copyright notice and this permission notice shall be included in all
#   copies or substantial portions of the Software.
#  
#   THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
#   IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
#   FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
#   AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
#   LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
#   OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
#   SOFTWARE.

from typing import Any
from collections.abc import Sequence
import math
import mogp_emulator as mogp
import numpy as np
import scipy
from exauq.core.modelling import TrainingDatum


class MogpEmulator(object):
    def __init__(self, gp: mogp.GaussianProcess):
        self._gp = self._validate_gp(gp)
        self._training_data = TrainingDatum.list_from_arrays(
            self._gp.inputs, self._gp.targets
            )
    
    @staticmethod
    def _validate_gp(gp: Any) -> mogp.GaussianProcess:
        """Checks that `gp` is an mogp GuassianProcess object, returning `gp` if
        so and raising a TypeError if not.
        """
        if not isinstance(gp, mogp.GaussianProcess):
            raise TypeError(
                "Argument 'gp' must be of type GaussianProcess from the "
                "mogp-emulator package"
                )

        return gp

    @property
    def gp(self) -> mogp.GaussianProcess:
        """(Read-only) Get the underlying mogp GaussianProcess for this
        emulator."""
        return self._gp

    @property
    def training_data(self) -> list[TrainingDatum]:
        """(Read-only) Get the data on which the emulator has been, or will be,
        trained."""
        return self._training_data
    
    def fit(self, hyperparameter_bounds : Sequence[tuple[float, float]] = None) -> None:
        """Train the emulator, including estimation of hyperparameters.

        This method will train the underlying GaussianProcess object, `self.gp`,
        that this object wraps, using the training data currently stored in
        `self.training_data`. Hyperparameters are estimated
        as part of this training, by maximising the log-posterior. If
        bounds are supplied for the hyperparameters, then the estimated
        hyperparameters will respect these bounds (the underlying maximisation
        is constrained by the bounds).

        Parameters
        ----------
        hyperparameter_bounds : sequence of tuple[float, float], optional
            (Default: None) A sequence of bounds to apply to hyperparameters
            during estimation, of the form ``(lower_bound, upper_bound)``. All
            but the last tuple should represent bounds for the correlation
            length parameters, while the last tuple should represent bounds for
            the covariance.
        """
        if hyperparameter_bounds is None:
            self._gp = mogp.fit_GP_MAP(self.gp)
            return
        
        raw_hyperparameter_bounds = self._compute_raw_param_bounds(
            hyperparameter_bounds
            )

        # NOTE (TH, 2023-05-16): from here code is adapted from the
        # mogp-emulator package (see the comment at the beginning of this file).
        # This is currently just a quick (and largely untested) implementation
        # of hyperparameter estimation that respects bounds on the
        # hyperparameters.
        # TODO (TH, 2023-05-16): This code should be revisited at a later date
        # (see #58 https://github.com/UniExeterRSE/EXAUQ-Toolbox/issues/58)
        
        np.seterr(divide = 'raise', over = 'raise', invalid = 'raise')

        logpost_values = []
        theta_values = []
        n_tries = 15  # the default in mogp
        for _ in range(n_tries):
            theta = self.gp.priors.sample()
            try:
                min_dict = scipy.optimize.minimize(
                    self.gp.logposterior, theta, method = "L-BFGS-B",
                    jac = self.gp.logpost_deriv,
                    bounds=raw_hyperparameter_bounds
                    )

                min_theta = min_dict['x']
                min_logpost = min_dict['fun']

                logpost_values.append(min_logpost)
                theta_values.append(min_theta)

            except scipy.linalg.LinAlgError:
                print("Matrix not positive definite, skipping this iteration")
            
            except FloatingPointError:
                print("Floating point error in optimization routine, skipping this iteration")

        if len(logpost_values) == 0:
            print("Minimization routine failed to return a value")
            self._gp.theta = None
        else:
            logpost_values = np.array(logpost_values)
            idx = np.argmin(logpost_values)

            self._gp.fit(theta_values[idx])
    
    @staticmethod
    def _compute_raw_param_bounds(bounds: Sequence[tuple[float, float]]) \
            -> tuple[tuple[float, ...]]:
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
