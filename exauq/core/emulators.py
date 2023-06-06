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

from collections.abc import Sequence
import math
from typing import Optional
from mogp_emulator import GaussianProcess
import numpy as np
from exauq.core.modelling import TrainingDatum
from exauq.utilities.mogp_fitting import fit_GP_MAP


class MogpEmulator(object):
    def __init__(self, *args, **kwargs):
        self._gp_kwargs = kwargs
        self._gp = self._make_gp(*args, **kwargs)
        self._training_data = TrainingDatum.list_from_arrays(
            self._gp.inputs, self._gp.targets
            )
    
    @staticmethod
    def _make_gp(*args, **kwargs) -> GaussianProcess:
        """Create an mogp GaussianProcess, raising a RuntimeError if this fails.
        """
        try:
            return GaussianProcess(*args, **kwargs)

        except BaseException:
            msg = ("Could not construct an underlying mogp-emulator "
                   "GaussianProcess during initialisation")
            raise RuntimeError(msg)

    @property
    def gp(self) -> GaussianProcess:
        """(Read-only) Get the underlying mogp GaussianProcess for this
        emulator."""
        return self._gp

    @property
    def training_data(self) -> list[TrainingDatum]:
        """(Read-only) Get the data on which the emulator has been, or will be,
        trained."""
        return self._training_data
    
    def fit(
            self, training_data: Optional[list[TrainingDatum]] = None,
            hyperparameter_bounds : Sequence[tuple[float, float]] = None
            ) -> None:
        """Train the emulator, including estimation of hyperparameters.

        This method will train the underlying ``GaussianProcess``, `self.gp`,
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
        if self._is_empty(training_data) and self._is_empty(self._training_data):
            raise ValueError(
                ("Cannot fit emulator if no training data supplied and the "
                 "'training_data' property is empty")
            )
        elif training_data is not None:
            inputs = np.array([datum.input.value for datum in training_data])
            targets = np.array([datum.output for datum in training_data])
            self._gp = GaussianProcess(inputs, targets, **self._gp_kwargs)

        if hyperparameter_bounds is None:
            self._gp = fit_GP_MAP(self.gp)
            if training_data is not None:
                self._training_data = training_data
            
            return None
        
        raw_hyperparameter_bounds = self._compute_raw_param_bounds(
            hyperparameter_bounds
            )
        self._gp = fit_GP_MAP(self.gp, bounds=raw_hyperparameter_bounds)
        
        return None
    
    @staticmethod
    def _is_empty(_list: Optional[list]) -> bool:
        return _list is None or len(_list) == 0

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
