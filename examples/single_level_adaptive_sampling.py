# A quick-and-dirty implementation of cross-validation-based single level
# adaptive sampling experimental design, using mogp-emulator
#
# Code in the function fit_with_hyperparameter_estimation has been adapted from
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

import itertools
import math
import numpy as np
import mogp_emulator
import scipy


TOLERANCE = 1e-09

class Domain:
    """Represents a domain on which a simulator function is defined. Currently
    only models d-dimensional cubes. `interval` should be a tuple defining the
    lower and upper bounds of each input coordinate, while `dim` gives the
    number of input dimensions."""
    def __init__(self, interval, dim):
        self.dim = dim
        self.interval = interval
        self.raw_parameter_bounds = self._calculate_raw_parameter_bounds(interval, dim)
        self._bounds = self._build_bounds()
    
    def _calculate_raw_parameter_bounds(self, interval, dim):
        """Calculate the bounds on raw hyperparameters needed when fitting a GP.
        
        See https://mogp-emulator.readthedocs.io/en/latest/implementation/GPParams.html#mogp_emulator.GPParams.GPParams
        for information on how the raw parameters are derived from the
        covariance and correlation length scale parameters.
        """
        length = abs(interval[1] - interval[0])
        theta_lb = 0.25 * length / math.sqrt(math.log(10))  # from HM et al paper
        
        # The lower bounds on the correlation length scale params
        # correspond to upper bounds on the raw correlation parameters, because
        # the transformation is a decreasing function: theta_raw = -2 * log(theta) 
        raw_theta_ub = to_mogp_raw_correlation(theta_lb)

        # NOTE: we are hardcoding that the covariance parameter is the last
        # coordinate for the bounds. In general for mogp, would need to look
        # at the cov_index attribute of a GPParams object. 
        lower_bounds = [-np.inf] * (dim + 1)
        upper_bounds = ([raw_theta_ub] * dim) + [np.inf]  # covariance unrestricted
        return scipy.optimize.Bounds(lower_bounds, upper_bounds)
    
    def _build_bounds(self):
        """Create bounds for optimising functions over this domain."""
        lower_bounds = [self.interval[0]] * self.dim
        upper_bounds = [self.interval[1]] * self.dim
        return scipy.optimize.Bounds(lower_bounds, upper_bounds)
    
    def latin_hypercube_samples(self, n_samples):
        """Generate latin hypercube samples of points from this domain."""
        one_shot_designer = mogp_emulator.LatinHypercubeDesign(self.dim, self.interval)
        return one_shot_designer.sample(n_samples)

    def find_pseudopoints(self, experiments):
        """Find the pseudo points in the domain for a given set of experiments."""
        # Get corners of domain
        intervals = [self.interval] * self.dim
        pseudopoints = [np.array(x) for x in itertools.product(*intervals)]

        # Project nearest point onto boundaries 
        for d, interval in enumerate(intervals):
            lb, ub = interval
            pseudopoints.append(get_boundary_pseudopoint(experiments, d, lower=lb))
            pseudopoints.append(get_boundary_pseudopoint(experiments, d, upper=ub))

        return pseudopoints

    def argmax(self, objective_function):
        """Find the point in this domain that maximises a function."""
        # For now, we just use some optimisation routine without worrying about
        # whether it's a good one to use. Using dual annealing for finding the
        # global optimum; see
        # https://docs.scipy.org/doc/scipy/reference/generated/scipy.optimize.dual_annealing.html
        obj_fn = lambda x: -objective_function(x)
        result = scipy.optimize.dual_annealing(obj_fn, self._bounds)
        return result.x


def get_boundary_pseudopoint(experiments, d, lower=None, upper=None):
    """Get the pseudopoint corresponding to one face of the cube defining this
    domain (not including corner points)."""
    if lower is not None:
        exp_idx = np.argmin([x[d] - lower for x in experiments])
        return replace(experiments[exp_idx], d, lower)
    elif upper is not None:
        exp_idx = np.argmin([upper - x[d] for x in experiments])
        return replace(experiments[exp_idx], d, upper)
    else:
        return None


def to_mogp_raw_correlation(theta):
    """Transform a correlation length scale parameter into a raw correlation
    parameter as used in MOGP.
    
    See https://mogp-emulator.readthedocs.io/en/latest/implementation/GPParams.html#mogp_emulator.GPParams.GPParams.corr
    """
    return - 2 * math.log(theta)


def to_mogp_raw_covariance(cov):
    """Transform a covariance parameter into a raw covariance parameter as used
    in MOGP.
    
    See https://mogp-emulator.readthedocs.io/en/latest/implementation/GPParams.html#mogp_emulator.GPParams.GPParams.cov
    """
    return math.log(cov)


def replace(x, d, new):
    """Create a copy of an array with the `d`th element replaced with a new value."""
    y = np.copy(x)
    y[d] = new
    return y


class PEICalculator:
    """Encapsulates the computation of pseudo-expected improvement of GP."""
    def __init__(self, gp, pseudopoints):
        self.gp = gp
        self.pseudopoints = pseudopoints

        # Cache various values / objects for later computation
        self._max_targets = max(gp.targets)
        self._norm = scipy.stats.norm(loc=0, scale=1)
        self._kernel = mogp_emulator.Kernel.Matern52()
    
    def _expected_improvement(self, x):
        """Computes expected improvement at an input for the GP."""
        mean, variance, _ = gp.predict(x)
        if math.isclose(variance, 0, abs_tol=TOLERANCE):
            return 0.
        
        u = (mean - self._max_targets) / variance
        return ((mean - self._max_targets) * self._norm.cdf(u) +
                variance * self._norm.pdf(u))
    
    def _correlation(self, x, y):
        """Computes the correlation function at two (arrays of) points for the
        GP."""
        # NOTE: can verify from the MOGP source code that this is the
        #       correct calculation (see source for gp.get_K_matrix())
        return self._kernel.kernel_f(x, y, self.gp.theta.corr_raw)
    
    def _repulsion(self, x):
        """Computes the repulsion function at an input for the GP."""
        var = self.gp.theta.cov
        correlations = gp.get_cov_matrix(np.array([x])) / var
        inputs_term = np.product(1 - correlations, axis=0)
        pseudopoints_term = np.product(1 - self._correlation(x, np.array(self.pseudopoints)), axis=1)
        return inputs_term * pseudopoints_term

    def compute(self, x):
        """Compute the pseudo-expected improvement at an input for the GP."""
        return self._expected_improvement(x) * self._repulsion(x)


def f(x) -> float:
    """The simulator being emulated"""
    if not 0 <= x[0] <= 1 and 0 <= x[1] <= 1:
        raise ValueError("ERROR: input variable(s) out of range")

    return x[0] + x[0]**2 + x[1]**2 - math.sqrt(2)


def pop_at(array, i):
    """Return the i-th element from an array and the remaining array."""
    rows = list(array)
    x = rows.pop(i)
    return x, np.array(rows)


def calculate_esloo_error(gp, i):
    """Calculate the ES-LOO error for a Gaussian Process by leaving the ith
    training input and observation out."""
    loo_input, loo_training_inputs = pop_at(gp.inputs, i)
    loo_observation, loo_training_observations = pop_at(gp.targets, i)
    loo_gp = mogp_emulator.GaussianProcess(loo_training_inputs,
                                           loo_training_observations,
                                           kernel='Matern52')
    loo_gp.fit(gp.theta)
    result = loo_gp.predict(loo_input)
    loo_mean = float(result['mean'])
    loo_var = float(result['unc'])
    sq_error = (loo_mean - loo_observation) ** 2
    numerator = (loo_var ** 2) + sq_error
    denominator = math.sqrt(2 * (loo_var ** 4) + 4 * (loo_var ** 2) * sq_error)
    return numerator / denominator


def fit_with_hyperparameter_estimation(gp, bounds=None, n_tries=15):
    """Fit a GP, including hyperparameter estimation, with optional bounds on
    the permitted values of the hyperparameters."""

    np.seterr(divide = 'raise', over = 'raise', invalid = 'raise')

    logpost_values = []
    theta_values = []

    for _ in range(n_tries):
        theta = gp.priors.sample()
        try:
            min_dict = scipy.optimize.minimize(
                gp.logposterior, theta, method = "L-BFGS-B", jac = gp.logpost_deriv,
                bounds=bounds
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
        gp.theta = None
    else:
        logpost_values = np.array(logpost_values)
        idx = np.argmin(logpost_values)

        gp.fit(theta_values[idx])

    return gp


if __name__ == "__main__":
    
    # Create initial design
    domain = Domain((0, 1), 2)  # the domain of the simulator, which experiments are drawn from
    n_initial_samples = 7  # number of samples for initial design
    initial_experiments = domain.latin_hypercube_samples(n_initial_samples)
    initial_observations = np.array([f(x) for x in initial_experiments])

    # Get bounds on parameters that must be satisfied during
    # estimation
    parameter_bounds = domain.raw_parameter_bounds
    
    # Initialise a GP (using a Matern kernel)
    gp = mogp_emulator.GaussianProcess(initial_experiments, initial_observations, kernel='Matern52')

    # Fit GP to the initial data
    gp = mogp_emulator.fit_GP_MAP(gp)

    n_adaptive_iterations = 3
    experiments = initial_experiments
    observations = initial_observations
    for iteration in range(n_adaptive_iterations):
        print("Iteration", iteration, '\n')
        
        # Observations for ES-LOO
        observations_e = np.array(
            [calculate_esloo_error(gp, i) for i in range(len(experiments))]
        )

        # Fit GP to ES-LOO errors
        gp_e = mogp_emulator.GaussianProcess(experiments, observations_e, kernel='Matern52')
        gp_e = fit_with_hyperparameter_estimation(gp_e, bounds=parameter_bounds)

        print("Parameter bounds:", str(parameter_bounds))
        print("Parameters: ", gp_e.theta.get_data())

        # Create pseudopoints
        pseudopoints = domain.find_pseudopoints(experiments)

        print("Pseudopoints:\n", pseudopoints)

        # Optimise over pseudoexpected improvement
        pei = PEICalculator(gp_e, pseudopoints)
        
        # Check PEI of training inputs is zero
        for x in experiments:
            assert math.isclose(pei.compute(x), 0., abs_tol=TOLERANCE)

        x_new = domain.argmax(pei.compute)
        y_new = f(x_new)
        experiments = np.concatenate((experiments, np.array([x_new])))
        observations = np.concatenate((observations, np.array([y_new])))

        print("PEI of new design pt:", pei.compute(x_new))
        print("EI of new design pt:", pei._expected_improvement(x_new))
        print("Repulsion of new design pt:", pei._repulsion(x_new))

        # Update the original GP with the new experiment and observation
        gp = mogp_emulator.GaussianProcess(experiments, observations, kernel='Matern52')
        gp = mogp_emulator.fit_GP_MAP(gp)

    print(f'Initial design ({len(initial_experiments)} pts):\n', initial_experiments)
    print('Initial outputs:\n', initial_observations, '\n')
    print(f'New design points:\n', gp.inputs[-n_adaptive_iterations:])
    print('New outputs:\n', gp.targets[-n_adaptive_iterations:], '\n')
