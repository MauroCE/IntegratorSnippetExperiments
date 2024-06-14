"""Contains functions to perform logistic regression."""
import numpy as np
import scipy as sp
import jax.numpy as jnp
from scipy.special import logsumexp


data = np.load("/Users/za19162/Documents/Code/integrator_snippets_exp/sonar.npy")
y = -data[:, 0]  # Shape (208,)
Z = data[:, 1:]  # Shape (208, 61)
scales = np.array([5]*61)
scales[0] = 20
scales_jpn = jnp.asarray(scales)
y_jnp = jnp.asarray(y)
Z_jnp = jnp.asarray(Z)


def exp_and_normalise(lw):
    """FROM CHOPIN. Exponentiate, then normalise (so that sum equals one).

    Arguments
    ---------
    lw : ndarray
        log weights.

    Returns
    -------
    W : ndarray of the same shape as lw
        W = exp(lw) / sum(exp(lw))

    Note
    ----
    uses the log_sum_exp trick to avoid overflow (i.e. subtract the max
    before exponentiating)

    See also
    --------
    log_sum_exp
    log_mean_exp

    """
    w = np.exp(lw - lw.max())
    return w / w.sum()


def essl(lw):
    """FROM CHOPIN!! ESS (Effective sample size) computed from log-weights.

    Parameters
    ----------
    lw : (N, ) ndarray
        log-weights

    Returns
    -------
    float
        the ESS of weights w = exp(lw), i.e. the quantity
        sum(w**2) / (sum(w))**2

    Note
    ----
    The ESS is a popular criterion to determine how *uneven* are the weights.
    Its value is in the range [1, N], it equals N when weights are constant,
    and 1 if all weights but one are zero.

    """
    w = np.exp(lw - lw.max())
    return (w.sum()) ** 2 / np.sum(w ** 2)


def log_prior(theta):
    """Log-prior is a product of 61 zero-mean normals with standard deviations 20, 5, ...., 5.
    This works on a single vector theta of shape (p,)."""
    return -0.5*np.sum((theta / scales)**2)


def log_likelihood(theta):
    """Log-likelihood for a parameter value. Data is fixed.
    This works on a single vector theta of shape (p, )."""
    ll = -np.sum(np.log(1 + np.exp(-y*(Z.dot(theta)))))
    return np.nan_to_num(ll, copy=False, nan=-np.inf)


def log_temp_post(theta, gamma):
    """Log-tempered posterior."""
    return log_prior(theta) + gamma*log_likelihood(theta)


def log_prior_vect(theta):
    """Log-prior is a product of 61 zero-mean normals with standard deviations 20, 5, ...., 5.
    Expects theta to be a matrix of shape (N, 61)."""
    return -0.5*61*np.log(2*np.pi) - 0.5*np.log(400.*(25**60)) - 0.5*np.sum((theta / scales)**2, axis=1)


def log_likelihood_vect(theta):
    """Log-likelihood for a parameter value. Data is fixed.
    Works for a matrix of shape (N, 61)."""
    ll = - np.sum(np.logaddexp(0.0, -y*theta.dot(Z.T)), axis=1)
    return np.nan_to_num(ll, copy=False, nan=-np.inf)


def log_temp_post_vect(theta, gamma):
    """Log-tempered posterior for a matrix input."""
    return log_prior_vect(theta) + gamma*log_likelihood_vect(theta)


def sample_prior(n, rng):
    """Samples n particles from the prior."""
    return scales * rng.normal(loc=0.0, scale=1.0, size=(n, 61))


def next_annealing_param(gamma, ESSrmin, llk):
    """Find next annealing exponent by solving ESS(exp(lw)) = alpha * N."""
    N = llk.shape[0]

    def f(e):
        ess = essl(e * llk) if e > 0.0 else N  # avoid 0 x inf issue when e==0
        return ess - ESSrmin * N
    if f(1. - gamma) < 0.:
        return gamma + sp.optimize.brentq(f, 0.0, 1.0 - gamma)
    else:
        return 1.0


def next_annealing_param_unfolded(gamma, ESSrmin, lvn, lvd, nlps, nlls, T, N):
    """This works specifically for the unfolded algorithm. The idea is that inside f we compute the folded ESS
    but using the quantities from the unfolded algorithm somehow."""
    def f(e):
        # Compute ESS if e is larger than 0.0 (like Chopin)
        if e > 0.0:
            # Compute ESS
            log_num = (- nlps) + e * (- nlls) + lvn  # (N, T+1)
            log_den = (- nlps[:, 0]) + gamma * (- nlls[:, 0]) + lvd  # (N, )
            logw = log_num - log_den[:, None]  # (N, T+1)
            logw_folded = logsumexp(logw, axis=1) - np.log(T + 1)  # (N, )
            W_folded = np.exp(logw_folded - logsumexp(logw_folded))
            ess = 1 / np.sum(W_folded**2)
        else:
            ess = N
        return ess - ESSrmin * N
    if np.sign(f(gamma)) != np.sign(f(1.0)):
        return sp.optimize.brentq(f, gamma, 1.0)  # gamma + sp.optimize.brentq(f, 0.0, 1.0 - gamma)
    else:
        return 1.0


def path_sampling_estimate(delta, ps, ll):
    grid_size = 10
    bin_width = delta / (grid_size - 1)
    new_ps_estimate = ps
    ps_list = [ps]
    ls = np.linspace(0.0, delta, grid_size)
    for i, e in enumerate(ls):
        mult = 0.5 if i == 0 or i == grid_size - 1 else 1.0
        new_ps_estimate += mult * bin_width * np.average(ll, weights=exp_and_normalise(e*ll))
        ps_list.append(new_ps_estimate)
    return ps_list


def grad_neg_post(theta, gamma):
    """Gradient of negative log posterior. Takes theta (N, 61)."""
    # Grad negative log prior (GNLP)
    gnlp = theta / (scales**2)
    # Grad negative log likelihood (GNLL)
    arg = -y*theta.dot(Z.T)  # (N, 208)
    gnll = - np.sum((np.exp(arg) / np.logaddexp(0.0, arg)) @ (y[:, None]*Z), axis=0)  # (N, 208) x (208, 61) = (N, 61)
    return gamma*np.nan_to_num(gnll, copy=False, nan=-np.inf) + gnlp


def grad_neg_log_post(theta, gamma):
    """Gradient of negative log posterior. Theta (N, 61). Z (208, 61), y (208,)."""
    gnl_prior = theta / (scales**2)
    return gnl_prior - gamma*np.matmul(
        np.exp(-np.logaddexp(0.0, y[None, :] * theta.dot(Z.T))) * y[None, :], Z)  # (N, 61)


def neg_log_likelihood_jax(theta):
    """Computes neg log-likelihood using Jax. This will be fed to jax.value_and_grad to compute both the log-likelihood
    and its gradient. Computes it for one particle, hence theta is a vector of shape (61,)."""
    nll = jnp.sum(jnp.logaddexp(0.0, -y_jnp*Z_jnp.dot(theta)))  # scalar
    return jnp.nan_to_num(nll, copy=False, nan=-jnp.inf)


def hmc_integrator(x, v, L, epsilon, grad_U_vect, gamma):
    """L steps of Leapfrog integrator with step size epsilon."""
    # Store the entire HMC trajectory
    N, d = x.shape
    trajectory = np.full((N, L+1, 2*d), np.nan)
    trajectory[:, 0] = np.hstack((x, v))

    # First half-momentum step
    v = v - 0.5*epsilon*grad_U_vect(x, gamma)

    # L - 1 full position and momentum steps
    for ell in range(L - 1):
        x = x + epsilon*v                      # Full position step
        v = v - epsilon*grad_U_vect(x, gamma)  # Full momentum step
        trajectory[:, ell+1] = np.hstack((x, v))

    # Final position half-step
    x = x + 0.5*epsilon*v

    # Final momentum half-step
    v = v - 0.5*epsilon*grad_U_vect(x, gamma)

    trajectory[:, -1] = np.hstack((x, -v))  # Flip the sign of v
    return trajectory
