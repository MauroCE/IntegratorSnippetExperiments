"""
WASTE-FREE SMC USING RWM.
"""
import pickle
import numpy as np
import scipy as sp
from aaa_logistic_functions import sample_prior, next_annealing_param, log_likelihood_vect
from aaa_logistic_functions import log_prior_vect, path_sampling_estimate
from particles.particles.resampling import systematic, wmean_and_var


def smc_wf_rwm(n_particles, len_chain, maxiter=1000, ESSrmin=0.5, seed=1234, verbose=False):
    """Performs with K MH steps. The Random-Walk kernel adapts the covariance to a fraction of the empirical
    covariance matrix of the particles.
    N: Number of particles.
    K: Number of MH steps per particle.
    """
    # Setup
    rng = np.random.default_rng(seed=seed)
    verboseprint = print if verbose else lambda *a, **k: None
    # Parameters, unsure if they are correct or not
    M = n_particles
    P = len_chain
    N0 = M * P

    # Initialise particles. For Waste-Free we initialize N0 particles
    x = sample_prior(N0, rng)  # (N0, 61)
    W = np.full(N0, 1/N0)       # (N0, )
    ll_curr = log_likelihood_vect(x)  # (N0, )
    lprior_curr = log_prior_vect(x)  # (N0, )

    # Initial settings
    d = x.shape[1]  # Dimension of the parameter space (61)
    gammas = [0.0]
    path_sampling = [0.0]
    mean_aps = []
    ess = []
    means = []
    variances = []
    logLt = 0.0
    n = 0

    while not ((n >= maxiter) or (gammas[-1] >= 1.0)):
        verboseprint("Iteration: ", n, " Gamma: ", gammas[-1])

        # RESAMPLING
        if n > 0:
            # Estimate covariance matrix
            cov = np.cov(x.T, aweights=W, ddof=0)  # empirical covariance matrix (61,61)
            L = 2.38*sp.linalg.cholesky(cov, lower=True)/np.sqrt(d)
            verboseprint("\tCovariance matrix estimated.")  # TODO: Check that cov is estimated with all N0 particles

            A = systematic(W=W, M=M, rng=rng)  # resample M out of N0
            x = x[A]  # (M, 61)
            ll_curr = ll_curr[A]
            lprior_curr = lprior_curr[A]
            verboseprint("\tParticles resampled. Unique: ", np.unique(x, axis=1).shape[0])

            # MUTATION STEP
            mean_ap = []
            states = np.full((P, M, 61), np.nan)
            log_priors = np.full((P, M), np.nan)
            log_likelihoods = np.full((P, M), np.nan)
            states[0] = x
            log_priors[0] = lprior_curr
            log_likelihoods[0] = ll_curr
            for k in range(P-1):  # repeat for P-1 times, since the initial state is included in the chain
                # Generate a proposal
                x_prop = x + rng.normal(loc=0.0, scale=1.0, size=x.shape) @ L.T
                # Compute acceptance probability
                ll_prop, lprior_prop = log_likelihood_vect(x_prop), log_prior_vect(x_prop)  # log-likelihoods and priors
                log_ar = (lprior_prop + gammas[-1]*ll_prop) - (lprior_curr + gammas[-1]*ll_curr)
                ap = np.exp(np.clip(log_ar, a_min=None, a_max=0.0))  # acceptance probability min(1, ar)
                mean_ap.append(np.mean(ap))
                # MH Accept-Reject
                accept = rng.uniform(low=0.0, high=1.0, size=M) < ap
                x[accept] = x_prop[accept]
                ll_curr[accept] = ll_prop[accept]  # store log-likelihood values for annealing
                lprior_curr[accept] = lprior_prop[accept]  # store log-prior values for efficiency
                # Store new states
                states[k+1] = x
                log_priors[k+1] = lprior_curr
                log_likelihoods[k+1] = ll_curr
            mean_aps.append(mean_ap)
            # Gather together the particles. Notice that we also need to remember to store the log prior, and
            # log likelihood carefully, so that we can use them to find the next annealing exponent
            x = states.reshape(N0, 61)  # (N0, 61)
            lprior_curr = log_priors.flatten()  # (N0, )
            ll_curr = log_likelihoods.flatten()  # (N0, )
            verboseprint("\tParticles mutated. Mean AP: ", np.mean(mean_ap))

        # RE-WEIGHTING
        # Compute new tempering parameter
        gammas.append(next_annealing_param(gammas[-1], ESSrmin, ll_curr))
        delta_gammas = gammas[-1] - gammas[-2]
        verboseprint("\tNew Gamma: ", gammas[-1])
        # Update path sampling estimate
        path_sampling.extend(path_sampling_estimate(delta=delta_gammas, ps=path_sampling[-1], ll=ll_curr))
        # Compute log-incremental-weights (log-G)
        logw = delta_gammas*ll_curr
        logw[np.isnan(logw)] = -np.inf
        m = logw.max()
        w = np.exp(logw - m)
        s = w.sum()
        log_mean = m + np.log(s / N0)  # we have N0 particles
        W = w / s
        ess.append(1.0 / np.sum(W**2))
        verboseprint("\tParticles re-weighted.")

        # COMPUTE SUMMARIES
        mean, var = wmean_and_var(W=W, x=x).values()
        means.append(mean)
        variances.append(var)
        logLt += log_mean
        verboseprint("\tSummaries computed. LogLt: ", logLt)

        n += 1
    return {'gammas': gammas, 'ps': path_sampling, 'mean_aps': mean_aps, 'ess': ess,
            'means': means, 'variances': variances, 'x': x, 'logLt': logLt}


if __name__ == "__main__":
    n_runs = 100
    overall_seed = 1234
    seeds = np.random.default_rng(overall_seed).integers(low=1, high=10000, size=n_runs)
    budget = 10**4
    Ns = [10, 20, 50, 100, 200]  # Number of resampled particles
    results = []
    for N in Ns:
        print("N: ", N)
        for i in range(n_runs):
            print("\tRun: ", i)
            P = budget // N  # length of chain
            res = {'N': N, 'P': P}
            out = smc_wf_rwm(n_particles=N, len_chain=P, verbose=False, seed=int(seeds[i]))
            res.update({'type': 'tempering', 'logLt': out['logLt'], 'waste': False, 'out': out})
            results.append(res)
    with open("results/aah_is_hmc_3e_atfull08_ft_rep100/wf_smc.pkl", "wb") as file:
        pickle.dump(results, file)
