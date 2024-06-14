"""Standard SMC-HMC sampler. Performs K steps of HMC for each of the N particles."""
import numpy as np
import scipy as sp
import pickle
from logistic_functions import sample_prior, next_annealing_param, log_likelihood_vect
from logistic_functions import log_prior_vect, path_sampling_estimate
from particles.particles.resampling import systematic, wmean_and_var


def smc_rwm(n_particles, n_mcmc, maxiter=1000, ESSrmin=0.5, seed=1234, verbose=False):
    """Performs SMC-RW with K MH steps. The Random-Walk kernel adapts the covariance to a fraction of the empirical
    covariance matrix of the particles.
    N: Number of particles.
    K: Number of MH steps per particle.
    """
    # Setup
    rng = np.random.default_rng(seed=seed)
    print("RNG: ", rng)
    verboseprint = print if verbose else lambda *a, **kwargs: None

    # Initialise particles
    x = sample_prior(n_particles, rng)  # (N, 61)
    W = np.full(n_particles, 1/n_particles)       # (N, )
    ll_curr = log_likelihood_vect(x)  # (N, )
    lprior_curr = log_prior_vect(x)  # (N, )

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
            verboseprint("\tCovariance matrix estimated.")

            A = systematic(W=W, M=n_particles, rng=rng)
            x = x[A]
            ll_curr = ll_curr[A]
            lprior_curr = lprior_curr[A]
            verboseprint("\tParticles resampled. Unique: ", np.unique(x, axis=1).shape[0])

            # MUTATION STEP
            mean_ap = []
            for k in range(n_mcmc):
                # Generate a proposal
                x_prop = x + rng.normal(loc=0.0, scale=1.0, size=x.shape) @ L.T
                # Compute acceptance probability
                ll_prop, lprior_prop = log_likelihood_vect(x_prop), log_prior_vect(x_prop)  # log-likelihoods and priors
                log_ar = (lprior_prop + gammas[-1]*ll_prop) - (lprior_curr + gammas[-1]*ll_curr)
                ap = np.exp(np.clip(log_ar, a_min=None, a_max=0.0))  # acceptance probability min(1, ar)
                mean_ap.append(np.mean(ap))
                # MH Accept-Reject
                accept = rng.uniform(low=0.0, high=1.0, size=n_particles) < ap
                x[accept] = x_prop[accept]
                ll_curr[accept] = ll_prop[accept]  # store log-likelihood values for annealing
                lprior_curr[accept] = lprior_prop[accept]  # store log-prior values for efficiency
            mean_aps.append(mean_ap)
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
        log_mean = m + np.log(s / n_particles)
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
    N0 = 2 * 10 ** 5
    Ks = [5, 20, 100, 500, 1000]   # K number of MCMC steps
    Ms = [50, 100, 200, 400, 800]  # M number of resampled particles in Waste-Free smc
    results = []
    for M, K in zip(Ms, Ks):
        for i in range(n_runs):
            N = N0 // K
            res = {'N': N, 'K': K}
            out = smc_rwm(n_particles=N, n_mcmc=K, verbose=True, seed=int(seeds[i]))
            res.update({'type': 'tempering', 'logLt': out['logLt'], 'waste': False, 'out': out})
            results.append(res)
    with open("results_full_mauro_smcrwm.pkl", "wb") as file:
        pickle.dump(results, file)
