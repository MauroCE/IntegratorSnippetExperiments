"""
Run IntegratorSnippet with a single epsilon and a fixed integration time. I want to basically reproduce Chang's results
except I will only run a single epsilon, since now we have the adaptation and that will go in a different section.

The main aim of this script, is that I want to see if the schedule of tempering parameters that I get from using the
importance part of the weight leads to a smaller variance of the estimates of the log normalizing constant.
"""
import numpy as np
from scipy.special import logsumexp
from aaa_logistic_functions import sample_prior, next_annealing_param
import numpy.linalg as la
import pickle
import time


def nlp_gnlp_nll_and_gnll(_X, _y, _Z, _scales):
    """Computes negative log likelihood and its gradient manually.

    nll(x) = sum_{i=1}^{n_d} log(1 + exp(-y_i x^top z_i))

    gnll(x) = sum_{i=1}^{n_d} frac{exp(-y_i x^top z_i)}{1 + exp(-y_i x^top z_i)} y_i z_i
    """
    # Negative log prior
    nlp = 0.5*61*np.log(2*np.pi) + 0.5*np.log(400.*(25.0**60)) + 0.5*np.sum((_X / _scales)**2, axis=1)
    gnlp = _X / (scales**2)
    # Here I use D for the number of data points (n_d)
    logE = (-_y[None, :] * _X.dot(_Z.T)).T  # (N, n_d)
    laeE = np.logaddexp(0.0, logE)  # (n_d, N)
    gnll = - np.einsum('DN, D, Dp -> Np', np.exp(logE - laeE), _y, _Z)  # (N, 61)
    return nlp, gnlp, np.sum(laeE, axis=0), gnll  # (N, ) and (N, 61)


def compute_folded_ess_for_each_k(logw):
    """Given unfolded weights, compute the folded ESS for each value of k."""
    T = logw.shape[1] - 1
    folded_ess = np.full(T, np.nan)
    for k in range(1, T+1):
        logw_folded = logsumexp(logw[:, :k+1], axis=1) - np.log(k+1)
        W_folded = np.exp(logw_folded - logsumexp(logw_folded))
        folded_ess[k-1] = 1 / np.sum(W_folded**2)
    return folded_ess


def smc_hmc_int_snip(N, T, epsilon, _y, _Z, _scales, ESSrmin=0.9, seed=1234, verbose=False):
    """
    Epsilon and tau fixed. Tempering found using importance part of the weight.
    """
    start_time = time.time()
    # Setup
    rng = np.random.default_rng(seed=seed)
    verboseprint = print if verbose else lambda *a, **kwargs: None

    # Initialise positions and velocities for N particles
    x = sample_prior(N, rng)                          # Positions (N, 61)
    v = rng.normal(loc=0.0, scale=1.0, size=(N, 61))  # Velocities (N, 61)

    # Storage
    d = x.shape[1]  # Dimension of the parameter space (61)
    pms = []
    mips = []
    pds = []
    mpds = []
    longest_batches = []
    ess_running = []
    ess = []
    ess_by_group = []
    logLt = 0.0
    kappas = []
    gammas = [0.0]
    logLt_traj = [0.0]
    n = 1

    while gammas[n-1] < 1.0:
        verboseprint("Iteration: ", n, " Gamma: ", gammas[n-1])

        # --- CONSTRUCT TRAJECTORIES USING psi_{n-1} ---
        # IMPORTANT: To maintain vectorization (since kappa is not too large) I will construct all trajectories up to
        # the largest of the Ts and then in case drop some trajectory points when needed.
        # Setup storage
        trajectories = np.full((N, T+1, 2*d), np.nan)
        trajectories[:, 0] = np.hstack((x, v))
        # Store only the log densities (used for log-weights), not gradients
        nlps = np.full((N, T+1), np.nan)   # Negative log priors
        nlls = np.full((N, T+1), np.nan)   # Negative log-likelihoods
        # First half-momentum step
        nlps[:, 0], gnlps, nlls[:, 0], gnlls = nlp_gnlp_nll_and_gnll(x, _y, _Z, _scales)
        v = v - 0.5*epsilon*(gnlps + gammas[n-1]*gnlls)  # (N, 61)
        # T-1 full position and momentum steps
        for k in range(T - 1):
            # Full position step
            x = x + epsilon*v
            # Full momentum step (compute nlls and gnlls)
            nlps[:, k+1], gnlps, nlls[:, k+1], gnlls = nlp_gnlp_nll_and_gnll(x, _y, _Z, _scales)
            v = v - epsilon*(gnlps + gammas[n-1]*gnlls)  # (N, 61)
            # Store trajectory
            trajectories[:, k+1] = np.hstack((x, v))
        # Final position half-step
        x = x + epsilon*v
        # Final momentum half-step
        nlps[:, -1], gnlps, nlls[:, -1], gnlls = nlp_gnlp_nll_and_gnll(x, _y, _Z, _scales)
        v = v - 0.5*epsilon*(gnlps + gammas[n-1]*gnlls)  # (N, 61)
        # Store trajectories
        trajectories[:, -1] = np.hstack((x, v))

        # --- SELECT NEXT TEMPERING PARAMETER BASED ON IMPORTANCE WEIGHT ---
        # The importance part of the unfolded weight is simply L(x)**(gamma_n - gamma_{n-1}) evaluated at the seeds
        # Therefore we use exactly Chopin's next_tempering_parameter function and feed in the seeds' log likelihoods
        gammas.append(next_annealing_param(gammas[n-1], ESSrmin, -nlls[:, 0]))
        verboseprint("\tNew Gamma: ", gammas[n])

        # --- COMPUTE LOG-WEIGHTS AND FOLDED ESS ---
        # Compute unfolded log-weights (remember we need the POSITIVE log densities)
        log_num = (- nlps) + gammas[n]*(- nlls) - 0.5*la.norm(trajectories[:, :, d:], axis=2)**2  # (N, T+1)
        log_den = (- nlps[:, 0]) + gammas[n-1]*(- nlls[:, 0]) - 0.5*la.norm(trajectories[:, 0, d:], axis=1)**2  # (N, )
        logw = log_num - log_den[:, None]  # (N, T+1)
        W = np.exp(logw - logsumexp(logw))
        logw_folded = logsumexp(logw, axis=1) - np.log(T+1)  # (N, )
        W_folded = np.exp(logw_folded - logsumexp(logw_folded))
        ess.append(1 / np.sum(W_folded**2))
        verboseprint("\tTrajectories and weights computed. Folded ESS: ", ess[-1])

        # --- RESAMPLING ---
        # Importantly, notice that the pool of possible indices to resample from is not so trivial
        # One needs to compute it
        A = rng.choice(a=N*(T+1), size=N, replace=True, p=W.ravel())
        # Unravelling requires segmentation by particle subgroup
        n_indices, k_indices = np.unravel_index(A, (N, T+1))
        x = trajectories[n_indices, k_indices, :d]  # (N, 61)
        pms.append(np.sum(k_indices > 0) / N)
        mips.append(np.median(k_indices) / T)
        pds.append((len(np.unique(n_indices)) - 1) / (N - 1))
        mpds.append(np.sqrt(mips[-1] * pds[-1]))
        verboseprint("\tParticles resampled. MIPs: ", mips[-1], " PMs: ", pms[-1], " PDs: ", pds[-1],
                     " MPDs: ", mpds[-1])

        # --- REFRESH VELOCITIES
        v = rng.normal(loc=0.0, scale=1.0, size=v.shape)

        # --- COMPUTE LOG NORMALIZING CONSTANT ---
        # We use the fact that the folded algorithm is indeed an SMC sampler
        logLt_traj.append(logsumexp(logw_folded) - np.log(N))
        logLt += logLt_traj[-1]
        verboseprint("\tLog NC: ", logLt)

        # --- COMPUTE LONGEST BATCHES ---
        # Longest batch corresponds to the first k such that (x_k - x_0) v_k < 0
        longest_batches.append(
            np.argmax(
                np.einsum(
                    'ijk, ijk->ij',
                    trajectories[:, 1:, :d] - trajectories[:, 0:1, :d],
                    trajectories[:, 1:, d:])
                < 0,
                axis=1)
        )

        n += 1

    return {'logLt': logLt, 'pms': pms, 'mips': mips, 'ess': ess, 'longest_batches': longest_batches,
            'ess_running': ess_running, 'kappas': kappas, 'pds': pds, 'mpds': mpds, 'gammas': gammas,
            'logLt_traj': logLt_traj, 'runtime': time.time() - start_time}


if __name__ == "__main__":

    # Grab data
    data = np.load("/Users/za19162/Documents/Code/integrator_snippets_exp/sonar.npy")
    y = -data[:, 0]  # Shape (208,)
    Z = data[:, 1:]  # Shape (208, 61)
    scales = np.array([5] * 61)
    scales[0] = 20

    # Settings
    n_runs = 100
    budget = 10**4
    overall_seed = 1234
    seeds = np.random.default_rng(overall_seed).integers(low=1, high=10000, size=n_runs)
    results = []
    N_list = [10, 20, 50, 100, 200]

    for _N in N_list:
        print("N: ", _N)
        for i in range(n_runs):
            if _N == 100 and i == 94:
                _T = budget // _N
                res = {'N': _N, 'T': _T}
                out = smc_hmc_int_snip(N=_N, T=_T, epsilon=0.1, ESSrmin=0.8, _y=y, _Z=Z, _scales=scales, verbose=False,
                                       seed=int(seeds[i]))
                res.update({'type': 'tempering', 'logLt': out['logLt'], 'waste': False, 'out': out})
                print("\t\tRun ", i, " LogLt: ", out['logLt'], " Runtime: ", out['runtime'])
                results.append(res)

    # Save data
    # with open("results/aaj_is_hmc_1e_atis08_ft/int_snip_eps01_time.pkl", "wb") as file:
    #     pickle.dump(results, file)
