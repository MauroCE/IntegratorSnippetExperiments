"""
NO ADAPTATION, 3 EPSILONS, STORE CRITERIA ALONG SNIPPET. INTERMEDIARY DISTRIBUTIONS FOUND USING THE FULL WEIGHTS.
"""
import numpy as np
from scipy.special import logsumexp
from aaa_logistic_functions import sample_prior, next_annealing_param_unfolded
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


def criteria_numerator(xnk, logw):
    """Expects `xnk` of shape (N, T+1, d). Expects `logw` to be the unnormalized
    unfolded weights of shape `(N, T+1)`."""
    # Compute squared norm
    mu_k_given_z = np.exp(logw - logsumexp(logw, axis=1, keepdims=True))  # (N, T+1)
    diff = xnk - np.sum(xnk * mu_k_given_z[:, :, None], axis=1, keepdims=True)  # (N, 1, d)
    sq_norm = np.linalg.norm(diff, axis=2)**2  # (N, T+1)
    # Compute weighted squared norm and sum up
    return np.sum(sq_norm * np.exp(logw - logsumexp(logw)))


def criteria_numerator_upto_t(xnk, logw):
    """Only the outer expectation is truncated."""
    mu_k_given_z = np.exp(logw - logsumexp(logw, axis=1, keepdims=True))  # (N, T+1)
    diff = xnk - np.sum(xnk * mu_k_given_z[:, :, None], axis=1, keepdims=True)  # (N, 1, d)
    sq_norm = np.linalg.norm(diff, axis=2)**2  # (N, T+1)
    W = np.exp(logw - logsumexp(logw))
    # compute cumulative sum of weighted squared norm and sum up
    return np.cumsum(np.sum(sq_norm * W, axis=0))


def criteria_numerator_upto_t_mean_upto_t(xnk, logw):
    """Here we also compute the mean up to t."""
    T = xnk.shape[1] - 1
    w_folded = np.exp(logw - logsumexp(logw, axis=1, keepdims=True))  # (N, T+1)
    W_unfolded = np.exp(logw - logsumexp(logw))  # (N, T+1)
    criteria = np.zeros(T+1)
    for t in range(T+1):
        diff = xnk[:, :t] - np.sum(xnk[:, :t] * w_folded[:, :t, None], axis=1, keepdims=True)  # (N, t+1, d)
        sq_norm = np.linalg.norm(diff, axis=2) # (N, t+1)
        criteria[t] = np.sum(sq_norm * W_unfolded[:, :t])
    return criteria  # (T+1)


def criteria_denominator1(xnk, logw):
    """Computes the denominator. Here we assume that we only use the uniform distribution for
    the expectation inside the squared norm, not as the weighting."""
    diff = xnk - np.mean(xnk, axis=1, keepdims=True)  # (N, 1, d)
    sq_norm = np.linalg.norm(diff, axis=2)**2  # (N, T+1)
    # Compute weighted squared norm and sum up
    return np.sum(sq_norm * np.exp(logw - logsumexp(logw)))


def true_criterion(xnk, logw, iotas):
    """True criterion, found using maths derivation on 22/08/2024.
    Expects xnk to have shape (N, T+1, d), logw to have shape (N, T+1) and iotas to have shape (N, )."""
    W_unfolded = np.exp(logw - logsumexp(logw))  # (N, T+1) complete set of unfolded weights
    mu_k_eps_given_z = np.exp(logw - logsumexp(logw, axis=1, keepdims=True))  # (N, T+1)
    cond_exp = np.sum(xnk * mu_k_eps_given_z[:, :, None], axis=1, keepdims=True)  # (N, 1, d)
    criteria = []
    for group in range(3):
        flag = (iotas == group)
        sq_norm = np.linalg.norm(xnk[flag] - cond_exp[flag], axis=2)**2  # (NG, T+1)
        criteria.append(np.sum(sq_norm * W_unfolded[flag]))  # scalar for each grop
    return criteria


def smc_hmc_int_snip(N, T, _epsilons, _y, _Z, _scales, ESSrmin=0.9, seed=1234, verbose=False):
    start_time = time.time()
    # Setup
    rng = np.random.default_rng(seed=seed)
    verboseprint = print if verbose else lambda *a, **kwargs: None
    # Hyperparameter settings
    epsilons = _epsilons
    taus = T * epsilons

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
    epsilons_history = [epsilons]
    kappas = []
    gammas = [0.0]
    logLt_traj = [0.0]
    tau_history = [taus]
    n_unique = []
    adapt_choice = []
    # store criteria
    criteria1 = [np.nan]
    criteria2 = [np.nan]
    criteria3 = [np.nan]
    # k resampled
    k_resampled = []
    n_resampled = []
    iotas_list = []


    n = 1

    while gammas[n-1] < 1.0:
        verboseprint("Iteration: ", n, " Gamma: ", gammas[n-1], " T: ", T, " epsilons: ", epsilons, " taus: ", taus,
                     " c1: ", criteria1[-1], " c2: ", criteria2[-1], " c3: ", criteria3[-1])

        # --- CHOOSE WHICH PARTICLES WILL BE ASSIGNED TO WHICH EPSILON/T COMBINATION.
        iotas = np.array(rng.choice(a=len(epsilons), size=N))  # one flag for each particle
        epsilons_vector = epsilons.reshape(-1, 1)[iotas]  # use for integration
        iotas_list.append(iotas)

        # --- CONSTRUCT TRAJECTORIES USING psi_{n-1} ---
        # Setup storage
        trajectories = np.full((N, T+1, 2*d), np.nan)
        trajectories[:, 0] = np.hstack((x, v))
        # Store only the log densities (used for log-weights), not gradients
        nlps = np.full((N, T+1), np.nan)   # Negative log priors
        nlls = np.full((N, T+1), np.nan)   # Negative log-likelihoods
        # First half-momentum step
        nlps[:, 0], gnlps, nlls[:, 0], gnlls = nlp_gnlp_nll_and_gnll(x, _y, _Z, _scales)
        v = v - 0.5*epsilons_vector*(gnlps + gammas[n-1]*gnlls)  # (N, 61)
        # T-1 full position and momentum steps
        for k in range(T - 1):
            # Full position step
            x = x + epsilons_vector*v
            # Full momentum step (compute nlls and gnlls)
            nlps[:, k+1], gnlps, nlls[:, k+1], gnlls = nlp_gnlp_nll_and_gnll(x, _y, _Z, _scales)
            v = v - epsilons_vector*(gnlps + gammas[n-1]*gnlls)  # (N, 61)
            # Store trajectory
            trajectories[:, k+1] = np.hstack((x, v))
        # Final position half-step
        x = x + epsilons_vector*v
        # Final momentum half-step
        nlps[:, -1], gnlps, nlls[:, -1], gnlls = nlp_gnlp_nll_and_gnll(x, _y, _Z, _scales)
        v = v - 0.5*epsilons_vector*(gnlps + gammas[n-1]*gnlls)  # (N, 61)
        # Store trajectories
        trajectories[:, -1] = np.hstack((x, v))

        # --- SELECT NEXT TEMPERING PARAMETER BASED ON IMPORTANCE WEIGHT ---
        # Compute next annealing parameter  ESSrmin, lvn, lvd, nlps, nlls, T, N
        lvn = - 0.5*np.linalg.norm(trajectories[:, :, d:], axis=2)**2
        lvd = - 0.5*np.linalg.norm(trajectories[:, 0, d:], axis=1)**2
        gammas.append(
            next_annealing_param_unfolded(
                gamma=gammas[n-1],
                ESSrmin=ESSrmin,
                lvn=lvn,
                lvd=lvd,
                nlps=nlps,
                nlls=nlls,
                T=T,
                N=N)
        )
        verboseprint("\tNew Gamma: ", gammas[n])

        # --- COMPUTE LOG-WEIGHTS AND FOLDED ESS ---
        # Compute unfolded log-weights (remember we need the POSITIVE log densities)
        log_num = (- nlps) + gammas[n]*(- nlls) - 0.5*la.norm(trajectories[:, :, d:], axis=2)**2  # (N, T+1)
        log_den = (- nlps[:, 0]) + gammas[n-1]*(- nlls[:, 0]) - 0.5*la.norm(trajectories[:, 0, d:], axis=1)**2  # (N, )
        logw = log_num - log_den[:, None]  # (N, T+1)
        # IMPORTANT: In this version, each group shares the same T, only epsilon/tau changes. This means that all
        # points generated up to now are valid. Computing quantities is easy
        # Normalized Unfolded weights
        W = np.exp(logw - logsumexp(logw))  # normalized unfolded weights
        # Normalized Folded Weights and Folded ESS
        logw_folded = logsumexp(logw, axis=1) - np.log(T+1)
        W_folded = np.exp(logw_folded - logsumexp(logw_folded))
        ESS_folded = 1 / np.sum(W_folded**2)
        # Folded ESS by group
        ess_folded_by_group = np.zeros(3)
        for group in range(3):
            logw_folded_group = logw_folded[iotas == group]
            W_folded_group = np.exp(logw_folded_group - logsumexp(logw_folded_group))
            ess_folded_by_group[group] = 1 / np.sum(W_folded_group ** 2)
        ess_by_group.append(ess_folded_by_group)
        verboseprint("\tTrajectories and weights computed. Folded ESS: ", ESS_folded)
        verboseprint("\tESSs by group: ", ess_folded_by_group)

        # --- RESAMPLING ---
        # Importantly, notice that the pool of possible indices to resample from is not so trivial
        # One needs to compute it
        A = rng.choice(a=N*(T+1), size=N, replace=True, p=W.ravel())
        # Unravelling requires segmentation by particle subgroup
        n_indices, k_indices = np.unravel_index(A, (N, T+1))
        x = trajectories[n_indices, k_indices, :d]  # (N, 61)
        pms.append(
            [np.sum(k_indices[iotas[n_indices] == group] > 0) / np.sum(iotas[n_indices] == group) for group in range(3)]
        )
        mips.append([np.median(k_indices[iotas[n_indices] == group]) / T for group in range(3)])
        pds.append(
            [(len(np.unique(n_indices[iotas[n_indices] == group])) - 1) / (np.sum(iotas == group) - 1)
             for group in range(3)]
        )
        mpds.append(np.sqrt(np.array(mips[-1]) * np.array(pds[-1])))
        n_unique.append(len(np.unique(A)))
        k_resampled.append(k_indices)
        n_resampled.append(n_indices)
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

        # COMPUTE CRITERIA 1
        # c1 = []
        # c2 = []
        # c3 = []
        # for group in range(3):
        #     c1.append(criteria_numerator(trajectories[iotas == group, :, :d], logw[iotas == group]))
        #     c2.append(criteria_denominator1(trajectories[iotas == group, :, :d], logw[iotas == group]))
        #     c3.append(c1[-1] / c2[-1])
        # criteria1.append(c1)
        # criteria2.append(c2)
        # criteria3.append(c3)

        # PRETEND
        # criteria = np.zeros((3, T))
        # for group in range(3):
        #     xnk_group = trajectories[iotas == group]
        #     logw_group = logw[iotas == group]
        #     for t in range(1, T+1):
        #         criteria[group, t-1] = criteria_numerator(xnk_group[:, :t, :d], logw_group[:, :t])
        # criteria1.append(criteria)

        # truncate first sum
        criteria = np.zeros((3, T+1))
        for group in range(3):
            criteria[group, :] = criteria_numerator_upto_t_mean_upto_t(
                xnk=trajectories[iotas == group],
                logw=logw[iotas == group]
            )
        criteria1.append(criteria)

        # STORE
        epsilons_history.append(epsilons)
        tau_history.append(taus)
        ess.append(ESS_folded)

        n += 1

    return {'logLt': logLt, 'pms': pms, 'mips': mips, 'ess': ess, 'longest_batches': longest_batches,
            'ess_running': ess_running, 'T': T, 'epsilons_history': epsilons_history, 'kappas': kappas,
            'ess_by_group': ess_by_group, 'pds': pds, 'mpds': mpds, 'gammas': gammas, 'logLt_traj': logLt_traj,
            'tau_history': tau_history, 'n_unique': n_unique, 'runtime': time.time() - start_time,
            'adapt_choice': adapt_choice, 'criteria1': criteria1, 'criteria2': criteria2, 'criteria3': criteria3,
            'k_resampled': k_resampled, 'iotas_list': iotas_list, 'n_resampled': n_resampled}


if __name__ == "__main__":

    # Grab data
    data = np.load("/Users/za19162/Documents/Code/integrator_snippets_exp/sonar.npy")
    y = -data[:, 0]  # Shape (208,)
    Z = data[:, 1:]  # Shape (208, 61)
    scales = np.array([5] * 61)
    scales[0] = 20

    # Settings
    n_runs = 5
    overall_seed = np.random.randint(low=10, high=29804393)  # 1234
    seeds = np.random.default_rng(overall_seed).integers(low=1, high=10000, size=n_runs)
    _epsilons = np.array([2.0, 0.1, 0.01])
    _N = 1000
    _T = 50

    results = []
    for i in range(n_runs):
        res = {'N': _N, 'T': _T, 'epsilons': _epsilons}
        out = smc_hmc_int_snip(N=_N, T=_T, _epsilons=_epsilons, ESSrmin=0.8, _y=y, _Z=Z, _scales=scales,
                               verbose=False, seed=int(seeds[i]))
        res.update({'type': 'tempering', 'logLt': out['logLt'], 'waste': False, 'out': out})
        print("\t\tRun ", i, " LogLt: ", out['logLt'], " Final ESS: ", out['ess'][-1], 'final eps: ',
              out['epsilons_history'][-1])
        results.append(res)

    with open(f"results/aaq/results_2_01_001_uptot_meanuptot_T{_T}.pkl", "wb") as file:
        pickle.dump(results, file)
