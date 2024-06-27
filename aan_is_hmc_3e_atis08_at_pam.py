"""
Simpler adaptation scheme that preserves parallelism. I have copied AAM and then simplified from there, meaning
that now the three groups keep the same number of integration steps.
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


def smc_hmc_int_snip(N, T, epsilon_init, _y, _Z, _scales, ESSrmin=0.9, seed=1234, Tmin=2, verbose=False,
                     mult=2.0, adaptive=True, pm_min=1e-2):
    start_time = time.time()
    # Setup
    rng = np.random.default_rng(seed=seed)
    verboseprint = print if verbose else lambda *a, **kwargs: None
    # Hyperparameter settings
    taus = np.array([T*epsilon_init]*3)
    epsilons = taus/T

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

    n = 1

    while gammas[n-1] < 1.0:
        verboseprint("Iteration: ", n, " Gamma: ", gammas[n-1], " T: ", T, " epsilons: ", epsilons, " taus: ", taus)

        # --- CHOOSE WHICH PARTICLES WILL BE ASSIGNED TO WHICH EPSILON/T COMBINATION.
        iotas = np.array(rng.choice(a=len(epsilons), size=N))  # one flag for each particle
        epsilons_vector = epsilons.reshape(-1, 1)[iotas]  # use for integration

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
        # The importance part of the unfolded weight is simply L(x)**(gamma_n - gamma_{n-1}) evaluated at the seeds
        # Therefore we use exactly Chopin's next_tempering_parameter function and feed in the seeds' log likelihoods
        gammas.append(next_annealing_param(gammas[n-1], ESSrmin, -nlls[:, 0]))
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

        # --- ADAPT EPSILON ---
        if adaptive:
            if np.all(np.array(pms[-1]) <= pm_min):
                # Step size is way too large, half it
                epsilons = epsilons / mult
                taus = epsilons * T
                verboseprint("\t\t\tPMS<PM_MIN: T: ", T, " Epsilons: ", epsilons, " taus: ", taus)
            else:
                # Compute lmax for the whole population
                lmax_groups = np.array([longest_batches[-1][iotas == group].max() for group in range(3)])
                lmax = lmax_groups.max()  # Overall maximum U-turn index
                verboseprint("\tLongest batch: ", lmax, " LB-Groups: ", lmax_groups)
                # Assumption: step size too small when no U-turn is detected and mip/pm are high and ESS is borderline
                if Tmin < lmax < T:  # U-turn detected: integration time is too large (we are doubling back)
                    verboseprint("\tTmin < lmax < Tmax")
                    # When 3 groups are equal, then we find the new integration time using lmax
                    if np.all(taus == taus[0]) and np.all(epsilons == epsilons[0]):
                        # tau = lmax * epsilons[0]  # integration time to the furthest U-turn (smaller than previous tau)
                        # verboseprint("\t\tEqual groups. Tau: ", tau)
                        if (ESS_folded < ESSrmin*N) and np.all(ess_folded_by_group < ESSrmin*N/3):
                            # Step size too large: create 3 groups with increasing (but smaller) step sizes
                            taus = epsilons[0] * np.floor(
                                np.quantile(longest_batches[-1], q=[0.5, 0.75, 1.0])
                            ).astype(int)
                            verboseprint("\t\t\tESS<alphaESS. taus: ", taus)
                        epsilons = taus / T
                        verboseprint("\t\tEpsilons: ", epsilons)
                    else:
                        # 3 groups have different epsilon/tau but the same T. Remember that we always keep T fixed,
                        # meaning that we will use the U-turn statistics per group to adjust any step size/tau
                        if (ess_folded_by_group[-1] < ESSrmin*N/3) and np.all(ess_folded_by_group[:-1] >= ESSrmin*N/3):
                            # right group (large step size, small tau) has low ESS.
                            # Keep left step size unchanged. Move right one between right one and middle one,
                            # and then move middle one between left and right.
                            epsilons = np.array([
                                epsilons[0],
                                0.5*epsilons[0] + 0.25*epsilons[1] + 0.25*epsilons[2],
                                0.5*(epsilons[1] + epsilons[2])
                            ])
                            verboseprint("\t\t\tESS+<alphaESS/3: Epsilons: ", epsilons)
                        elif np.all(ess_folded_by_group[-2:] < ESSrmin*N/3) and ess_folded_by_group[0] >= ESSrmin*N/3:
                            # both middle and right group have low ESS.
                            epsilons = np.array([
                                epsilons[0],
                                0.75*epsilons[0] + 0.25*epsilons[1],
                                0.5*epsilons[0] + 0.5*epsilons[1]
                            ])
                            verboseprint("\t\t\tESS,ESS+<alphaESS/3: Epsilons: ", epsilons)
                        # Adjust integration time
                        taus = epsilons * T
                        verboseprint("\t\t\tTaus: ", taus)
                else:
                    # No longest batch detected + large MIP --> step size too small, need to increase it
                    if np.all(np.array(mips[-1]) > 0.4):
                        epsilons = epsilons * mult  # double the step sizes, keep Ts the same
                        taus = epsilons * T  # there should be only one value TODO: CHECK
                        verboseprint("\t\t\tMIPS>0.4: T: ", T, " Epsilons: ", epsilons, " taus: ", taus)
        # STORE
        epsilons_history.append(epsilons)
        tau_history.append(taus)
        ess.append(ESS_folded)

        n += 1

    return {'logLt': logLt, 'pms': pms, 'mips': mips, 'ess': ess, 'longest_batches': longest_batches,
            'ess_running': ess_running, 'T': T, 'epsilons_history': epsilons_history, 'kappas': kappas,
            'ess_by_group': ess_by_group, 'pds': pds, 'mpds': mpds, 'gammas': gammas, 'logLt_traj': logLt_traj,
            'tau_history': tau_history, 'n_unique': n_unique, 'runtime': time.time() - start_time}


if __name__ == "__main__":

    # Grab data
    data = np.load("/Users/za19162/Documents/Code/integrator_snippets_exp/sonar.npy")
    y = -data[:, 0]  # Shape (208,)
    Z = data[:, 1:]  # Shape (208, 61)
    scales = np.array([5] * 61)
    scales[0] = 20

    # Settings
    n_runs = 100
    overall_seed = np.random.randint(low=10, high=29804393)  # 1234
    seeds = np.random.default_rng(overall_seed).integers(low=1, high=10000, size=n_runs)
    _epsilons = [0.001, 0.1, 2.0]
    _N = 1000
    _T = 100

    for eps_ix, _epsilon in enumerate(_epsilons):
        print("Epsilon: ", _epsilon)
        results = []
        for i in range(n_runs):
            res = {'N': _N, 'T': _T}
            out = smc_hmc_int_snip(N=_N, T=_T, epsilon_init=_epsilon, ESSrmin=0.8, _y=y, _Z=Z, _scales=scales,
                                   verbose=False, seed=int(seeds[i]), adaptive=True)
            res.update({'type': 'tempering', 'logLt': out['logLt'], 'waste': False, 'out': out})
            print("\t\tRun ", i, " LogLt: ", out['logLt'])
            results.append(res)

        with open(f"results/aan/eps_ix{eps_ix}_adaptive_100runs_timed.pkl", "wb") as file:
            pickle.dump(results, file)
