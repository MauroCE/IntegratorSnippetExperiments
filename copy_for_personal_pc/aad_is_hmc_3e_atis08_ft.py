"""
INTEGRATOR SNIPPET.

- NEW GAMMA FOUND USING "IMPORTANCE PART" OF THE WEIGHT.
- THREE GROUPS SHARE EXACTLY THE SAME STEP SIZE AND INTEGRATION STEPS: THIS IS JUST FOR BENCHMARKING HERE.

USER WILL PROVIDE AN INITIAL INTEGRATION TIME THROUGH A TMAX AND AN INITIAL STEP SIZE. THEN WE WILL KEEP INTEGRATION
TIME FIXED AND VARY BOTH THE STEP SIZE AND THE NUMBER OF INTEGRATION STEPS.
"""
import numpy as np
from scipy.special import logsumexp
from aaa_logistic_functions import sample_prior, next_annealing_param
import numpy.linalg as la
import pickle


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


def normalise_segmented_weights_and_compute_folded_ess(logw, iotas, Ts):
    """Computes normalised weights W, folded ESS for each group, and total folded ESS. Importantly, notice that each of
    the three groups uses a different T, and therefore we need to only consider the correct amount of logw.

    Parameters
    ----------
    :param logw: Log-weights, shape `(N, T+1)`
    :param iotas: Group assignment for each particle, shape `(N, )`
    :type iotas: np.ndarray
    :param Ts: Number of integration steps for each group, shape `(3, )`
    """
    ess_folded = []
    logw_folded = np.full(len(iotas), np.nan)
    for group in range(3):
        flag = iotas == group
        # Set to -np.inf the weights that are larger than the given T
        logw[flag, Ts[group]+1:] = -np.inf
        # Compute folded ESS for group i. (Next) Folded Weights for group i, shape `(n_i, )`
        logw_i_folded = logsumexp(logw[flag, :Ts[group]+1], axis=1) - np.log(Ts[group]+1)
        # Normalised folded weights for group i
        W_i_folded = np.exp(logw_i_folded - logsumexp(logw_i_folded))
        ess_folded.append(1 / np.sum(W_i_folded**2))  # Folded ESS for group i
        # Store logw FOLDED for group i
        logw_folded[flag] = logw_i_folded
    # Compute unfolded normalized weights
    W_unfolded = np.exp(logw - logsumexp(logw))
    # Compute total folded ESS
    ess_folded_total = 1 / np.sum(np.exp(logw_folded - logsumexp(logw_folded))**2)
    return W_unfolded, np.array(ess_folded), ess_folded_total, logw_folded


def smc_hmc_int_snip(N, T, epsilon, _y, _Z, _scales, ESSrmin=0.9, seed=1234, verbose=False):
    """
    Weight Decomposition
    --------------------
    We use the importance part of the (unfolded) weights to select the next tolerance.
    The trajectory part of the (unfolded) weights is used to compute the folded weights from mu_n to mu_n, and we
    compute the respective folded ESS for the three groups and use it to determine how good or bad the T/epsilon values
    are.

    Hyperparameters
    ---------------
    User provides the maximum sequential budget (Tmax) and the initial epsilon (epsilon_init), thus specifying a total
    integration time tau = Tmax * epsilon_init. For now, we keep this fixed.


    Iteration 0
    -----------
    We run all particles with T=Tmax and epsilon=epsilon_init, to get a baseline and see if it is even sensible.
    """
    # Setup
    rng = np.random.default_rng(seed=seed)
    verboseprint = print if verbose else lambda *a, **kwargs: None
    # Hyperparameter settings
    tau = T * epsilon    # Fixed
    Ts = np.array([T, T, T])  # Fixed
    epsilons = np.array([tau/t for t in Ts])  # Fixed to epsilon_init

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
    n = 1

    while gammas[n-1] < 1.0:
        verboseprint("Iteration: ", n, " Gamma: ", gammas[n-1])

        # --- CHOOSE WHICH PARTICLES WILL BE ASSIGNED TO WHICH EPSILON/T COMBINATION.
        iotas = np.array(rng.choice(a=len(epsilons), size=N))  # one flag for each particle
        epsilons_vector = epsilons.reshape(-1, 1)[iotas]  # use for integration

        # --- CONSTRUCT TRAJECTORIES USING psi_{n-1} ---
        # IMPORTANT: To maintain vectorization (since kappa is not too large) I will construct all trajectories up to
        # the largest of the Ts and then in case drop some trajectory points when needed.
        # Setup storage
        trajectories = np.full((N, Ts[0]+1, 2*d), np.nan)
        trajectories[:, 0] = np.hstack((x, v))
        # Store only the log densities (used for log-weights), not gradients
        nlps = np.full((N, Ts[0]+1), np.nan)   # Negative log priors
        nlls = np.full((N, Ts[0]+1), np.nan)   # Negative log-likelihoods
        # First half-momentum step
        nlps[:, 0], gnlps, nlls[:, 0], gnlls = nlp_gnlp_nll_and_gnll(x, _y, _Z, _scales)
        v = v - 0.5*epsilons_vector*(gnlps + gammas[n-1]*gnlls)  # (N, 61)
        # T-1 full position and momentum steps
        for k in range(Ts[0] - 1):
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
        # IMPORTANT: The log weights are all computed for the trajectories up to Ts[-1] but this is only done to
        # preserve vectorization. In practice, trajectories are of different lengths (except the first iteration).
        W, ess_folded_by_group, ESS_folded, logw_folded = normalise_segmented_weights_and_compute_folded_ess(
            logw, iotas, Ts)
        ess_by_group.append(ess_folded_by_group)
        verboseprint("\tTrajectories and weights computed. Folded ESS: ", ESS_folded)
        verboseprint("\tESSs by group: ", ess_folded_by_group)

        # --- RESAMPLING ---
        # Importantly, notice that the pool of possible indices to resample from is not so trivial
        # One needs to compute it
        A = rng.choice(a=N*(Ts[0]+1), size=N, replace=True, p=W.ravel())
        # Unravelling requires segmentation by particle subgroup
        n_indices, k_indices = np.unravel_index(A, (N, Ts[0]+1))
        x = trajectories[n_indices, k_indices, :d]  # (N, 61)
        pms.append(
            [np.sum(k_indices[iotas[n_indices] == group] > 0) / np.sum(iotas[n_indices] == group) for group in range(3)]
        )
        mips.append([np.median(k_indices[iotas[n_indices] == group]) / Ts[group] for group in range(3)])
        pds.append(
            [(len(np.unique(n_indices[iotas[n_indices] == group])) - 1) / (np.sum(iotas == group) - 1)
             for group in range(3)]
        )
        mpds.append(np.sqrt(np.array(mips[-1]) * np.array(pds[-1])))
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

        ess.append(ESS_folded)

        n += 1

    return {'logLt': logLt, 'pms': pms, 'mips': mips, 'ess': ess, 'longest_batches': longest_batches,
            'ess_running': ess_running, 'Ts': Ts, 'epsilons_history': epsilons_history, 'kappas': kappas,
            'ess_by_group': ess_by_group, 'pds': pds, 'mpds': mpds, 'gammas': gammas, 'logLt_traj': logLt_traj}


if __name__ == "__main__":

    # Grab data
    data = np.load("/Users/za19162/Documents/Code/integrator_snippets_exp/sonar.npy")
    y = -data[:, 0]  # Shape (208,)
    Z = data[:, 1:]  # Shape (208, 61)
    scales = np.array([5] * 61)
    scales[0] = 20

    # Settings
    n_runs = 1
    overall_seed = 1234
    seeds = np.random.default_rng(overall_seed).integers(low=1, high=10000, size=n_runs)
    results = []

    for i in range(n_runs):
        _N = 1000
        _T = 100
        print("T: ", _T)
        res = {'N': _N, 'T': _T}
        out = smc_hmc_int_snip(N=_N, T=_T, epsilon=2, ESSrmin=0.8, _y=y, _Z=Z, _scales=scales, verbose=True,
                               seed=int(seeds[i]))
        res.update({'type': 'tempering', 'logLt': out['logLt'], 'waste': False, 'out': out})
        print("\t\tLogLt: ", out['logLt'])
        results.append(res)

    # Save data
    # with open("results/aad_is_hmc_3e_atis08_ft/T100/eps2_T100_N1000.pkl", "wb") as file:
    #     pickle.dump(results, file)
