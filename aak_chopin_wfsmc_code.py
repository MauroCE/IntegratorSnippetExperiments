#!/usr/bin/env python

import numpy as np
import seaborn as sb
from matplotlib import pyplot as plt
from numpy import random

import particles.particles
from particles.particles import datasets as dts
from particles.particles import distributions as dists
from particles.particles import smc_samplers as ssps
from particles.particles.collectors import Moments
import pickle

datasets = {'sonar': dts.Sonar}
dataset_name = 'sonar'  # choose one of the three
data = datasets[dataset_name]().data
T, p = data.shape

# Standard SMC: N is number of particles, K is number of MCMC steps
# Waste-free SMC: M is number of resampled particles, P is length of MCMC
# chains (same notations as in the paper)
# All of the runs are such that N*K or M*P equal N0
alg_type = 'tempering'
N0 = 10**4
Ms = [10, 20, 50, 100, 200]

# prior & model
scales = 5. * np.ones(p)
scales[0] = 20.  # intercept has a larger scale
prior = dists.StructDist({'beta': dists.MvNormal(scale=scales, cov=np.eye(p))})


class LogisticRegression(ssps.StaticModel):
    def logpyt(self, theta, t):
        # log-likelihood factor t, for given theta
        lin = np.matmul(theta['beta'], data[t, :])
        return - np.logaddexp(0., -lin)


nruns = 100
results = []

# runs
for M in Ms:
    print("N: ", M)
    for i in range(nruns):
        print("\tRun: ", i)
        # need to shuffle the data for IBIS
        random.shuffle(data)
        model = LogisticRegression(data=data, prior=prior)
        N, lc = M, N0 // M
        res = {'M': M, 'P': lc}
        waste = True
        fk = ssps.AdaptiveTempering(model=model, len_chain=lc, wastefree=waste)
        pf = particles.particles.SMC(fk=fk, N=N, collect=[Moments], verbose=False)
        pf.run()
        print('loglik: %f' % pf.logLt)
        res.update({'type': alg_type,
                    'out': pf.summaries,
                    'waste': waste,
                    'cpu': pf.cpu_time})
        results.append(res)


with open("results/aak_chopin_wfsmc_code/wf_smc_chopin.pkl", "rb") as file:
    pickle.dump(results, file)