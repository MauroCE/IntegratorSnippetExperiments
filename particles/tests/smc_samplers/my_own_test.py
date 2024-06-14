"""
For now, I am copying https://github.com/nchopin/particles/blob/master/papers/wastefreeSMC/logistic.py.
"""
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sb
import particles
from particles import datasets as dts
from particles import distributions as dists
from particles import smc_samplers as ssps
from particles.collectors import Moments


datasets = {'pima': dts.Pima, 'eeg': dts.Eeg, 'sonar': dts.Sonar}
dataset_name = 'sonar'  # choose one of the three
data = datasets[dataset_name]().data
T, p = data.shape  # T=208, p=61

# if dataset_name == 'sonar':
alg_type = 'tempering'
N0 = 2 * 10**5
Ks = [5, 20]#, 100, 500, 1000]   # K number of MCMC steps
Ms = [50, 100]#, 200, 400, 800]  # M number of resampled particles in Waste-Free smc
# We choose the parameters so that NK = N0 and MP = N0 where P is the length of MCMC chain


# prior & model
scales = 5. * np.ones(p)
scales[0] = 20.  # intercept has a larger scale
prior = dists.StructDist({'beta':dists.MvNormal(scale=scales,
                                                cov=np.eye(p))})

class LogisticRegression(ssps.StaticModel):
    def logpyt(self, theta, t):
        # log-likelihood factor t, for given theta
        lin = np.matmul(theta['beta'], data[t, :])
        return - np.logaddexp(0., -lin)

nruns = 10  #100
results = []

# runs
print('Dataset: %s' % dataset_name)
for M, K in zip(Ms, Ks):
    for i in range(nruns):
        # need to shuffle the data for IBIS
        np.random.shuffle(data)
        model = LogisticRegression(data=data, prior=prior)
        for waste in [False]:
            if waste:
                # M = Number of resampled particles, i.e. the number of actual SMC particles, number of snippets
                # N0 // M is P, the length of the MCMC chain
                N, lc = M, N0 // M   # lc is length of chain, here it is P
                res = {'M': M, 'P': lc}
            else:
                # Otherwise, if we don't do waste-free then N is N0//K
                # lc is the length of the chain, which is K+1 (beginning plus K)
                N, lc = N0 // K, K + 1
                res = {'N': N, 'K': K}
            fk = ssps.AdaptiveTempering(model=model, len_chain=lc, wastefree=waste)
            pf = particles.SMC(fk=fk, N=N, collect=[Moments], verbose=True)
            print('%s, waste:%i, lc=%i, run %i' % (alg_type, waste, lc, i))
            pf.run()
            print('CPU time (min): %.2f' % (pf.cpu_time / 60))
            print('loglik: %f' % pf.logLt)
            res.update({'type': alg_type,
                        'out': pf.summaries,
                        'waste': waste,
                        'cpu': pf.cpu_time})
            results.append(res)

# plots
#######
savefigs = True  # do you want to save figures as pdfs
plt.style.use('ggplot')

algs = ['std', 'wf']
colors = {'std': 'black', 'wf': 'white'}
titles = {'std': 'standard SMC', 'wf': 'waste-free SMC'}
plots = {'log marginal likelihood': lambda rout: rout.logLts[-1],
         'post expectation average pred':
         lambda rout: np.mean(rout.moments[-1]['mean']['beta'])
        }

for plot, func in plots.items():
    fig, axs = plt.subplots(1, 2, sharey=True)
    for alg, ax in zip(algs, axs):
        if titles[alg] == 'waste-free SMC':
            rez = [r for r in results if r['waste']]
            xlab = 'M'
            ylab = ''
        else:
            rez = [r for r in results if not r['waste']]
            xlab = 'K'
            ylab = plot
        sb.boxplot(x=[r[xlab] for r in rez],
                   y=[func(r['out']) for r in rez],
                   color=colors[alg], ax=ax)
        ax.set(xlabel=xlab, title=titles[alg], ylabel=ylab)
        fig.tight_layout()
    if savefigs:
        fig.savefig(f'{dataset_name}_boxplots_{plot}.pdf')
