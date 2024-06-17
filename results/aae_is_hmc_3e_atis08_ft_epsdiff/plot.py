import pickle
import numpy as np
import matplotlib.pyplot as plt

with open("eps001_01_1_T250_N1000.pkl", "rb") as file:
    data = pickle.load(file)


with open("eps01_01_05_T250_N1000.pkl", "rb") as file:
    data05 = pickle.load(file)
    ess_group05 = np.vstack(data05[0]['out']['ess_by_group'])[:, -1]
    gammas05 = data05[0]['out']['gammas']

with open("eps01_01_5_T250_N1000.pkl", "rb") as file:
    data5 = pickle.load(file)
    ess_group5 = np.vstack(data5[0]['out']['ess_by_group'])[:, -1]
    gammas5 = data5[0]['out']['gammas']


# ESS folded by group
N = 1000
ESSrmin = 0.8

ess_by_group = np.vstack(data[0]['out']['ess_by_group'])
gammas = data[0]['out']['gammas']
ess = data[0]['out']['ess']

colors1 = ['lightcoral', 'indianred', 'firebrick']
colors05 = ['violet', 'mediumorchid', 'darkorchid']
colors01 = ['lightblue', 'deepskyblue', 'dodgerblue']
colors001 = ['lightgreen', 'forestgreen', 'darkgreen']
colors0001 = ['khaki', 'gold', 'goldenrod']

colors = ['indianred', 'mediumorchid', 'goldenrod', 'deepskyblue', 'forestgreen']


fig, ax = plt.subplots()
epsilons = [0.01, 0.1, 1.0]
# Grouped folded ESS
for g in range(3):
    ax.plot(gammas[1:], ess_by_group[:, g], label=f'eps={epsilons[g]}', c=colors[g], marker='o', markersize=4)
ax.plot(gammas05[1:], ess_group05, label='eps=0.5', c=colors[-2], marker='o', markersize=4)
ax.plot(gammas5[1:], ess_group5, label='eps=5.0', c=colors[-1], marker='o', markersize=4)
# Total ESS
# ax.plot(gammas1[1:], ess1, label='eps=1', c='red')
# ax.plot(gammas05[1:], ess05, label='eps=0.5', c='violet')
# ax.plot(gammas01[1:], ess01, label='eps=0.1', c='blue')
# ax.plot(gammas001[1:], ess001, label='eps=0.01', c='green')
# ax.plot(gammas0001[1:], ess0001, label='eps=0.001', c='yellow')
ax.axhline(y=N*ESSrmin/3, color='black', linestyle='--', label=r'$\alpha N / 3$')
ax.set_xscale('log')
ax.set_ylabel("Folded ESS on " + r"$\mathregular{\mu_n(\psi^k(z)) / \mu_n(z)}$")
ax.set_xlabel("Log Tempering Parameter")
ax.legend()
plt.show()


# LOG NORMALIZING CONSTANT (RUNNING)
# lnc = np.cumsum(data[0]['out']['logLt_traj'])
#
# ms = 4
# marker = 'o'
# fig, ax = plt.subplots()
# ax.plot(gammas, lnc, label='eps=1', c='red', marker=marker, ms=ms)
# ax.axhline(y=-125, color='black', linestyle='--')
# ax.set_xscale('log')
# ax.set_ylabel("Running log normalizing constant")
# ax.set_xlabel("Log Tempering Parameter")
# ax.legend()
# plt.show()
