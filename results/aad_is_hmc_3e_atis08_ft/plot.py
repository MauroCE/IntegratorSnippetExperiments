import pickle
import numpy as np
import matplotlib.pyplot as plt

folder = "T100"

with open(f"{folder}/eps1_{folder}_N1000.pkl", "rb") as file:
    data1 = pickle.load(file)
with open(f"{folder}/eps05_{folder}_N1000.pkl", "rb") as file:
    data05 = pickle.load(file)
with open(f"{folder}/eps01_{folder}_N1000.pkl", "rb") as file:
    data01 = pickle.load(file)
with open(f"{folder}/eps001_{folder}_N1000.pkl", "rb") as file:
    data001 = pickle.load(file)
with open(f"{folder}/eps0001_{folder}_N1000.pkl", "rb") as file:
    data0001 = pickle.load(file)

# Log-gamma / Iterations
# fig, ax = plt.subplots()
# ax.plot(data01[0]['out']['gammas'], label='Gammas')
# ax.set_yscale('log')
# plt.show()


# ESS folded by group
N = 1000
ESSrmin = 0.8

ess_by_group1 = np.vstack(data1[0]['out']['ess_by_group'])
gammas1 = data1[0]['out']['gammas']
ess1 = data1[0]['out']['ess']

ess_by_group05 = np.vstack(data05[0]['out']['ess_by_group'])
gammas05 = data05[0]['out']['gammas']
ess05 = data05[0]['out']['ess']

ess_by_group01 = np.vstack(data01[0]['out']['ess_by_group'])
gammas01 = data01[0]['out']['gammas']
ess01 = data01[0]['out']['ess']

ess_by_group001 = np.vstack(data001[0]['out']['ess_by_group'])
gammas001 = data001[0]['out']['gammas']
ess001 = data001[0]['out']['ess']

ess_by_group0001 = np.vstack(data0001[0]['out']['ess_by_group'])
gammas0001 = data0001[0]['out']['gammas']
ess0001 = data0001[0]['out']['ess']

colors1 = ['lightcoral', 'indianred', 'firebrick']
colors05 = ['violet', 'mediumorchid', 'darkorchid']
colors01 = ['lightblue', 'deepskyblue', 'dodgerblue']
colors001 = ['lightgreen', 'forestgreen', 'darkgreen']
colors0001 = ['khaki', 'gold', 'goldenrod']


fig, ax = plt.subplots()
# Grouped folded ESS
# for g in range(3):
#     ax.plot(gammas1[1:], ess_by_group1[:, g], label=f'Group {g}, eps=1', c=colors1[g])
#     ax.plot(gammas05[1:], ess_by_group05[:, g], label=f'Group {g}, eps=0.1', c=colors05[g])
#     ax.plot(gammas01[1:], ess_by_group01[:, g], label=f'Group {g}, eps=0.1', c=colors01[g])
#     ax.plot(gammas001[1:], ess_by_group001[:, g], label=f'Group {g}, eps=0.01', c=colors001[g])
#     ax.plot(gammas0001[1:], ess_by_group0001[:, g], label=f'Group {g}, eps=0.01', c=colors0001[g])
ax.plot(gammas1[1:], ess_by_group1.sum(axis=1), label='eps=1', c='red')
ax.plot(gammas05[1:], ess_by_group05.sum(axis=1), label='eps=0.1', c='violet')
ax.plot(gammas01[1:], ess_by_group01.sum(axis=1), label='eps=0.1', c='blue')
ax.plot(gammas001[1:], ess_by_group001.sum(axis=1), label='eps=0.01', c='green')
ax.plot(gammas0001[1:], ess_by_group0001.sum(axis=1), label='eps=0.01', c='yellow')
# Total ESS
# ax.plot(gammas1[1:], ess1, label='eps=1', c='red')
# ax.plot(gammas05[1:], ess05, label='eps=0.5', c='violet')
# ax.plot(gammas01[1:], ess01, label='eps=0.1', c='blue')
# ax.plot(gammas001[1:], ess001, label='eps=0.01', c='green')
# ax.plot(gammas0001[1:], ess0001, label='eps=0.001', c='yellow')

ax.set_xscale('log')
# ax.axhline(y=N/3, color='r', linestyle='--', label='N/3')
# ax.axhline(y=ESSrmin*N/3, linestyle='--', label='0.8*(N/3)')
ax.set_ylabel("Folded ESS on " + r"$\mathregular{\mu_n(\psi^k(z)) / \mu_n(z)}$")
ax.set_xlabel("Log Tempering Parameter")
# ax.set_ylim(0, N)
ax.legend()
plt.show()


# LOG NORMALIZING CONSTANT (RUNNING)
lnc1 = np.cumsum(data1[0]['out']['logLt_traj'])
lnc05 = np.cumsum(data05[0]['out']['logLt_traj'])
lnc01 = np.cumsum(data01[0]['out']['logLt_traj'])
lnc001 = np.cumsum(data001[0]['out']['logLt_traj'])
lnc0001 = np.cumsum(data0001[0]['out']['logLt_traj'])

ms = 4
marker = 'o'
fig, ax = plt.subplots()
ax.plot(gammas1, lnc1, label='eps=1', c='red', marker=marker, ms=ms)
ax.plot(gammas05, lnc05, label='eps=0.5', c='violet', marker=marker, ms=ms)
ax.plot(gammas01, lnc01, label='eps=0.1', c='blue', marker=marker, ms=ms)
ax.plot(gammas001, lnc001, label='eps=0.01', c='green', marker=marker, ms=ms)
ax.plot(gammas0001, lnc0001, label='eps=0.001', c='yellow', marker=marker, ms=ms)
ax.axhline(y=-125, color='black', linestyle='--')
ax.set_xscale('log')
ax.set_ylabel("Running log normalizing constant")
ax.set_xlabel("Log Tempering Parameter")
ax.legend()
plt.show()
