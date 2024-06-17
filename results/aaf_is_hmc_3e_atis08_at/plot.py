import pickle
import matplotlib.pyplot as plt
import numpy as np

with open("T100/eps00001_T100_N1000.pkl", "rb") as file:
    data = pickle.load(file)

ess_by_group = np.vstack(data[0]['out']['ess_by_group'])
gammas = data[0]['out']['gammas']
eps_history = np.vstack(data[0]['out']['epsilons_history'])
Ts_history = np.vstack(data[0]['out']['Ts_history'])
tau_history = np.array(data[0]['out']['tau_history'])
ess = np.array(data[0]['out']['ess'])


fig, ax = plt.subplots(ncols=2, nrows=2, figsize=(10, 6))
# Plot 1 - Folded ESS by group
for g in range(3):
    ax[0, 0].plot(gammas[1:], ess_by_group[:, g], label=f'Group {g}')
ax[0, 0].set_xscale('log')
ax[0, 0].axhline(y=1000*0.8/3, color='black', linestyle='--', label=r'$\alpha N / 3$')
ax[0, 0].legend()
ax[0, 0].set_ylabel("Folded ESS on trajectory weight.")
# Plot 2 - Step size by group
for g in range(3):
    ax[0, 1].plot(gammas, eps_history[:, g], label=f'Group {g}')
ax[0, 1].set_xscale('log')
ax[0, 1].set_yscale('log')
ax[0, 1].set_ylabel("Step size")
ax[0, 1].set_xlabel("Tempering Parameter")
ax[0, 1].legend()
# Plot 2 - Integration Steps
for g in range(3):
    ax[1, 0].plot(gammas, Ts_history[:, g], label=f'Group {g}')
ax[1, 0].set_xscale('log')
ax[1, 0].set_yscale('log')
ax[1, 0].set_ylabel("Integration Steps")
ax[1, 0].set_xlabel("Tempering Parameter")
ax[1, 0].legend()
# Plot 3 - Total ESS
ax[1, 1].plot(gammas[1:], ess)
ax[1, 1].set_xscale('log')
# ax[1, 1].set_yscale('log')
ax[1, 1].set_ylabel("Total folded ESS")
ax[1, 1].set_xlabel("Tempering Parameter")
ax[1, 1].axhline(y=1000*0.8, color='black', linestyle='--', label=r'$\alpha N$')
ax[1, 1].legend()
ax[1, 1].set_ylim([0, 1000])
plt.tight_layout()
plt.show()


# ESS over tempering
# fig, ax = plt.subplots()
# for g in range(3):
#     ax.plot(gammas[1:], ess_by_group[:, g], label=f'Group {g}')
# ax.set_xscale('log')
# ax.axhline(y=1000*0.8/3, color='black', linestyle='--', label=r'$\alpha N / 3$')
# ax.legend()
# plt.show()

# Step size over tempering
# fig, ax = plt.subplots()
# for g in range(3):
#     ax.plot(gammas, eps_history[:, g], label=f'Group {g}')
# ax.set_xscale('log')
# ax.set_yscale('log')
# ax.set_ylabel("Step size")
# ax.set_xlabel("Tempering Parameter")
# ax.legend()
# plt.show()

# INTEGRATION STEPS/TEMPERING
# fig, ax = plt.subplots()
# for g in range(3):
#     ax.plot(gammas, Ts_history[:, g], label=f'Group {g}')
# ax.set_xscale('log')
# ax.set_yscale('log')
# ax.set_ylabel("Integration Steps")
# ax.set_xlabel("Tempering Parameter")
# ax.legend()
# plt.show()

# INTEGRATION TIME / TEMPERING
# fig, ax = plt.subplots()
# ax.plot(gammas, tau_history)
# ax.set_xscale('log')
# ax.set_yscale('log')
# ax.set_ylabel("Integration time")
# ax.set_xlabel("Tempering Parameter")
# ax.legend()
# plt.show()

# INTEGRATION TIME / TEMPERING
# compute it manually
# fig, ax = plt.subplots()
# for g in range(3):
#     ax.plot(gammas, (eps_history * Ts_history)[:, g], label=f'Group {g}')
# ax.set_xscale('log')
# ax.set_yscale('log')
# ax.set_ylabel("Integration time")
# ax.set_xlabel("Tempering Parameter")
# ax.legend()
# plt.show()
