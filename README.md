# Integrator Snippet Experiments
Experiments about adaptivity with Integrator Snippets.

- `logistic_functions.py`: Functions for the Sonar logistic regression problem.

# Nomenclature

- `3e` means we use three groups with (potentially) different epsilons and different T.
- `atis08` means that we choose the next tempering parameter adaptively, and we do so using the importance part of the weights (`is`) and we aim to keep the ESS on this part of the weights at `0.8N`, where `N` is the number of particles.
- `ft` means that tau, the total integration time, is fixed. 
- `at` means that the total integration time is chosen adaptively (using U-turns most likely).
- `epsdiff` means that in the three groups we assume the step sizes are different (and so the number of integration steps), but we do not adapt.
- `ua` means "unified adaptation", meaning we don't distinguish between iterations where T and epsilons are all equal, vs those where they are different.

# Thoughts

1. KNOWN: If we choose the tempering parameter using the ESS from the importance part of the weights, then the tempering schedule will not depend on the step size of the integrator.
2. KNOWN: If we sum up the folded ESS by groups, we pretty much obtain the total folded ESS.
3. EXPERIMENT: Run an integrator snippet with fixed step size and number of integration steps. Choose gamma based on the importance part of the weight. Keep track of the folded ESS on the trajectory part of the weight. Run this for a few different step sizes. The aim is to see how the folded-trajectory-ESS evolves with different step sizes.