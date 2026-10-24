# What you can honestly say (and what to avoid)

For your applications. Every statement here is backed by the run in `results/airfrans/`.

## Safe to say

- "I built an ML surrogate for airfoil lift and drag trained on the AirfRANS dataset: 1,000 steady 2D RANS solutions
  (OpenFOAM, k-omega SST) of NACA 4- and 5-digit airfoils at Reynolds numbers of 2 to 6 million."
- "I computed lift and drag coefficients by integrating surface pressure and wall shear stress from the stored fields."
- "An RBF-kernel SVR reached a Cl error of 0.006 and a Cd error of about 1.6 drag counts on the held-out test set,
  with Cl/Cd rank correlation 0.998; random forest and MLP were less accurate."
- "Using the surrogate inside a genetic algorithm, a search of about 6,000 evaluations took under 3 seconds,
  versus an estimated 100+ days with CFD at about 25 minutes per solution (an estimate from the dataset paper)."
- "I compared the surrogate with NASA TM 4074 wind-tunnel data for the NACA 0012 at Re = 6 million. It tracks the
  experiment to within a few hundredths in Cl up to the edge of stall, and fails at stall: it predicts Cl 1.55 to 1.8
  where the measured value is about 1.1, and under-predicts drag by a factor of 7 to 11."
- "I showed that the random forest's own uncertainty estimate did not warn of the failure, and explained why from the
  physics of boundary-layer separation."

## Do not say

- That you trained on 9,000 geometries. The dataset has 1,000 simulations.
- That the optimal airfoil is "optimal" or "verified". It is the surrogate's prediction, it sits at the edge of the
  design space, and the three models disagree about it. Say "surrogate-predicted optimum".
- That the CFD cost was measured. It is an estimate based on the dataset paper's figure for one solution.
- That the surrogate "learns stall" or "predicts stall". It does not.
- That the failure is proven by a statistical test. It is six post-stall experimental points from one source.
- That the surrogate-versus-experiment gap at stall is the surrogate's error alone. It also contains RANS model
  error and wind-tunnel three-dimensionality.
- That the data are "high fidelity" or "experimental". They are RANS simulations.
