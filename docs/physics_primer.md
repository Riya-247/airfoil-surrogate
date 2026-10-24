# Physics primer: what the surrogate is learning, and why stall breaks it

Written for someone who knows algorithms and numerical methods but has not taken fluid mechanics.
Sections 1-3 define the quantities. Sections 4-6 explain the physics of the lift curve and stall.
Section 7 explains what CFD and RANS compute. Section 8 is the argument the project is built around.
Section 9 explains each modelling choice.

---

## 1. The problem in one paragraph

An airfoil is the 2D cross-section of a wing. Air flows past it at some speed and angle. The air pushes on the
surface in two ways: **pressure** acts normal to the surface, and **viscous shear** (friction against the
surface) acts tangentially. Add up both over the whole surface and you get one force. The component
perpendicular to the oncoming flow is **lift** (what holds the aircraft up). The component along the flow is
**drag** (what the engine must overcome). The surrogate takes (shape, angle, speed) and predicts lift and drag.
A surrogate is useful because getting these two numbers from a RANS CFD solution costs about 25 minutes on 16
cores (AirfRANS paper, Table 4), while a trained model answers in well under a millisecond.

## 2. The quantities

| Symbol | Name | Meaning |
|---|---|---|
| c | chord | Length of the airfoil from leading edge to trailing edge (1 m in AirfRANS). All lengths are measured in chords. |
| alpha | angle of attack | Angle between the chord line and the oncoming flow. The main control a pilot has. Here -5 to 15 degrees. |
| U | freestream speed | Speed of the undisturbed air far ahead of the airfoil. Here about 31-94 m/s. |
| q = 0.5 rho U^2 | dynamic pressure | The natural pressure scale of the flow. Forces are divided by it to make them dimensionless. |
| Cl = L / (q c) | lift coefficient | Lift per unit span, made dimensionless. Typically around 0 to 1.5 for the attached regime. |
| Cd = D / (q c) | drag coefficient | Same for drag. Typically 0.005-0.02 here. 1 "drag count" = 0.0001. |
| Cl/Cd | lift-to-drag ratio | The efficiency figure. Higher means more lift for less drag. This is what the design search maximises. |
| Re = U c / nu | Reynolds number | Ratio of inertial to viscous effects in the flow. See below. nu is the air's kinematic viscosity. |
| M = U / a | Mach number | Speed relative to the speed of sound a. Below about 0.3, air behaves as if it were incompressible. |
| Cp | pressure coefficient | Surface pressure, made dimensionless the same way. The lift is the integral of Cp around the airfoil. |

**Why dimensionless coefficients?** A larger or faster airfoil produces more force, but the *shape of the
flow* is the same if Re and Mach (and alpha, and the shape) are the same. Cl and Cd strip out size and speed,
so one number describes the airfoil's behaviour across scales. That is the reason the surrogate can predict Cl
and Cd from just geometry, alpha and Re.

**What Reynolds number means physically.** Right at the surface, air sticks to it (no-slip), so there is a thin
layer, the **boundary layer**, where the speed climbs from zero to the freestream value. Re controls how thin
that layer is and whether it is smooth (laminar) or chaotic (turbulent). At Re of 2-6 million the boundary layer
on the airfoil is turbulent over most of its length, and its thickness is on the order of a percent or two of
the chord (a flat-plate turbulence correlation gives about 1.6% of the chord at Re = 6 million, which is only an
estimate). A turbulent layer resists separating from the surface better than a laminar one, which is part of why
airfoils at high Re stall later and more gracefully than at low Re.

**Why Mach is not an input here.** All the simulations have Mach below 0.3, so compressibility is negligible
and the equations solved are the incompressible ones. In the incompressible case, with the chord and air
properties fixed, the *only* parameter of the flow is Re, which in this dataset is set entirely by U. So the
flow conditions are two numbers: alpha and Re.

## 3. The input features, and what each one means

The models never see raw coordinates. They see a short vector of quantities with physical meaning
(`features.py`, feature set `physics`).

| Feature | What it is | Why it is there |
|---|---|---|
| `alpha_deg` | Angle of attack. | The dominant driver of lift. |
| `log10_Re` | Log of the Reynolds number. | Re changes boundary-layer thickness and therefore friction drag and separation. Log because its effects scale roughly with orders of magnitude, and so a tree or a kernel sees even spacing. |
| `t_max` | Maximum thickness as a fraction of chord. | Thickness raises the lift slope slightly, raises pressure drag, and strongly affects how the boundary layer handles the leading-edge suction peak (thin sections stall abruptly, thicker ones more gradually). |
| `m_max` | Maximum camber (the curvature of the mean line), signed. | Camber shifts the whole lift curve up: a cambered airfoil lifts even at zero angle. |
| `x_m` | Chordwise position of maximum camber. | Where the camber sits sets the pitching moment and the zero-lift angle. |
| `z75` | Camber at 75% chord. | A measure of aft loading. Aft camber puts more of the lift near the trailing edge. |
| `alpha_L0_deg` | Zero-lift angle from thin-airfoil theory (computed from the camber line). | The angle at which this shape produces no lift. About -2 degrees for a NACA 2412. See section 4. |
| `alpha_eff_deg` | `alpha - alpha_L0`. | The angle that actually matters for lift in the attached regime. A derived feature that makes the dominant relationship nearly linear, which trees and kernels find easy. |

There is also an alternative `cst` set: 12 Kulfan CST coefficients that describe the upper and lower surface
(a smooth fixed-length shape description that works for any airfoil, 4-digit, 5-digit or otherwise). It is
there for comparison and for extending the work to shapes beyond NACA.

## 4. Why the lift curve is linear at low angles

Cl rises in a straight line with alpha until the mid-teens of degrees. The reason:

1. **Potential-flow picture.** Away from the thin boundary layer, the flow is well described by an inviscid,
   irrotational flow. For such a flow around a lifting airfoil there is a net **circulation** Gamma around the
   section, and the lift per unit span is L = rho U Gamma (Kutta-Joukowski theorem).
2. **The Kutta condition.** Real flow leaves a sharp trailing edge smoothly. That requirement picks one specific
   circulation out of the infinite family of mathematically valid solutions.
3. **Linearity.** In thin-airfoil theory (small angles, thin shape), the equations are linear, and the result is

       Cl = 2 pi (alpha - alpha_L0)        (alpha in radians)

   a slope of about 0.11 per degree. Real airfoils come out close to this (a little below it, because the
   boundary layer effectively thickens the body; a thicker section slightly above).
4. **Why it holds.** While the boundary layer is attached and thin, it is a small correction to the potential
   flow. Doubling alpha doubles the circulation. The boundary layer stays a perturbation.

Drag in the same regime is a smooth, roughly parabolic function of lift: a base level set by skin friction
(Re, thickness) plus a term that grows with the square of the effective angle as the pressure distribution
becomes more demanding. The dataset stores these two parts separately (`Cd_pressure`, `Cd_viscous`).

## 5. What changes at stall

As alpha increases, the air accelerates over the upper surface and creates a strong low-pressure suction peak
near the leading edge. Behind the peak the air must slow down again and the pressure must climb back up toward
the trailing edge. That climb is an **adverse pressure gradient**: the fluid near the wall has little momentum
(it is slowed by friction), and a rising pressure pushes against it.

Increase alpha and the gradient steepens. The boundary layer thickens, the wall shear stress drops, and at some
point it reaches zero and then reverses: the flow near the wall runs backwards. This is **separation**. The
boundary layer lifts off the surface and a region of slow, recirculating flow forms between the surface and the
outer flow.

Consequences, which are all sudden compared with the gentle attached regime:

- **Lift stops growing.** The separated region changes the effective shape the outer flow sees, and the
  suction on the upper surface no longer recovers to the trailing-edge pressure. Cl bends over, reaches a
  maximum (Cl,max), and then falls.
- **Drag rises sharply.** The separated wake carries momentum deficit and the pressure on the rear of the
  airfoil no longer balances the front: **pressure (form) drag** takes over from skin friction.
- **Two stall styles.** Thick, moderately cambered airfoils like the NACA 0012 at these Reynolds numbers
  typically stall from the **trailing edge**: the separation point creeps forward from the rear as alpha grows,
  and the lift curve rounds over gradually. Thin airfoils tend to separate at the **leading edge** (a short
  bubble that bursts), and then Cl collapses abruptly.
- **The flow can become unsteady.** Past a certain alpha the wake sheds vortices and forces fluctuate. A
  steady-state solver may then not converge to a single answer at all. This is exactly why the AirfRANS authors
  capped alpha at 15 degrees: they chose it to avoid stall and unsteady wake patterns.

For the NACA 0012 at Reynolds numbers of a few million, stall begins somewhere in the mid-teens of degrees. The
exact numbers should be read from the Ladson experimental file (see the README), not from this text.

## 6. Why separation makes the geometry-to-force map hard to learn

Everything above compresses to one statement: **in the attached regime, the map (shape, alpha, Re) -> (Cl, Cd)
is smooth and close to linear; near and past stall it is not.** In detail:

1. **It is a change of regime, not a steeper slope.** Past stall the derivative of Cl with alpha changes sign.
   A function that was almost linear becomes non-monotonic. A model fitted to the first regime contains no
   information about the second.
2. **A tiny change in input can cause a large change in output.** Where exactly the flow separates depends
   sensitively on the leading-edge radius, thickness, camber and Re together. Two airfoils that look almost
   identical can stall at angles several degrees apart. The map has near-discontinuities.
3. **Data imbalance.** Most of any dataset sits in the attached regime. Even a dataset that includes stall has
   few samples per stall configuration, and every stall behaviour (trailing-edge, leading-edge, mixed) is a
   different local pattern to learn.
4. **The features do not carry the mechanism.** Camber, thickness and alpha describe the shape and the
   condition. Whether the boundary layer separates depends on the *pressure distribution*, which is an output
   of the flow, not an input to the model. The surrogate is asked to predict a threshold phenomenon from
   quantities that sit upstream of it.
5. **The labels themselves get less reliable.** Steady RANS with an eddy-viscosity closure is least trustworthy
   exactly where the flow separates and becomes unsteady. The NASA Turbulence Modeling Resource, which hosts the
   NACA 0012 validation case, restricts CFD-versus-experiment comparisons to angles up to the highest one that
   yields steady, non-oscillating CFD results, and notes that experiments near stall are far from
   two-dimensional. So close to stall there are two sources of divergence from experiment (the CFD and the
   wind tunnel) before the surrogate adds its own.

## 7. What CFD computes, and what a RANS solution approximates

**The governing equations.** Air motion is described by the **Navier-Stokes equations**: conservation of mass
and momentum with viscous stresses. At Re of millions the flow is turbulent: it contains eddies over a huge
range of sizes, from the airfoil scale down to fractions of a millimetre, all fluctuating in time.

**Direct simulation is too expensive.** Resolving every eddy (DNS) at these Reynolds numbers is far beyond
reach for routine engineering use. Even resolving the large eddies (LES) is orders of magnitude more costly
than what the dataset used.

**Reynolds averaging.** RANS (Reynolds-Averaged Navier-Stokes) splits each quantity into a time-averaged part
and a fluctuation, and solves only for the average. Averaging the nonlinear momentum equation produces an
extra term, the **Reynolds stress**: the average effect of the turbulent fluctuations on the mean flow. It is
unknown, so it has to be **modelled**. This is the **closure problem**.

**The closure used here.** AirfRANS uses the **k-omega SST** model (Menter). It assumes the Reynolds stress acts like
an extra viscosity (the *eddy viscosity*, `nu_t`, one of the stored fields) and computes it from two additional
transport equations, one for turbulent kinetic energy `k` and one for its specific dissipation rate `omega`.
The "SST" variant blends a near-wall formulation with a freestream one and limits the shear stress so it
behaves sensibly in adverse pressure gradients, which is why it is popular in aerodynamics.

**What this approximates, and where it fails.**

- The mean flow is solved well, and so are pressure distributions and lift in attached flow.
- The turbulence is *modelled*, not computed. The model carries assumptions (an eddy-viscosity relation,
  calibration on simple flows) that hold best for attached, near-equilibrium boundary layers.
- Strong separation violates those assumptions, so this is where RANS predictions are least reliable.
- The solution is **steady**: the solver iterates to a time-independent answer. If the real flow is unsteady,
  a steady solver cannot capture it.
- The flow is **2D** and **incompressible**, which is an idealisation of a real wing.

**How the forces are obtained.** The solver gives velocity, pressure and `nu_t` at every mesh node. The forces
come from post-processing: integrate pressure over the surface (pressure force), and integrate the wall shear
stress, computed from the velocity gradient at the wall (viscous force). Split into components along and across
the freestream, divide by `q c`, and you have Cd and Cl. The mesh has 250,000-300,000 cells and its first cell
is 2 micrometres off the wall (y+ about 1), so that the boundary layer is resolved all the way to the surface.
The dataset ships fields, not coefficients, so this integration is the step `scripts/01_build_dataset.py` performs.

## 8. The argument: why a model trained mostly on attached flow is confident and wrong near stall

**Setup.** AirfRANS covers alpha from -5 to 15 degrees by design, with the upper limit chosen to avoid stall.
Whatever the labels contain near 15 degrees, essentially all of the training data is attached or very mildly
separated flow. Cross-validation, test-set MAE and Spearman correlation inside this range will look excellent,
and should. That is the trap: they all measure interpolation inside the regime the data covers.

**Step 1: what the model actually learns.** In the attached regime, Cl is almost exactly linear in the
effective angle, with slope close to 2 pi. A model that fits this data has learned, in effect, "Cl rises
by about 0.11 per degree of effective angle, plus small corrections for thickness and Re". That is a correct
description of attached-flow physics. It is not a description of separation, because separation never occurs
in the training data.

**Step 2: what happens beyond the data.** At an angle past the training range, each model class continues
in its own way:

- A **random forest** returns averages of training labels, so it cannot predict a value outside the range of
  labels it has seen. In this project's run it flattened out near Cl = 1.56 past 15 degrees. (With the `alpha_eff`
  feature a symmetric section at 20 degrees can resemble a cambered section at a lower alpha that is in the
  training set, so a forest can in principle carry the trend further; it has no way to know the difference.)
- An **MLP with ReLU units** extends piecewise-linearly. If the lift slope was positive at the end of the
  training range, it keeps going up.
- An **RBF-kernel SVR** decays back toward the mean of its training targets away from the data.

None of the three has any mechanism that could bend the lift curve at stall, because bending needs information
that is absent. The real curve peaks and falls, so all three get Cl wrong, in different directions, and none
predicts the drag rise.

**Step 3: why it gives no warning.** The usual uncertainty signals only measure *distance to the data* or
*disagreement between components that were trained on the same data*. Trees trained on the same attached-flow
data agree on how to continue it, so the spread across trees does not respond to the physics changing. In the
Ladson comparison the forest's 2-sigma spread was about 0.19 inside the training range and 0.25 at the stalled
points, while the actual error rose from about 0.02 to about 0.46. The spread was not small, it was roughly
constant, and it measures the models' shared blind spot rather than the size of the physics being missed.
Cross-validation error is computed on data from the same distribution and cannot see the failure either.
Nothing in the model knows that the physics has changed, because "the flow separates here" is not something
the inputs encode.

**Step 4: why this is a physics statement and not just generic extrapolation.** In an ordinary smooth
function you can hope that a model extrapolates a bit. Here, the output is controlled by a threshold
phenomenon (boundary-layer separation) whose onset depends on quantities that are not in the input (the pressure
distribution). Past that threshold the map has a different structure (a maximum, a sign change, a drag
jump). A model that was never shown the threshold cannot find it from inside the attached regime, no matter how
well it fits there.

**Step 5: keep the two error sources apart.** Close to and beyond stall, any difference between the surrogate and
the wind-tunnel data has at least three contributions: the surrogate's extrapolation error, the RANS model's own
error against the real flow, and the wind-tunnel's departure from 2D. The validation plot shows the surrogate
against experiment. It cannot tell you how much of the gap is the surrogate. To isolate the surrogate's share,
compare it with the RANS labels at the same angles. That requires stall-resolved CFD labels, which AirfRANS
does not provide.

**What the project shows empirically** (Ladson NACA 0012 data, Re = 6 million, 53 points; the numbers are in the
README, section 5):

- Experimental lift peaks at Cl = 1.635 near 17.2 degrees and falls to about 1.0 to 1.2 by 18 degrees, with drag
  jumping from about 0.025 to 0.19 to 0.43.
- Up to the edge of stall the SVR and MLP track experiment (Cl error about 0.025) even beyond the 15-degree limit
  of the training data, because lift stays linear until the flow separates.
- At the six stalled points all three models predict Cl of 1.55 to 1.80 against a measured mean of 1.10, and
  under-predict drag by a factor of 7 to 11.
- The random forest's own uncertainty rose by about 30% across the boundary while its error rose more than
  twenty-fold.
- `02_train.py --task aoa` shows the smaller, attached-flow version of the same effect inside the dataset: beyond
  the training range all three models under-predict drag, while lift extrapolates much better.

Caveats: six post-stall points from a single experiment; and near stall the gap to experiment also contains RANS
model error and wind-tunnel three-dimensionality, which this comparison cannot separate from the surrogate's own error.

**What would reduce the problem** (relevant to "extend it myself"): include stall-resolved data (for example
the NREL HAM2D airfoil set, which goes to 20 degrees); add a regime classifier (attached or separated) and treat
it as a separate learning problem; use features that carry the mechanism (pressure-gradient or boundary-layer
shape indicators from a cheap panel code); use out-of-distribution detection based on distance in feature space;
and make the design search refuse to propose designs near the edge of the data.

## 9. Modelling choices and why

| Choice | Why |
|---|---|
| Predict `[Cl, log10(Cd)]`, not `[Cl, Cd]` | Drag is strictly positive and spans a decade or more. In log space the errors are relative, and the inverse transform can never give negative drag, which would make Cl/Cd meaningless. |
| One model predicts both targets | Lift and drag share inputs and structure. Targets are standardised inside the model so neither dominates the loss. |
| Geometry as physical descriptors (default) | 8 interpretable inputs for roughly 800 training samples, instead of hundreds of coordinates. Each one has a physical effect, and the model works for 4- and 5-digit airfoils alike. |
| `alpha_eff = alpha - alpha_L0` as a feature | Puts the strongest relationship (linear lift curve) directly in the inputs. Without it, a tree model must approximate the shift by cambered shape with many splits. |
| CV grouped by geometry | If one airfoil appeared at several angles, a random split would put near-duplicates on both sides. Grouping measures generalisation to *new shapes*. (In AirfRANS each simulation has its own airfoil, so groups are single rows, but the machinery is correct if you add multi-alpha data.) |
| Nested cross-validation | Hyper-parameters are re-tuned inside every outer fold, so the CV error is not optimistic from tuning on the test fold. |
| Official train/test split kept | AirfRANS publishes an 800/200 split. Using it makes results comparable and avoids leakage between train and test. |
| MAE (Cl in absolute units; Cd in drag counts) | MAE is robust and in physical units. A Cl error of 0.02 or a drag error of 5 counts is something an aerodynamicist can judge. |
| Spearman rank correlation of Cl/Cd | The AirfRANS authors argue, correctly, that for design the *ranking* of candidates matters more than the absolute error. A surrogate that orders designs correctly finds the right optimum even if its values are off. |
| MAE split at alpha >= 10 degrees | The closest available proxy for "near stall" given the data stops at 15. |
| `--task aoa` | The dataset's own extrapolation split, giving an in-dataset measure of what extrapolation costs. |
| GA bounds equal to the data's sampling ranges | The search stays where the model has seen data. Otherwise the optimiser would exploit extrapolation artefacts. |
| Fixed Reynolds number in the search | Re is a flight condition, not a design choice. Search shape and alpha for a stated flight condition. |
| Genetic algorithm | The surrogate is cheap but not differentiable (random forest) and the landscape may have local optima. A GA needs only function values. A random-search baseline with the same budget shows whether the GA is earning its keep. |
