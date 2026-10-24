# Dataset facts: AirfRANS

Every fact below was checked against the paper (Bonnet, Mazari, Cinnella, Gallinari, "AirfRANS: High Fidelity
Computational Fluid Dynamics Dataset for Approximating Reynolds-Averaged Navier-Stokes Solutions", NeurIPS 2022
Datasets and Benchmarks Track) or the official `airfrans` library documentation and source, unless marked.

## What it is

| Item | Fact | Source |
|---|---|---|
| Physics | 2D, steady-state, incompressible RANS, subsonic, sea level, 298.15 K | paper, section 3 |
| Solver | OpenFOAM v2112, `simpleFoam` (steady, SIMPLEC) | paper, section 3 |
| Turbulence model | k-omega SST | paper, section 3 |
| Number of simulations | 1,000 | paper, section 3 |
| Reynolds number | 2 to 6 million (inlet velocity roughly 31 to 94 m/s, by `Re = U c / nu` with `c = 1 m`, `nu = 1.56e-5`) | paper; docs; the velocity range is my arithmetic |
| Mach number | Below 0.3. This is a bound that justifies the incompressible assumption, not a sampled variable. | paper |
| Angle of attack | -5 to 15 degrees. The upper limit was chosen to avoid stall and unsteady wakes. | paper, section 3 |
| Geometries | NACA 4-digit (M, P, XX) and 5-digit (L, P, Q, XX) airfoils with real-valued parameters. A NACA 0012 is `(0, 0, 12)`. | paper; docs |
| Mesh | C-grid, 250,000-300,000 cells, first cell 2 micrometres, y+ about 1 in the worst case, far field at 200 chords | paper |
| Cost per simulation | about 25 minutes on 16 cores of an AMD Threadripper 3960X; whole dataset about 20 days | paper, Table 4 |

## What the labels are

AirfRANS stores **fields** per mesh node: velocity (2 components), pressure divided by density, and turbulent
kinematic viscosity (`nu_t`). It does **not** store Cl and Cd as labels. They are computed from the reference
fields by integrating surface pressure and wall shear stress (`Simulation.force_coefficient(reference=True)`
returns `((cd, cd_pressure, cd_viscous), (cl, cl_pressure, cl_viscous))`, per the library source).
The wall shear stress comes from a discrete velocity-gradient filter, so Cd inherits the accuracy of that gradient.
The paper itself reports that drag is much harder than lift for models trained on the fields.

## Official splits (used by `--task`)

- `full`: 800 train / 200 test
- `scarce`: 200 train / same 200 test
- `reynolds`: train on Re 3-5 million, test on 2-3 and 5-6 million
- `aoa`: train on -2.5 to 12.5 degrees, test on -5 to -2.5 and 12.5 to 15 degrees

## What this means for the project, and what you can honestly say

- You have **1,000** RANS solutions, not thousands. Surrogate accuracy claims apply to this size.
- The dataset has essentially **no stalled flow**. Claims about stall behaviour come from extrapolating beyond the
  data and comparing with experiment, not from held-out CFD.
- The coefficient labels are **derived** by surface integration.
- RANS with k-omega SST is a *model* of turbulence. The labels carry model-form error, which is largest in
  separated flow.
- Geometry is restricted to NACA 4- and 5-digit sections. The paper states that models trained on the
  dataset are expected to struggle with more exotic shapes.

## The "9,000 geometries" dataset

Not identified. If it has its own labels, make a CSV with the columns in `airfoil_surrogate/data.py::REQUIRED`
and everything downstream works unchanged. **Say which solver produced those labels**: if they come from
XFOIL (a viscous panel method) they are not RANS, and XFOIL's non-converged post-stall cases are usually dropped
from datasets, which silently removes stall from the labels.

## Alternatives that reach past 15 degrees (not integrated, not downloaded)

- **NREL "Airfoil CFD, 2k shapes, 25 AoA, 3 Re" (HAM2D)**, CC BY 4.0: 1,830 shapes per the description, HAM2D
  finite-volume RANS (Spalart-Allmaras with Medida-Baeder transition), Re 3/6/9 million, Mach 0.1, alpha from -4 to
  20 degrees in 1-degree steps, with flow fields and summary Cl, Cd and a third coefficient. The record gives
  both "over 250,000 runs" and a shape count that implies fewer, so check the actual counts. Landing page:
  https://data.openei.org/submissions/5970. Using it needs an adapter that turns its airfoil descriptions into
  coordinates (so the geometry descriptors can be computed); that adapter is not written.
- **PALMO (NASA Ames)**: OVERFLOW CFD database over camber, thickness, Mach, Reynolds number and angle of attack.
  Angle-of-attack range not checked.
