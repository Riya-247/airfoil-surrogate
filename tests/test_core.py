import numpy as np
import pytest

from airfoil_surrogate.data import make_synthetic, parse_simulation_name, split_masks
from airfoil_surrogate.features import feature_matrix, frame_from_naca4, targets
from airfoil_surrogate.geometry import (cst_coefficients, descriptors, naca4_contour,
                                        naca4_surfaces, reynolds_from_velocity, XS)
from airfoil_surrogate.optimise import run_ga


def test_naca0012_basics():
    d = descriptors(*naca4_surfaces(0, 0, 12))
    assert d["t_max"] == pytest.approx(0.12, abs=1e-3)
    assert abs(d["m_max"]) < 1e-9 and abs(d["alpha_L0_deg"]) < 1e-6


def test_zero_lift_angle_naca2412():
    # thin airfoil theory gives about -2 deg for the NACA 2412
    d = descriptors(*naca4_surfaces(2, 4, 12))
    assert -2.4 < d["alpha_L0_deg"] < -1.8
    assert d["m_max"] == pytest.approx(0.02, abs=1.5e-3)
    assert d["x_m"] == pytest.approx(0.4, abs=0.03)


def test_cst_reconstruction():
    xu, yu, xl, yl = naca4_surfaces(4, 4, 12)
    c = cst_coefficients(xu, yu, xl, yl)
    from math import comb
    cls = XS**0.5 * (1 - XS)
    basis = np.stack([comb(5, i) * XS**i * (1 - XS) ** (5 - i) for i in range(6)], axis=1)
    yu_fit = cls * (basis @ np.array([c[f"cst_u{i}"] for i in range(6)]))
    order = np.argsort(xu)
    err = np.abs(yu_fit - np.interp(XS, xu[order], yu[order]))
    # Skip the first 2% of chord: for cambered sections the rotated leading-edge points are not
    # single-valued in x there, so neither the interpolated surface nor a CST fit is meaningful.
    assert err[XS > 0.02].max() < 2e-3


def test_matches_airfrans_generator():
    af_gen = pytest.importorskip("airfrans.naca_generator")
    for params in [(0, 0, 12), (2, 4, 12), (5.3, 3.2, 8.7)]:
        mine = naca4_contour(*params, n=200)
        ref = af_gen.naca_generator(np.array(params), nb_samples=200, verbose=False)
        assert np.max(np.abs(mine - ref)) < 1e-9


def test_name_parsing_and_reynolds():
    u, a, p = parse_simulation_name("airFoil2D_SST_43.597_5.932_3.551_3.1_1.0_18.252")
    assert (u, a, p) == (43.597, 5.932, [3.551, 3.1, 1.0, 18.252])
    assert reynolds_from_velocity(93.6) == pytest.approx(6.0e6, rel=1e-3)


def test_synthetic_table_and_splits():
    df = make_synthetic(200, seed=1)
    X, names = feature_matrix(df, "physics")
    Y = targets(df)
    assert X.shape == (200, len(names)) and Y.shape == (200, 2)
    tr, te = split_masks(df, "full")
    assert tr.sum() + te.sum() == 200 and not (tr & te).any()
    tr, te = split_masks(df, "aoa")
    assert df.loc[tr, "alpha_deg"].between(-2.5, 12.5).all()


def test_frame_from_naca4_matches_table_features():
    df = frame_from_naca4([2.0, 2.0], [4.0, 4.0], [12.0, 12.0], [3.0, 7.0], [4e6, 4e6])
    assert len(df) == 2 and df["alpha_L0_deg"].nunique() == 1
    X, names = feature_matrix(df, "physics")
    assert np.isfinite(X).all()


def test_ga_finds_known_optimum():
    f = lambda x: -np.sum((x - np.array([0.3, -1.0, 2.0])) ** 2, axis=1)
    res = run_ga(f, [(-2, 2)] * 2 + [(0, 3)], pop_size=60, generations=60, seed=0)
    assert res.best_f > -1e-3
    assert res.history["best"][-1] >= res.history["best"][0]
    assert res.n_evals == 60 + 60 * (60 - 2)
