"""NACA 4-digit geometry and family-agnostic geometry descriptors.

Conventions follow the AirfRANS dataset (Bonnet et al., NeurIPS 2022):
  * chord = 1 m
  * 4-digit airfoils are described by (M, P, XX) = (max camber in % chord,
    position of max camber in tenths of chord, max thickness in % chord),
    and these are real numbers, not integers
  * kinematic viscosity was fixed at 1.56e-5 m^2/s, so with a 1 m chord the
    Reynolds number is simply  Re = U / 1.56e-5.
"""
from __future__ import annotations

import numpy as np

NU_AIR = 1.56e-5  # m^2/s, value fixed when the AirfRANS dataset was generated


def reynolds_from_velocity(u):
    return np.asarray(u, dtype=float) / NU_AIR


def velocity_from_reynolds(re):
    return np.asarray(re, dtype=float) * NU_AIR


# --------------------------------------------------------------------------
# NACA 4-digit surfaces
# --------------------------------------------------------------------------
def _thickness(t: float, x: np.ndarray) -> np.ndarray:
    """Half-thickness distribution (closed trailing edge); t is a fraction of chord."""
    return 5 * t * (0.2969 * np.sqrt(x) - 0.1260 * x - 0.3516 * x**2
                    + 0.2843 * x**3 - 0.1036 * x**4)


def _camber4(m: float, p: float, x: np.ndarray):
    """Camber line y_c and slope dy_c/dx. m, p are fractions (M/100, P/10)."""
    yc = np.zeros_like(x)
    dyc = np.zeros_like(x)
    if m == 0.0:
        return yc, dyc
    if p == 0.0:  # degenerate case, mirrors the AirfRANS generator
        return yc, -2 * m * x
    fwd, aft = x < p, x >= p
    yc[fwd] = m / p**2 * (2 * p * x[fwd] - x[fwd] ** 2)
    dyc[fwd] = 2 * m / p**2 * (p - x[fwd])
    yc[aft] = m / (1 - p) ** 2 * ((1 - 2 * p) + 2 * p * x[aft] - x[aft] ** 2)
    dyc[aft] = 2 * m / (1 - p) ** 2 * (p - x[aft])
    return yc, dyc


def naca4_surfaces(m_pct: float, p_tenths: float, t_pct: float, n: int = 200):
    """Upper and lower surface coordinates, each ordered trailing edge -> leading edge.

    Returns (xu, yu, xl, yl).
    """
    beta = np.pi * np.linspace(1.0, 0.0, n + 1)
    x = (1 - np.cos(beta)) / 2
    yc, dyc = _camber4(m_pct / 100.0, p_tenths / 10.0, x)
    yt = _thickness(t_pct / 100.0, x)
    th = np.arctan(dyc)
    xu, yu = x - yt * np.sin(th), yc + yt * np.cos(th)
    xl, yl = x + yt * np.sin(th), yc - yt * np.cos(th)
    return xu, yu, xl, yl


def naca4_contour(m_pct: float, p_tenths: float, t_pct: float, n: int = 200) -> np.ndarray:
    """Closed contour (TE -> upper -> LE -> lower -> TE) as an (N, 2) array, for plotting."""
    xu, yu, xl, yl = naca4_surfaces(m_pct, p_tenths, t_pct, n)
    x = np.concatenate([xu, xl[:-1][::-1]])
    y = np.concatenate([yu, yl[:-1][::-1]])
    pos = np.stack([x, y], axis=-1)
    pos[0], pos[-1] = (1.0, 0.0), (1.0, 0.0)
    return pos


def contour_to_surfaces(contour: np.ndarray):
    """Split a closed contour in the AirfRANS/NACA ordering back into (xu, yu, xl, yl)."""
    n = (len(contour) - 1) // 2 + 1  # n+1 points per surface (LE shared)
    upper = contour[:n]
    lower = contour[n - 1:][:]  # LE -> TE
    return upper[:, 0], upper[:, 1], lower[:, 0], lower[:, 1]


# --------------------------------------------------------------------------
# Family-agnostic descriptors (work for 4- and 5-digit airfoils alike)
# --------------------------------------------------------------------------
_THETA = np.linspace(0.0, np.pi, 201)
XS = 0.5 * (1.0 - np.cos(_THETA))  # common chordwise grid, LE -> TE, cosine-spaced


def _on_grid(x, y):
    order = np.argsort(x)
    return np.interp(XS, x[order], y[order])


def thickness_and_camber(xu, yu, xl, yl):
    yu_g, yl_g = _on_grid(xu, yu), _on_grid(xl, yl)
    return yu_g - yl_g, 0.5 * (yu_g + yl_g), yu_g, yl_g


def zero_lift_angle_deg(camber: np.ndarray) -> float:
    """Thin-airfoil-theory zero-lift angle of attack (degrees) from the camber line.

        alpha_L0 = (1/pi) * Int_0^pi (dz/dx) (1 - cos theta) d theta,   x = (1 - cos theta)/2

    Negative for positively cambered sections (they still lift at alpha = 0).
    """
    slope = np.gradient(camber, XS)
    val = np.trapezoid(slope * (1.0 - np.cos(_THETA)), _THETA) / np.pi
    return float(np.degrees(val))


def descriptors(xu, yu, xl, yl) -> dict:
    """Low-dimensional physical description of a section.

    t_max   maximum thickness / chord
    m_max   maximum camber / chord (signed)
    x_m     chordwise position of maximum camber
    z75     camber at 75% chord (aft-loading indicator)
    alpha_L0_deg  thin-airfoil zero-lift angle (deg)
    """
    thick, camber, _, _ = thickness_and_camber(xu, yu, xl, yl)
    i_m = int(np.argmax(np.abs(camber)))
    return {
        "t_max": float(thick.max()),
        "m_max": float(camber[i_m]),
        "x_m": float(XS[i_m]) if abs(camber[i_m]) > 1e-6 else 0.0,
        "z75": float(np.interp(0.75, XS, camber)),
        "alpha_L0_deg": zero_lift_angle_deg(camber),
    }


def cst_coefficients(xu, yu, xl, yl, order: int = 5) -> dict:
    """Kulfan CST fit (class function x^0.5 (1-x), Bernstein shape function of given order).

    Returns {'cst_u0'.., 'cst_l0'..}. A smooth, fixed-length, family-agnostic
    parameterisation that works for any airfoil coordinate set.
    """
    from math import comb

    _, _, yu_g, yl_g = thickness_and_camber(xu, yu, xl, yl)
    cls = XS**0.5 * (1 - XS)
    basis = np.stack([comb(order, i) * XS**i * (1 - XS) ** (order - i) for i in range(order + 1)], axis=1)
    design = cls[:, None] * basis
    out = {}
    for tag, y in (("u", yu_g), ("l", yl_g)):
        coef, *_ = np.linalg.lstsq(design[1:-1], y[1:-1], rcond=None)
        out.update({f"cst_{tag}{i}": float(c) for i, c in enumerate(coef)})
    return out


def geometry_row(xu, yu, xl, yl) -> dict:
    row = descriptors(xu, yu, xl, yl)
    row.update(cst_coefficients(xu, yu, xl, yl))
    return row
