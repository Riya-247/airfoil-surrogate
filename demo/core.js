// Browser port of airfoil_surrogate.geometry + the exported SVR. Verified against Python in test_core.mjs.
function linspace(a, b, n) { const o = new Array(n); for (let i = 0; i < n; i++) o[i] = a + (b - a) * i / (n - 1); return o; }
const THETA = linspace(0, Math.PI, 201);
const XS = THETA.map(t => 0.5 * (1 - Math.cos(t)));

function camber4(m, p, x) {
  const yc = new Array(x.length).fill(0), dyc = new Array(x.length).fill(0);
  if (m === 0) return [yc, dyc];
  if (p === 0) { for (let i = 0; i < x.length; i++) dyc[i] = -2 * m * x[i]; return [yc, dyc]; }
  for (let i = 0; i < x.length; i++) {
    const xi = x[i];
    if (xi < p) { yc[i] = m / (p * p) * (2 * p * xi - xi * xi); dyc[i] = 2 * m / (p * p) * (p - xi); }
    else { const q = (1 - p) * (1 - p); yc[i] = m / q * ((1 - 2 * p) + 2 * p * xi - xi * xi); dyc[i] = 2 * m / q * (p - xi); }
  }
  return [yc, dyc];
}
function thickness(t, x) { return x.map(v => 5 * t * (0.2969 * Math.sqrt(v) - 0.1260 * v - 0.3516 * v ** 2 + 0.2843 * v ** 3 - 0.1036 * v ** 4)); }

// Returns surfaces ordered trailing edge -> leading edge, like the Python code
function naca4Surfaces(M, P, XX, n = 200) {
  const x = linspace(1, 0, n + 1).map(b => (1 - Math.cos(Math.PI * b)) / 2);
  const [yc, dyc] = camber4(M / 100, P / 10, x), yt = thickness(XX / 100, x);
  const xu = [], yu = [], xl = [], yl = [];
  for (let i = 0; i < x.length; i++) {
    const th = Math.atan(dyc[i]), s = Math.sin(th), c = Math.cos(th);
    xu.push(x[i] - yt[i] * s); yu.push(yc[i] + yt[i] * c);
    xl.push(x[i] + yt[i] * s); yl.push(yc[i] - yt[i] * c);
  }
  return { xu, yu, xl, yl };
}

function interp(xq, xs, ys) { // xs ascending, clamped like numpy
  if (xq <= xs[0]) return ys[0];
  if (xq >= xs[xs.length - 1]) return ys[ys.length - 1];
  let lo = 0, hi = xs.length - 1;
  while (hi - lo > 1) { const mid = (lo + hi) >> 1; if (xs[mid] <= xq) lo = mid; else hi = mid; }
  return ys[lo] + (ys[hi] - ys[lo]) * (xq - xs[lo]) / (xs[hi] - xs[lo]);
}
function onGrid(x, y) {
  const idx = x.map((_, i) => i).sort((a, b) => x[a] - x[b]);
  const xs = idx.map(i => x[i]), ys = idx.map(i => y[i]);
  return XS.map(v => interp(v, xs, ys));
}
function gradient(f, x) { // numpy.gradient, non-uniform spacing
  const n = f.length, g = new Array(n);
  g[0] = (f[1] - f[0]) / (x[1] - x[0]);
  g[n - 1] = (f[n - 1] - f[n - 2]) / (x[n - 1] - x[n - 2]);
  for (let i = 1; i < n - 1; i++) {
    const hs = x[i] - x[i - 1], hd = x[i + 1] - x[i];
    g[i] = (hs * hs * f[i + 1] + (hd * hd - hs * hs) * f[i] - hd * hd * f[i - 1]) / (hs * hd * (hd + hs));
  }
  return g;
}
function descriptors(M, P, XX) {
  const s = naca4Surfaces(M, P, XX), yu = onGrid(s.xu, s.yu), yl = onGrid(s.xl, s.yl);
  const thick = yu.map((v, i) => v - yl[i]), cam = yu.map((v, i) => 0.5 * (v + yl[i]));
  let im = 0; for (let i = 1; i < cam.length; i++) if (Math.abs(cam[i]) > Math.abs(cam[im])) im = i;
  const slope = gradient(cam, XS), h = Math.PI / 200;
  let acc = 0;
  for (let i = 0; i < 200; i++) acc += 0.5 * h * (slope[i] * (1 - Math.cos(THETA[i])) + slope[i + 1] * (1 - Math.cos(THETA[i + 1])));
  return { t_max: Math.max(...thick), m_max: cam[im], x_m: Math.abs(cam[im]) > 1e-6 ? XS[im] : 0,
           z75: interp(0.75, XS, cam), alpha_L0_deg: (acc / Math.PI) * 180 / Math.PI };
}
function featureVector(d, alpha, re) {
  return [alpha, Math.log10(re), d.t_max, d.m_max, d.x_m, d.z75, d.alpha_L0_deg, alpha - d.alpha_L0_deg];
}
function makeModel(J) {
  function predict(x) {
    const z = x.map((v, i) => (v - J.xmean[i]) / J.xscale[i]);
    const y = J.targets.map((T, k) => {
      let f = T.b;
      for (let s = 0; s < T.sv.length; s++) {
        let d2 = 0; const sv = T.sv[s];
        for (let j = 0; j < 8; j++) { const e = z[j] - sv[j]; d2 += e * e; }
        f += T.coef[s] * Math.exp(-T.gamma * d2);
      }
      return f * J.yscale[k] + J.ymean[k];
    });
    return { cl: y[0], cd: Math.pow(10, y[1]), z };
  }
  function nnDistance(z) { let best = Infinity; for (const t of J.train_z) { let d2 = 0; for (let j = 0; j < 8; j++) { const e = z[j] - t[j]; d2 += e * e; } if (d2 < best) best = d2; } return Math.sqrt(best); }
  return { predict, nnDistance };
}
if (typeof module !== 'undefined') module.exports = { naca4Surfaces, descriptors, featureVector, makeModel };
