const J = require('./model.json'), C = require('./core.js'); const m = C.makeModel(J);
let worst = 0;
for (const r of J.ref) {
  const d = C.descriptors(r.M, r.P, r.XX), x = C.featureVector(d, r.a, r.re), p = m.predict(x);
  const dx = Math.max(...x.map((v, i) => Math.abs(v - r.x[i])));
  console.log(r.M, r.P, r.XX, r.a, 'feat maxdiff', dx.toExponential(2), 'cl', p.cl.toFixed(4), r.cl.toFixed(4), 'cd', p.cd.toFixed(5), r.cd.toFixed(5));
  worst = Math.max(worst, Math.abs(p.cl - r.cl), Math.abs(p.cd - r.cd));
}
console.log('worst abs diff', worst); if (worst > 1e-3) process.exit(1);
