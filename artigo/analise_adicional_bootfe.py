#!/usr/bin/env python3
"""Bootstrap de hospitais (conglomerados) para a diferença de risco por +10 pp nos modelos com efeitos fixos de
hospital: o IC inclui a incerteza dos interceptos. Matriz de desenho montada uma vez; cada réplica reajusta só o IRLS.
Saída: artigo/resultados_adicionais/R10f_FE_diferenca_risco_bootstrap.csv"""
import os
import sys

import numpy as np
import pandas as pd
from scipy.special import expit

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import analise_adicional as ar  # noqa: E402
from analise_v2 import ModeloFE, dr_continua  # noqa: E402

B = int(sys.argv[1]) if len(sys.argv) > 1 else 300
d = ar.preparar()
g = ar.cel(d, ["expo", "x_lag1"])
gi = g[g.expo.notna()].assign(e10=lambda z: z.expo / 10)
linhas = []
rng = np.random.default_rng(1)
for nome, gx in [("Same-year (leave-one-out)", gi), ("Prior-year", gi[gi.x_lag1.notna()].assign(e10=lambda z: z.x_lag1 / 10))]:
    m0 = ModeloFE(f"{ar.FORM} + e10", gx)
    r0 = m0.dr(dr_continua("e10", 1))
    W, k, n, gr = m0.X.values, m0.g.k.values.astype(float), m0.g.n.values.astype(float), m0.grupos
    j = m0.idx("e10")
    G = gr.max() + 1
    linhas_h = [np.where(gr == h)[0] for h in range(G)]
    vals = []
    for b in range(B):
        hs = rng.integers(0, G, G)
        idx = np.concatenate([linhas_h[h] for h in hs])
        grb = np.concatenate([np.full(len(linhas_h[h]), i) for i, h in enumerate(hs)])
        Wb, kb, nb = W[idx], k[idx], n[idx]
        ok = Wb.std(axis=0) > 0
        try:
            bb, ab, _ = ar.fe_irls(Wb[:, ok], kb, nb, grb, G, np.zeros(len(kb)))
        except np.linalg.LinAlgError:
            continue
        beta = np.zeros(W.shape[1])
        beta[ok] = bb
        eta = ab[grb] + Wb @ beta
        w = nb / nb.sum()
        vals.append(100 * np.sum(w * (expit(eta + beta[j]) - expit(eta))))
        if (b + 1) % 50 == 0:
            ar.log(f"{nome}: {b + 1} réplicas")
    vals = np.array(vals)
    linhas.append({"modelo": f"Within hospitals, {nome}", "dr_pp": r0["dr_pp"], "delta_lo": r0["lo"], "delta_hi": r0["hi"],
                   "boot_lo": np.percentile(vals, 2.5), "boot_hi": np.percentile(vals, 97.5), "replicas": len(vals)})
t = pd.DataFrame(linhas).round(4)
t.to_csv(os.path.join(ar.OUT, "R10f_FE_diferenca_risco_bootstrap.csv"), index=False)
print(t.to_string())
