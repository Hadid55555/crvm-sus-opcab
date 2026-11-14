#!/usr/bin/env python3
"""Análises adicionais post hoc (ver ANALYSIS_PLAN.md).

Itens 1–10. Reaproveita artigo/analise_v2.py (importado como módulo) com a
mesma especificação da análise principal: janela 2003–2025, sem caráter da internação, spline de idade + sexo +
diagnóstico + ano; região e natureza jurídica nos modelos sem efeitos fixos; erros robustos por hospital.
Não altera nenhuma saída existente. Saídas: artigo/resultados_adicionais/*.csv, resultados_adicionais.json e
artigo/tabelas_adicionais.xlsx. Tabelas por hospital são anonimizadas (H01, H02...).

Uso: python artigo/analise_adicional.py [--boot 9999] [--so 1,2,3,...]
"""
import argparse
import json
import os
import sys
import time

import numpy as np
import pandas as pd
import patsy
from scipy import stats
from scipy.special import expit

RAIZ = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, RAIZ)
import analise_v2 as av  # noqa: E402
from analise_v2 import Modelo, ModeloFE, dr_continua, dr_dummy, wald, ols_cluster, demean, wilson  # noqa: E402

# Janela e caráter parametrizáveis (padrão = análise de 2003–2025 sem caráter). Para a especificação pré-registrada:
# REV_INI=2008 REV_CARATER=1 python artigo/analise_adicional.py  → artigo/resultados_adicionais_2008_2025_carater/
INI_REV = int(os.environ.get("REV_INI", "2003"))
CAR_REV = os.environ.get("REV_CARATER", "0") == "1"
SUF = "" if (INI_REV, CAR_REV) == (2003, False) else f"_{INI_REV}_2025" + ("_carater" if CAR_REV else "_sem_carater")
av.INI, av.FIM, av.CARATER = INI_REV, 2025, CAR_REV
if not CAR_REV:
    av.CASEMIX = av.CASEMIX.replace(" + urg", "")
av.MICRO = os.path.join(RAIZ, "..", "saida_crvm", "crvm_2000_2025_internacao_microdados.csv")
CASEMIX, HOSP = av.CASEMIX, av.HOSP
FORM = f"C(ano) + {CASEMIX}"
OUT = os.path.join(RAIZ, "resultados_adicionais" + SUF)
av.OUT = OUT
R = {}
T0 = time.time()
# diferença de risco individual ajustada (pp) da análise principal da mesma especificação (M6)
JAN = f"{INI_REV}–2025"
DR_IND = {(2003, False): -2.36, (2008, True): -2.28, (2008, False): -2.28}[(INI_REV, CAR_REV)]
UM_ENXERTO = {"0406010927", "0406010943"}
CAT4 = [-0.001, 5, 15, 30, 101]
CAT4_LAB = ["0-5%", ">5-15%", ">15-30%", ">30%"]


def log(msg):
    print(f"[{time.time() - T0:7.0f}s] {msg}", flush=True)


def salvar(df, nome):
    df.to_csv(os.path.join(OUT, nome + ".csv"), index=False)
    return df


def anon(ids, prefixo="H"):
    return {c: f"{prefixo}{i + 1:02d}" for i, c in enumerate(ids)}


# ============================================================================ dados
def preparar():
    d = av.carregar()
    R["fluxo"] = av.R["fluxo"]
    d["uf"] = d.uf.astype(str)
    d["graft1"] = d.PROC_REA.isin(UM_ENXERTO).astype(int)
    d["dias"] = d.DIAS_PERM.astype(float)
    d["morte30"] = ((d.MORTE == 1) & (d.dias <= 30)).astype(int)
    d["transf"] = (d.cob == "31").astype(int)
    # hospital-ano: % sem CEC sem leave-one-out, volume, defasagens
    hy = d.groupby(["CNES", "ano"]).agg(n=("off", "size"), k=("off", "sum")).reset_index()
    hy["p"] = 100 * hy.k / hy.n
    full = hy.set_index(["CNES", "ano"])
    def lag(j):
        x = hy.assign(ano=hy.ano + j).set_index(["CNES", "ano"])
        return x.reindex(full.index)
    l1, l2, l3 = lag(1), lag(2), lag(3)
    hy["x_lag1"] = np.where(l1.n.values >= 10, l1.p.values, np.nan)
    nn = np.nansum(np.c_[l1.n.values, l2.n.values, l3.n.values], axis=1)
    kk = np.nansum(np.c_[l1.k.values, l2.k.values, l3.k.values], axis=1)
    hy["x_ma3"] = np.where(nn >= 10, 100 * kk / np.maximum(nn, 1), np.nan)
    hy["n_lag1"] = l1.n.values
    # % encolhida (empírico de Bayes) em direção à média do próprio hospital: λ = τ² / (τ² + p̄(1−p̄)/n)
    N_h = hy.groupby("CNES").n.transform("sum")
    pbar = hy.groupby("CNES").k.transform("sum") / N_h
    q = hy.p / 100
    dev2 = (q - pbar) ** 2
    ruido = pbar * (1 - pbar) * (1 / hy.n - 1 / N_h)
    multi = hy.groupby("CNES").ano.transform("size") >= 2
    tau2 = max(float(np.average(dev2[multi] - ruido[multi], weights=hy.n[multi])), 1e-8)
    lam = tau2 / (tau2 + pbar * (1 - pbar) / hy.n + 1e-12)
    hy["x_eb"] = 100 * (pbar + lam * (q - pbar))
    R["eb_tau2_intra_hospital"] = tau2
    R["eb_lambda_mediana"] = float(lam[multi].median())
    hy = hy.rename(columns={"p": "x_full"})
    d = d.merge(hy[["CNES", "ano", "x_full", "x_lag1", "x_ma3", "x_eb", "n_lag1"]], on=["CNES", "ano"], how="left")
    # maturidade: anos desde a primeira CRVM do hospital na série completa (2000–2025, todas as AIH)
    bruto = pd.read_csv(av.MICRO, usecols=["CNES", "ano_internacao"], dtype={"CNES": str})
    prim = bruto.groupby("CNES").ano_internacao.min()
    d["ano_prim"] = d.CNES.map(prim).fillna(d.groupby("CNES").ano.transform("min"))
    d["anos_prog"] = d.ano - d.ano_prim
    R["hospitais_primeira_crvm_em_2000_censurados"] = int((d.groupby("CNES").ano_prim.first() <= 2000).sum())
    d["logvol"] = np.log(d.hy_n)
    d["vol_cat"] = pd.cut(d.hy_n, [0, 49, 99, 199, 399, 1e9], labels=["<50", "50-99", "100-199", "200-399", "≥400"]).astype(str)
    d["cat_full"] = pd.cut(d.x_full, CAT4, labels=CAT4_LAB).astype(str)
    d["cat_lag"] = pd.cut(d.x_lag1, CAT4, labels=CAT4_LAB).astype(str)
    hoff = d.groupby(["CNES", "ano"]).off.transform("sum")
    d["offvol_cat"] = pd.cut(hoff, [0, 24, 99, 1e9], labels=["<25", "25-99", "≥100"]).astype(str)
    h = d.groupby("CNES").agg(n=("off", "size"), n_off=("off", "sum"))
    h["pct"] = 100 * h.n_off / h.n
    h["cat_hosp"] = pd.cut(h.pct, [-1, 0, 4.999999, 19.999999, 101],
                           labels=["None (0%)", "Low (>0–<5%)", "Intermediate (5–<20%)", "High (≥20%)"]).astype(str)
    d = d.join(h.cat_hosp, on="CNES")
    d["cronico"] = d.diag.isin(["I25 chronic IHD", "Other angina (I20.1-9)"]).astype(int)
    return d


BASE = ["CNES", "ano", "idade", "fem", "diag", "reg", "natj", "uf", "off"] + (["urg"] if CAR_REV else [])


def cel(d, extra=(), y="MORTE"):
    chaves = BASE + [c for c in extra if c not in BASE]
    return d.groupby(chaves, observed=True, dropna=False).agg(k=(y, "sum"), n=(y, "size")).reset_index()


def lin(nome, m, termo, dr=None, **kw):
    o = m.orci(termo)
    r = {"modelo": nome, "termo": termo, "or": o["or"], "lo": o["lo"], "hi": o["hi"], "p": o["p"],
         "aih": m.n_aih, "hospitais": m.n_hosp, "eventos": int(m.g.k.sum())}
    if dr is not None:
        r.update({"dr_pp": dr["dr_pp"], "dr_lo": dr["lo"], "dr_hi": dr["hi"]})
    r.update(kw)
    return r


# ============================================================================ bootstrap para efeitos fixos
def fe_irls(W, k, n, gr, G, off, b=None, maxit=80, tol=1e-11):
    kh, nh = np.bincount(gr, k, G), np.bincount(gr, n, G)
    a = np.log((kh + 0.5) / (nh - kh + 0.5))
    b = np.zeros(W.shape[1]) if b is None else b.copy()
    ll_old = -np.inf
    for _ in range(maxit):
        eta = a[gr] + W @ b + off
        p = expit(eta)
        w = np.maximum(n * p * (1 - p), 1e-12)
        z = eta - off + (k - n * p) / w
        ah = np.maximum(np.bincount(gr, w, G), 1e-12)
        ch = np.zeros((G, W.shape[1]))
        np.add.at(ch, gr, W * w[:, None])
        zh = np.bincount(gr, w * z, G)
        Hs = (W * w[:, None]).T @ W - (ch / ah[:, None]).T @ ch
        rs = (W * w[:, None]).T @ z - (ch / ah[:, None]).T @ zh
        b = np.linalg.solve(Hs, rs)
        a = (zh - ch @ b) / ah
        p = np.clip(expit(a[gr] + W @ b + off), 1e-15, 1 - 1e-15)
        ll = np.sum(k * np.log(p) + (n - k) * np.log(1 - p))
        if abs(ll - ll_old) < tol * (1 + abs(ll)):
            break
        ll_old = ll
    return b, a, p


def score_boot_fe(m, nome, b0=0.0, B=9999, seed=1):
    """Wild cluster score bootstrap (Kline & Santos 2012) no modelo com efeitos fixos de hospital: nula imposta,
    pesos Rademacher; o escore de cada hospital é projetado fora dos demais coeficientes e dos interceptos."""
    j = m.idx(nome)
    W = m.X.values
    Wr = np.delete(W, j, axis=1)
    k, n, gr = m.g.k.values.astype(float), m.g.n.values.astype(float), m.grupos
    G = gr.max() + 1
    _, _, p = fe_irls(Wr, k, n, gr, G, b0 * W[:, j], np.delete(m.b, j))
    resid = k - n * p
    S = np.zeros((G, W.shape[1]))
    np.add.at(S, gr, W * resid[:, None])
    w = n * p * (1 - p)
    ah = np.bincount(gr, w, G)
    ch = np.zeros((G, W.shape[1]))
    np.add.at(ch, gr, W * w[:, None])
    Hs = (W * w[:, None]).T @ W - (ch / ah[:, None]).T @ ch
    o = [i for i in range(W.shape[1]) if i != j]
    st = S[:, j] - S[:, o] @ np.linalg.solve(Hs[np.ix_(o, o)], Hs[o, j])
    T = st.sum() ** 2 / np.sum(st ** 2)
    rng = np.random.default_rng(seed)
    Tb = (rng.choice([-1.0, 1.0], size=(B, G)) @ st) ** 2 / np.sum(st ** 2)
    return float((1 + np.sum(Tb >= T)) / (B + 1))


def boot_ic_fe(m, nome, B=9999, alfa=0.05):
    j = m.idx(nome)
    b, se = m.b[j], np.sqrt(m.V[j, j])
    lim = []
    for lado in (-1, 1):
        dentro, fora = b, b + lado * 2.5 * se
        it = 0
        while score_boot_fe(m, nome, fora, B) >= alfa and it < 6:
            fora += lado * 1.5 * se
            it += 1
        for _ in range(12):
            meio = (dentro + fora) / 2
            if score_boot_fe(m, nome, meio, B) >= alfa:
                dentro = meio
            else:
                fora = meio
        lim.append((dentro + fora) / 2)
    return {"p_boot": score_boot_fe(m, nome, 0.0, B), "or_lo_boot": float(np.exp(lim[0])), "or_hi_boot": float(np.exp(lim[1]))}


# ============================================================================ 1. exposição predeterminada e erro de medida
def confiabilidade(d):
    hy = d.groupby(["CNES", "ano"]).agg(n=("off", "size"), k=("off", "sum")).reset_index()
    linhas = []
    for minimo in (2, 10, 30, 50):
        x = hy[hy.n >= minimo].copy()
        x = x[x.groupby("CNES").ano.transform("size") >= 2]
        N_h = x.groupby("CNES").n.transform("sum")
        pbar = x.groupby("CNES").k.transform("sum") / N_h
        q = x.k / x.n
        obs = np.average((q - pbar) ** 2, weights=x.n)
        ruido = np.average(pbar * (1 - pbar) * (1 / x.n - 1 / N_h), weights=x.n)
        obs_u = np.mean((q - x.groupby("CNES").apply(lambda z: (z.k / z.n).mean(), include_groups=False)
                         .reindex(x.CNES).values) ** 2)
        ruido_u = np.mean(pbar * (1 - pbar) * (1 / x.n) * (1 - 1 / x.groupby("CNES").ano.transform("size")))
        linhas.append({"hospital_anos_min_crvm": minimo, "hospital_anos": len(x), "hospitais": x.CNES.nunique(),
                       "var_intra_observada_pp2": 1e4 * obs, "var_binomial_esperada_pp2": 1e4 * ruido,
                       "confiabilidade_ponderada": 1 - ruido / obs,
                       "confiabilidade_nao_ponderada": 1 - ruido_u / obs_u})
    t = salvar(pd.DataFrame(linhas).round(4), "R1a_confiabilidade_pct_anual")
    R["confiabilidade"] = t.to_dict("records")


def expo_predeterminada(d, B):
    exps = [("x_loo", "Same-year leave-one-out % (original)"), ("x_full", "Same-year % including index patient (no LOO)"),
            ("x_lag1", "Prior-year % (predetermined; prior year ≥10 CABG)"),
            ("x_ma3", "Trailing 3-year % (t−1 to t−3; predetermined)"),
            ("x_eb", "Same-year % shrunk to hospital mean (empirical Bayes)")]
    d = d.assign(x_loo=d.expo)
    cols = [c for c, _ in exps]
    g = cel(d, cols + ["hy_n"])
    comum = g.x_loo.notna() & g.x_lag1.notna()
    linhas = []
    amostras = [("Own sample", None), ("Common sample (prior-year % available)", comum)]
    for nome_am, msk in amostras:
        for c, rot in exps:
            gx = g[g[c].notna() & (msk if msk is not None else True)].assign(e10=lambda z: z[c] / 10)
            if msk is None and c == "x_loo":
                continue  # já está em M4 (resultados_v2_2003_2025_sem_carater); repetido na amostra comum
            for fe in (False, True):
                m = (ModeloFE(f"{FORM} + e10", gx) if fe else Modelo(f"{FORM} + {HOSP} + e10", gx))
                r = lin(rot, m, "e10", m.dr(dr_continua("e10", 1)), amostra=nome_am, efeitos_fixos_hospital=fe)
                if fe and (msk is not None) and c in ("x_loo", "x_lag1", "x_ma3"):
                    r.update(boot_ic_fe(m, "e10", B))
                linhas.append(r)
                log(f"1: {nome_am} | {c} | FE={fe}")
    # restrição por volume do hospital-ano (exposição original e defasada)
    for minimo in (30, 50):
        for c, rot in [("x_loo", "Same-year leave-one-out %"), ("x_lag1", "Prior-year %")]:
            gx = g[(g.hy_n >= minimo) & g[c].notna()].assign(e10=lambda z: z[c] / 10)
            for fe in (False, True):
                m = (ModeloFE(f"{FORM} + e10", gx) if fe else Modelo(f"{FORM} + {HOSP} + e10", gx))
                linhas.append(lin(f"{rot}, hospital-years ≥{minimo} CABG", m, "e10", m.dr(dr_continua("e10", 1)),
                                  amostra=f"hospital-years ≥{minimo}", efeitos_fixos_hospital=fe))
            log(f"1: ≥{minimo} {c}")
    # bootstrap no modelo com efeitos fixos original (amostra completa)
    gx = g[g.x_loo.notna()].assign(e10=lambda z: z.x_loo / 10)
    m = ModeloFE(f"{FORM} + e10", gx)
    linhas.append(lin("Same-year leave-one-out % (original M4 FE model)", m, "e10", m.dr(dr_continua("e10", 1)),
                      amostra="Own sample", efeitos_fixos_hospital=True, **boot_ic_fe(m, "e10", B)))
    t = pd.DataFrame(linhas)
    t["esperado_dr_pp_se_OR_individual_causal"] = DR_IND / 10
    salvar(t.round(4), "R1b_exposicao_predeterminada")
    # hospital-anos por categoria no modelo com efeitos fixos (categorias da exposição original)
    hc = d[d.expo.notna()].groupby("expo_cat").agg(aih=("off", "size"), hospital_anos=("ano", lambda s: 0))
    hyc = d[d.expo.notna()].assign(cf=pd.cut(d.x_full, av.EXPO_CAT, labels=av.EXPO_LAB).astype(str))
    hyc = hyc.drop_duplicates(["CNES", "ano"]).groupby("cf").agg(hospital_anos=("ano", "size"), hospitais=("CNES", "nunique"))
    hc = hc.drop(columns="hospital_anos").join(hyc).reindex(av.EXPO_LAB).reset_index()
    # transições dentro do hospital entre categorias (hospital-anos consecutivos)
    x = d.drop_duplicates(["CNES", "ano"])[["CNES", "ano", "x_full"]].sort_values(["CNES", "ano"])
    x["cat"] = pd.cut(x.x_full, av.EXPO_CAT, labels=av.EXPO_LAB).astype(str)
    x["cat_ant"] = x.groupby("CNES").cat.shift()
    x["ano_ant"] = x.groupby("CNES").ano.shift()
    tr = pd.crosstab(x[x.ano - x.ano_ant == 1].cat_ant, x[x.ano - x.ano_ant == 1].cat).reindex(index=av.EXPO_LAB, columns=av.EXPO_LAB)
    salvar(hc, "R1c_hospital_anos_por_categoria")
    salvar(tr.reset_index().rename(columns={"cat_ant": "categoria_ano_anterior"}), "R1d_transicoes_categoria")


# ============================================================================ 2. estudo de eventos e hospitais identificadores
def eventos(d):
    hy = d.groupby(["CNES", "ano"]).agg(n=("off", "size"), k=("off", "sum")).reset_index()
    hy["p"] = 100 * hy.k / hy.n
    ev = []
    for cnes, x in hy.groupby("CNES"):
        x = x.set_index("ano")
        melhor = None
        for t in x.index:
            anos = [t - 2, t - 1, t, t + 1]
            if not all(a in x.index and x.loc[a, "n"] >= 10 for a in anos):
                continue
            pre = 100 * x.loc[[t - 2, t - 1], "k"].sum() / x.loc[[t - 2, t - 1], "n"].sum()
            d0, d1 = x.loc[t, "p"] - pre, x.loc[t + 1, "p"] - pre
            if abs(d0) >= 10 and abs(d1) >= 10 and np.sign(d0) == np.sign(d1):
                post = 100 * x.loc[[t, t + 1], "k"].sum() / x.loc[[t, t + 1], "n"].sum()
                if melhor is None or abs(post - pre) > abs(melhor["delta_pp"]):
                    melhor = {"CNES": cnes, "ano_evento": t, "pct_pre": pre, "pct_pos": post, "delta_pp": post - pre}
        if melhor:
            ev.append(melhor)
    ev = pd.DataFrame(ev)
    R["eventos_n"] = int(len(ev))
    R["eventos_aumento_queda"] = [int((ev.delta_pp > 0).sum()), int((ev.delta_pp < 0).sum())]
    # LPM com efeitos fixos de hospital e ano; regressores = dummies de tempo relativo × Δ/10 (sinal incluído)
    x = d.merge(ev[["CNES", "ano_evento", "delta_pp"]], on="CNES", how="left")
    x["rel"] = (x.ano - x.ano_evento).clip(-5, 5)
    x = x.reset_index(drop=True)
    gr = pd.factorize(x.CNES)[0]
    Wdf = patsy.dmatrix(FORM, x, return_type="dataframe").drop(columns="Intercept")
    tempos = [-5, -4, -3, -2, 0, 1, 2, 3, 4, 5]
    linhas = []
    for nome, sel, escala in [("Signed (per +10 pp change)", None, True), ("Increases only (per event)", 1, False),
                              ("Decreases only (per event)", -1, False)]:
        msk = np.ones(len(x), bool) if sel is None else (x.delta_pp.isna() | (np.sign(x.delta_pp) == sel)).values
        E = np.column_stack([((x.rel == r) & x.delta_pp.notna()).astype(float) *
                             (x.delta_pp.fillna(0) / 10 if escala else 1.0) for r in tempos])
        Xm = np.column_stack([E, Wdf.values])[msk]
        y = x.MORTE.values.astype(float)[msk]
        g2 = pd.factorize(gr[msk])[0]
        Xd = demean(Xm, g2)
        keep = Xd.std(axis=0) > 1e-12
        keep[:len(tempos)] = True
        yd = demean(y[:, None], g2)[:, 0]
        b, V = ols_cluster(yd, Xd[:, keep], g2)
        for i, r in enumerate(tempos):
            linhas.append({"versao": nome, "tempo_relativo": r, "coef_pp": 100 * b[i],
                           "lo": 100 * (b[i] - 1.96 * np.sqrt(V[i, i])), "hi": 100 * (b[i] + 1.96 * np.sqrt(V[i, i]))})
        pre = [0, 1, 2]  # −5..−3 e −2 vs −1 de referência: testar −4, −3, −2 (−5 é bin aberto)
        jj = [tempos.index(r) for r in (-4, -3, -2)]
        q = float(b[jj] @ np.linalg.solve(V[np.ix_(jj, jj)], b[jj]))
        jp = [tempos.index(r) for r in (0, 1, 2, 3)]
        cvec = np.zeros(len(b))
        cvec[jp] = 0.25
        med, se = float(cvec @ b), float(np.sqrt(cvec @ V @ cvec))
        R[f"evento_{nome}"] = {"pretendencia_chi2": q, "gl": 3, "p_pretendencia": float(stats.chi2.sf(q, 3)),
                               "media_pos_0a3_pp": 100 * med, "lo": 100 * (med - 1.96 * se), "hi": 100 * (med + 1.96 * se),
                               "hospitais_evento": int(ev.CNES.nunique() if sel is None else (np.sign(ev.delta_pp) == sel).sum()),
                               "aih": int(msk.sum())}
        if not escala:
            dm = ev.delta_pp[np.sign(ev.delta_pp) == sel].mean()
            R[f"evento_{nome}"]["delta_medio_pp"] = float(dm)
            R[f"evento_{nome}"]["esperado_pp_se_individual_causal"] = float(dm / 100 * DR_IND)
        else:
            R[f"evento_{nome}"]["esperado_pp_se_individual_causal"] = DR_IND / 10
        log(f"2: evento {nome}")
    salvar(pd.DataFrame(linhas).round(4), "R2a_estudo_eventos")
    return ev


def identificadores(d, ev):
    p = av.pesos_identificacao(d)
    neff = 1 / (p ** 2).sum()
    top = p.index[:int(round(neff))]
    hy = d.groupby(["CNES", "ano"]).agg(n=("off", "size"), k=("off", "sum")).reset_index()
    hy["p"] = 100 * hy.k / hy.n
    attr = d.groupby("CNES").agg(regiao=("reg", "first"), natureza=("natj", "first"), crvm_total=("off", "size"),
                                 pct_off_total=("off", "mean"), anos_ativos=("ano", "nunique"),
                                 primeiro_ano=("ano", "min"), ultimo_ano=("ano", "max"))
    attr["pct_off_total"] *= 100
    attr["crvm_por_ano"] = attr.crvm_total / attr.anos_ativos
    linhas = []
    mapa = anon(top)
    for c in top:
        x = hy[(hy.CNES == c) & (hy.n >= 10)]
        incl = np.polyfit(x.ano, x.p, 1, w=np.sqrt(x.n))[0] if len(x) >= 3 else np.nan
        sentido = ("increase" if incl * len(x) >= 10 else "decrease" if incl * len(x) <= -10 else "fluctuating/no trend")
        e = ev[ev.CNES == c]
        linhas.append({"hospital": mapa[c], "peso_identificacao_pct": 100 * p[c], **attr.loc[c].to_dict(),
                       "pct_min_ano(≥10)": x.p.min(), "pct_max_ano(≥10)": x.p.max(),
                       "inclinacao_pp_por_ano": incl, "sentido": sentido,
                       "evento_sustentado_>=10pp": "yes" if len(e) else "no",
                       "delta_evento_pp": float(e.delta_pp.iloc[0]) if len(e) else np.nan})
    t = pd.DataFrame(linhas)
    salvar(t.round(2), "R2b_hospitais_identificadores_anon")
    # resumo: identificadores × demais
    attr["grupo"] = np.where(attr.index.isin(top), f"Top {len(top)} by identification weight", "Other hospitals")
    res = []
    for gname, x in attr.groupby("grupo"):
        r = {"grupo": gname, "hospitais": len(x), "crvm_por_ano_mediana": x.crvm_por_ano.median(),
             "pct_off_total_mediana": x.pct_off_total.median(), "peso_identificacao_pct": 100 * p[p.index.isin(x.index)].sum()}
        for reg in ["North", "Northeast", "Southeast", "South", "Center-West"]:
            r[f"pct_{reg}"] = 100 * (x.regiao == reg).mean()
        for nj in ["Public", "Nonprofit/philanthropic", "Private for-profit", "Unknown"]:
            r[f"pct_{nj}"] = 100 * (x.natureza == nj).mean()
        res.append(r)
    salvar(pd.DataFrame(res).round(2), "R2c_identificadores_resumo")
    R["identificadores_n"] = len(top)
    R["identificadores_sentido"] = t.sentido.value_counts().to_dict()
    ea = ev.assign(hospital=ev.CNES.map(anon(ev.sort_values("delta_pp").CNES, "E")))
    ea = ea.join(attr[["regiao", "natureza", "crvm_por_ano"]], on="CNES")
    ea["entre_identificadores"] = ea.CNES.isin(top)
    salvar(ea.drop(columns="CNES").round(2), "R2d_eventos_anon")


# ============================================================================ 3. volume, maturidade e teste de seleção
def volume_maturidade(d):
    g = cel(d, ["expo", "hy_n", "logvol", "anos_prog", "vol_cat"])
    gi = g[g.expo.notna()].assign(e10=lambda z: z.expo / 10, algum=lambda z: (z.expo > 0).astype(int))
    VOL, MAT = "cr(logvol, df=4, constraints='center')", "cr(anos_prog, df=3, constraints='center')"
    HUF = "C(natj, Treatment('Nonprofit/philanthropic')) + C(uf)"
    linhas = []
    for nome, form in [("Primary model (reference)", f"{FORM} + {HOSP} + e10"),
                       ("+ annual hospital CABG volume (spline)", f"{FORM} + {HOSP} + {VOL} + e10"),
                       ("+ years since first CABG (spline)", f"{FORM} + {HOSP} + {MAT} + e10"),
                       ("+ volume + years since first CABG", f"{FORM} + {HOSP} + {VOL} + {MAT} + e10"),
                       ("+ volume + years since first CABG, state (27) instead of region", f"{FORM} + {HUF} + {VOL} + {MAT} + e10")]:
        m = Modelo(form, gi)
        linhas.append(lin(nome, m, "e10", m.dr(dr_continua("e10", 1)), efeitos_fixos_hospital=False))
        log(f"3: {nome}")
    m = ModeloFE(f"{FORM} + {VOL} + e10", gi)
    linhas.append(lin("Hospital FE + annual volume (spline)", m, "e10", m.dr(dr_continua("e10", 1)), efeitos_fixos_hospital=True))
    # 0% × algum sem CEC, separado da dose entre usuários
    for nome, form in [("Any vs none + dose among users", f"{FORM} + {HOSP} + algum + e10"),
                       ("Any vs none + dose, + volume + years since first CABG", f"{FORM} + {HOSP} + {VOL} + {MAT} + algum + e10")]:
        m = Modelo(form, gi)
        for t in ("algum", "e10"):
            linhas.append(lin(nome, m, t, efeitos_fixos_hospital=False))
    # linearidade: categorias acrescentadas ao termo linear
    cats = [f"C(expo_cat, Treatment('0%'))[T.{c}]" for c in av.EXPO_LAB[1:]]
    gc = gi.assign(expo_cat=pd.cut(gi.expo, av.EXPO_CAT, labels=av.EXPO_LAB).astype(str))
    mc = Modelo(f"{FORM} + {HOSP} + e10 + C(expo_cat, Treatment('0%'))", gc)
    R["institucional_teste_linearidade_wald"] = wald(mc, cats[1:] if False else cats)
    mc0 = Modelo(f"{FORM} + {HOSP} + C(expo_cat, Treatment('0%'))", gc)
    R["institucional_categorias_wald"] = wald(mc0, cats)
    mcv = Modelo(f"{FORM} + {HOSP} + {VOL} + {MAT} + C(expo_cat, Treatment('0%'))", gc)
    for c in cats:
        linhas.append(lin("Categories vs 0%, + volume + years since first CABG", mcv, c, efeitos_fixos_hospital=False))
    R["institucional_categorias_wald_ajust_volume"] = wald(mcv, cats)
    salvar(pd.DataFrame(linhas).round(4), "R3a_institucional_volume_maturidade")
    # volume × mortalidade (global e por técnica)
    vl = []
    for tec, gx in [("Overall", g), ("On-pump", g[g.off == 0]), ("Off-pump", g[g.off == 1])]:
        m = Modelo(f"{FORM} + {HOSP} + C(vol_cat, Treatment('≥400'))", gx)
        nomes = [c for c in m.cols if c.startswith("C(vol_cat")]
        for c in nomes:
            lab = c.split("[T.")[1][:-1]
            vl.append(lin(tec, m, c, m.dr(dr_dummy(c, nomes)), categoria_volume=lab,
                          aih_categoria=int(gx.loc[gx.vol_cat == lab, "n"].sum())))
        vl.append({"modelo": tec, "categoria_volume": "≥400 (reference)", "aih_categoria": int(gx.loc[gx.vol_cat == "≥400", "n"].sum())})
        R[f"volume_wald_{tec}"] = wald(m, nomes)
        ms = Modelo(f"{FORM} + {HOSP} + logvol", gx)
        R[f"volume_or_por_duplicacao_{tec}"] = {k: (v if k in ("p",) else float(np.exp(np.log(v) * np.log(2))) if k in ("or", "lo", "hi") else v)
                                                for k, v in ms.orci("logvol").items() if k in ("or", "lo", "hi", "p")}
    salvar(pd.DataFrame(vl).round(4), "R3b_volume_mortalidade")
    log("3: volume × mortalidade")


def selecao(d):
    linhas = []
    for var, rotulo, niveis in [("cat_full", "Hospital-year OPCAB % (same year, incl. patient)", CAT4_LAB),
                                ("cat_lag", "Hospital OPCAB % in prior year", CAT4_LAB),
                                ("cat_hosp", f"Hospital OPCAB % over {JAN}", ["Low (>0–<5%)", "Intermediate (5–<20%)", "High (≥20%)"]),
                                ("offvol_cat", "Hospital-year OPCAB volume", ["<25", "25-99", "≥100"])]:
        g = cel(d, [var])
        g = g[g[var].isin(niveis)]
        for nv in niveis:
            gx = g[g[var] == nv]
            if gx.k.sum() == 0 or gx.loc[gx.off == 1, "n"].sum() == 0:
                continue
            m = Modelo(f"{FORM} + {HOSP} + off", gx)
            r = lin(rotulo, m, "off", m.dr(dr_dummy("off", ["off"])), estrato=nv, efeitos_fixos_hospital=False,
                    aih_off=int(gx.loc[gx.off == 1, "n"].sum()), pct_off=100 * gx.loc[gx.off == 1, "n"].sum() / gx.n.sum())
            linhas.append(r)
            try:
                mf = ModeloFE(f"{FORM} + off", gx)
                linhas.append(lin(rotulo, mf, "off", estrato=nv, efeitos_fixos_hospital=True))
            except np.linalg.LinAlgError:
                pass
        mi = Modelo(f"{FORM} + {HOSP} + C({var}) * off", g)
        R[f"selecao_interacao_{var}"] = wald(mi, [c for c in mi.cols if ":off" in c or "off:" in c])
        log(f"3: seleção {var}")
    salvar(pd.DataFrame(linhas).round(4), "R3c_OR_individual_por_categoria_hospital")


# ============================================================================ 4. enxertos (2008–2025, descritivo/estratificação)
def enxertos(d):
    x = d[d.ano >= 2008]
    tec = np.where(x.off == 1, "Off-pump", "On-pump")
    a = x.groupby([x.ano, tec]).graft1.agg(["size", "sum", "mean"]).reset_index()
    a.columns = ["ano", "tecnica", "n", "n_1enxerto", "pct_1enxerto"]
    a["pct_1enxerto"] *= 100
    salvar(a.round(2), "R4a_enxertos_por_ano_tecnica")
    tabs = []
    for var, rot in [("cat_full", "Hospital-year OPCAB %"), ("cat_hosp", f"Hospital OPCAB % {JAN}")]:
        t = x.groupby([var, tec]).graft1.agg(["size", "mean"]).reset_index()
        t.columns = ["categoria", "tecnica", "n", "pct_1enxerto"]
        t["pct_1enxerto"] *= 100
        tabs.append(t.assign(variavel=rot))
    salvar(pd.concat(tabs)[["variavel", "categoria", "tecnica", "n", "pct_1enxerto"]].round(2), "R4b_enxertos_por_categoria_hospital")
    linhas = []
    for amostra, xx in [("All hospitals, 2008–2025", x), ("Stable graft-coding hospitals (130), 2008–2025", x[x.cod_estavel == 1])]:
        g = cel(xx, ["graft1"])
        for nome, gx in [("All codes (2008–2025)", g), ("1-graft codes", g[g.graft1 == 1]), ("≥2-graft codes", g[g.graft1 == 0])]:
            m = Modelo(f"{FORM} + {HOSP} + off", gx)
            linhas.append(lin(nome, m, "off", m.dr(dr_dummy("off", ["off"])), amostra=amostra, efeitos_fixos_hospital=False,
                              mort_on=100 * gx.loc[gx.off == 0, "k"].sum() / gx.loc[gx.off == 0, "n"].sum(),
                              mort_off=100 * gx.loc[gx.off == 1, "k"].sum() / gx.loc[gx.off == 1, "n"].sum()))
            mf = ModeloFE(f"{FORM} + off", gx)
            linhas.append(lin(nome, mf, "off", amostra=amostra, efeitos_fixos_hospital=True))
        mi = Modelo(f"{FORM} + {HOSP} + graft1 * off", g)
        R[f"enxerto_interacao_{amostra}"] = wald(mi, ["graft1:off"])
        log(f"4: {amostra}")
    # restrição a diagnósticos crônicos (substituto imperfeito de cirurgia eletiva), janela principal
    g = cel(d[d.cronico == 1])
    m = Modelo(f"{FORM} + {HOSP} + off", g)
    linhas.append(lin(f"Chronic presentations only (I25, I20.1–I20.9), {JAN}", m, "off", m.dr(dr_dummy("off", ["off"])),
                      amostra=f"All hospitals, {JAN}", efeitos_fixos_hospital=False))
    mf = ModeloFE(f"{FORM} + off", g)
    linhas.append(lin(f"Chronic presentations only (I25, I20.1–I20.9), {JAN}", mf, "off", amostra=f"All hospitals, {JAN}",
                      efeitos_fixos_hospital=True))
    salvar(pd.DataFrame(linhas).round(4), "R4c_OR_individual_por_enxertos")


# ============================================================================ 5. desfecho
def desfecho(d):
    ob = d[d.MORTE == 1]
    linhas = []
    for nome, x in [("On-pump", ob[ob.off == 0]), ("Off-pump", ob[ob.off == 1]), ("Overall", ob)]:
        for per, xx in [("2003–2025", x), ("2003–2007", x[x.ano <= 2007]), ("2008–2025", x[x.ano >= 2008])]:
            s = xx.dias
            linhas.append({"tecnica": nome, "periodo": per, "obitos": len(s), "dias_mediana": s.median(),
                           "p25": s.quantile(.25), "p75": s.quantile(.75), "pct_dia_0_1": 100 * (s <= 1).mean(),
                           "pct_<=7": 100 * (s <= 7).mean(), "pct_<=30": 100 * (s <= 30).mean(), "pct_>30": 100 * (s > 30).mean()})
    salvar(pd.DataFrame(linhas).round(2), "R5a_momento_obito")
    R["morte30_n"] = int(d.morte30.sum())
    # dias de permanência × datas
    dt = (pd.to_datetime(d.DT_SAIDA, format="%Y%m%d", errors="coerce") - pd.to_datetime(d.DT_INTER, format="%Y%m%d", errors="coerce")).dt.days
    R["dias_perm_igual_datas_pct"] = float(100 * (dt == d.dias).mean())
    # modelos com desfechos alternativos
    lm = []
    for nome, y, xx in [(f"In-hospital death (primary), {JAN}", "MORTE", d),
                        (f"Death ≤30 days from admission during index stay, {JAN}", "morte30", d),
                        ("In-hospital death, 2008–2025", "MORTE", d[d.ano >= 2008]),
                        ("Death or transfer out (COBRANCA 31), 2008–2025", "morte_transf", d[d.ano >= 2008])]:
        g = cel(xx, ["expo"], y=y)
        m = Modelo(f"{FORM} + {HOSP} + off", g)
        lm.append(lin(nome, m, "off", m.dr(dr_dummy("off", ["off"])), estimando="Patient level", efeitos_fixos_hospital=False))
        gi = g[g.expo.notna()].assign(e10=lambda z: z.expo / 10)
        m = Modelo(f"{FORM} + {HOSP} + e10", gi)
        lm.append(lin(nome, m, "e10", m.dr(dr_continua("e10", 1)), estimando="Hospital strategy (+10 pp)", efeitos_fixos_hospital=False))
        m = ModeloFE(f"{FORM} + e10", gi)
        lm.append(lin(nome, m, "e10", m.dr(dr_continua("e10", 1)), estimando="Hospital strategy (+10 pp)", efeitos_fixos_hospital=True))
        log(f"5: {nome}")
    salvar(pd.DataFrame(lm).round(4), "R5b_desfechos_alternativos")
    # transferências por técnica e categoria (2008–2025)
    x = d[d.ano >= 2008]
    tr = []
    for var, rot in [(np.where(x.off == 1, "Off-pump", "On-pump"), "Technique"), (x.cat_full.values, "Hospital-year OPCAB %"),
                     (x.cat_hosp.values, f"Hospital OPCAB % {JAN}")]:
        t = x.groupby(var).agg(n=("transf", "size"), transf=("transf", "sum"), obitos=("MORTE", "sum")).reset_index()
        t.columns = ["grupo", "n", "transferencias", "obitos"]
        t["pct_transf"] = 100 * t.transferencias / t.n
        t["pct_obito"] = 100 * t.obitos / t.n
        tr.append(t.assign(variavel=rot))
    salvar(pd.concat(tr)[["variavel", "grupo", "n", "transferencias", "pct_transf", "obitos", "pct_obito"]].round(3), "R5c_transferencias")
    # concordância MORTE × motivo de saída
    cobg = np.select([d.cob.isin(["41", "42", "43"]), d.cob.isin(["51", "52", "53"]), d.cob == "31"],
                     ["41–43 (death, SIGTAP coding)", "51–53 (death codes, pre-2008 coding)", "31 (transfer)"], "other")
    per = np.where(d.ano >= 2008, "2008–2025", "2003–2007")
    c = pd.crosstab([per, cobg], d.MORTE).reset_index()
    c.columns = ["periodo", "motivo_saida", "MORTE_0", "MORTE_1"]
    salvar(c, "R5d_concordancia_MORTE_motivo_saida")


# ============================================================================ 6. codificação e pagamento
def codificacao(d):
    x = d[d.ano >= 2020].copy()
    x["mes"] = x.DT_INTER.str[:6]
    m = x.groupby("mes").agg(n=("off", "size"), pct_off=("off", "mean"), pct_complemento=("fed", lambda s: (s > 0).mean()))
    on = x[x.off == 0].groupby("mes").graft1.mean().rename("pct_1enxerto_on")
    of = x[x.off == 1].groupby("mes").graft1.mean().rename("pct_1enxerto_off")
    cods = pd.crosstab(x.mes, x.PROC_REA)
    m = m.join(on).join(of).join(cods)
    for c in ["pct_off", "pct_complemento", "pct_1enxerto_on", "pct_1enxerto_off"]:
        m[c] *= 100
    salvar(m.reset_index().round(2), "R6a_serie_mensal_codigo_2020_2025")
    # séries interrompidas (mai/2023): logística com tendência, nível e mudança de inclinação; erros por hospital
    x["t"] = (x.ano - 2020) * 12 + x.DT_INTER.str[4:6].astype(int) - 1
    x["pos"] = (x.DT_INTER.str[:6] >= "202305").astype(int)
    x["tpos"] = x.pos * (x.t - 40)
    its = []
    for nome, xx, y in [("% off-pump (all CABG)", x, "off"), ("% 1-graft among on-pump", x[x.off == 0], "graft1"),
                        ("% 1-graft among off-pump", x[x.off == 1], "graft1")]:
        go = xx.groupby(["CNES", "t", "pos", "tpos"]).agg(k=(y, "sum"), n=(y, "size")).reset_index()
        for fe in (False, True):
            mm = ModeloFE("t + pos + tpos", go) if fe else Modelo("t + pos + tpos", go)
            for termo in ("pos", "tpos"):
                its.append(lin(nome, mm, termo, efeitos_fixos_hospital=fe))
    salvar(pd.DataFrame(its).round(4), "R6b_serie_interrompida_mai2023")
    # quem puxou a alta de 2024–2025 (contribuição exata para a variação nacional da %)
    p1, p2 = d[d.ano.between(2020, 2023)], d[d.ano.between(2024, 2025)]
    N1, N2 = len(p1), len(p2)
    h = pd.DataFrame({"n1": p1.groupby("CNES").size(), "off1": p1.groupby("CNES").off.sum(),
                      "n2": p2.groupby("CNES").size(), "off2": p2.groupby("CNES").off.sum()}).fillna(0)
    h["contrib_pp"] = 100 * (h.off2 / N2 - h.off1 / N1)
    h["pct1"] = 100 * h.off1 / h.n1.replace(0, np.nan)
    h["pct2"] = 100 * h.off2 / h.n2.replace(0, np.nan)
    h["partic_crvm1_pct"] = 100 * h.n1 / N1
    h["partic_crvm2_pct"] = 100 * h.n2 / N2
    h = h.sort_values("contrib_pp", ascending=False)
    delta = 100 * (p2.off.mean() - p1.off.mean())
    attr = d.groupby("CNES").agg(regiao=("reg", "first"), natureza=("natj", "first"))
    top = h.head(15).join(attr)
    top.index = top.index.map(anon(top.index, "C"))
    top = top.reset_index(names="hospital")
    top["pct_da_variacao_nacional"] = 100 * top.contrib_pp / delta
    salvar(top.round(3), "R6c_hospitais_alta_2024_2025_anon")
    R["alta_2024_25"] = {"pct_2020_2023": 100 * p1.off.mean(), "pct_2024_2025": 100 * p2.off.mean(), "delta_pp": delta,
                         "top1_pct_da_variacao": float(100 * h.contrib_pp.iloc[0] / delta),
                         "top5_pct_da_variacao": float(100 * h.contrib_pp.iloc[:5].sum() / delta),
                         "top10_pct_da_variacao": float(100 * h.contrib_pp.iloc[:10].sum() / delta),
                         "hospitais_aumento_>=5pp_(>=20 crvm em cada período)": int(((h.pct2 - h.pct1 >= 5) & (h.n1 >= 20) & (h.n2 >= 20)).sum()),
                         "hospitais_queda_>=5pp_(>=20 crvm em cada período)": int(((h.pct2 - h.pct1 <= -5) & (h.n1 >= 20) & (h.n2 >= 20)).sum())}
    # teste de coerência: inversão do preço relativo em 2008
    moda = lambda s: float(s.round(2).mode().iloc[0])
    y = d[d.ano.between(2003, 2012)]
    pr = y.groupby(["ano", np.where(y.off == 1, "Off-pump", "On-pump")]).agg(
        n=("off", "size"), sh_modal=("VAL_SH", moda), sp_modal=("VAL_SP", moda), tot_medio=("VAL_TOT", "mean")).unstack()
    pr.columns = [f"{a}_{b}" for a, b in pr.columns]
    pr["pct_off"] = 100 * pr["n_Off-pump"] / (pr["n_Off-pump"] + pr["n_On-pump"])
    pr["razao_sh_modal_off_on"] = pr["sh_modal_Off-pump"] / pr["sh_modal_On-pump"]
    pr["razao_sh_sp_modal_off_on"] = (pr["sh_modal_Off-pump"] + pr["sp_modal_Off-pump"]) / (pr["sh_modal_On-pump"] + pr["sp_modal_On-pump"])
    pr["razao_tot_medio_off_on"] = pr["tot_medio_Off-pump"] / pr["tot_medio_On-pump"]
    salvar(pr.reset_index().round(3), "R6d_preco_relativo_2003_2012")
    # mudança de nível jan/2008 dentro do hospital (hospitais com CRVM em 2007 e 2008); só se 2007 está na janela
    if INI_REV > 2007:
        log("6: codificação (sem transição 2008: janela começa em 2008)")
        return
    z = d[d.ano.isin([2007, 2008])]
    ambos = z.groupby("CNES").ano.nunique() == 2
    z = z[z.CNES.isin(ambos[ambos].index)].assign(pos08=lambda q: (q.ano == 2008).astype(int))
    gz = z.groupby(["CNES", "pos08"]).agg(k=("off", "sum"), n=("off", "size")).reset_index()
    mz = ModeloFE("pos08", gz)
    mon = d[d.DT_INTER.str[:6].between("200707", "200806")].groupby(d.DT_INTER.str[:6]).off.agg(["size", "mean"])
    R["transicao_2008"] = {"pct_off_2007": float(100 * d[d.ano == 2007].off.mean()), "pct_off_2008": float(100 * d[d.ano == 2008].off.mean()),
                           "OR_2008_vs_2007_dentro_hospital": mz.orci("pos08"), "hospitais": mz.n_hosp,
                           "mensal_jul07_jun08_pct_off": {k: round(100 * v, 2) for k, v in mon["mean"].items()}}
    log("6: codificação")


# ============================================================================ 7. regional
def regional(d):
    tabs = []
    for var, rot in [("reg", "Region"), ("natj", "Legal status")]:
        t = d.groupby(["ano", var]).off.agg(["size", "sum"]).reset_index()
        t.columns = ["ano", "grupo", "n", "n_off"]
        t["pct_off"] = 100 * t.n_off / t.n
        lo, hi = wilson(t.n_off, t.n)
        t["lo"], t["hi"] = 100 * lo, 100 * hi
        t["hospitais"] = d.groupby(["ano", var]).CNES.nunique().values
        tabs.append(t.assign(variavel=rot))
    salvar(pd.concat(tabs)[["variavel", "grupo", "ano", "n", "n_off", "pct_off", "lo", "hi", "hospitais"]].round(2), "R7a_serie_regiao_natureza")
    # conferência dos números regionais citados no texto (2003–2007 × 2020–2025)
    cit = {("reg", "Northeast"): (20.2, 33.6), ("reg", "Center-West"): (17.2, 4.2), ("reg", "Southeast"): (10.0, 5.7),
           ("natj", "Private for-profit"): (15.1, 29.1)}
    linhas = []
    for (var, grp), (r1, r2) in cit.items():
        a = 100 * d[(d[var] == grp) & d.ano.between(2003, 2007)].off.mean()
        b = 100 * d[(d[var] == grp) & d.ano.between(2020, 2025)].off.mean()
        linhas.append({"grupo": grp, "citado_2003_2007": r1, "calculado_2003_2007": round(a, 2), "citado_2020_2025": r2,
                       "calculado_2020_2025": round(b, 2), "confere": abs(a - r1) < 0.06 and abs(b - r2) < 0.06})
    hi_ = d[d.cat_hosp == "High (≥20%)"]
    linhas.append({"grupo": "High-use hospitals (≥20% over 2003–2025)", "citado_2003_2007": 35.0,
                   "calculado_2003_2007": round(100 * hi_[hi_.ano <= 2007].off.mean(), 2), "citado_2020_2025": 49.0,
                   "calculado_2020_2025": round(100 * hi_[hi_.ano >= 2020].off.mean(), 2), "confere": None})
    salvar(pd.DataFrame(linhas), "R7b_conferencia_numeros_regionais")
    # quantos hospitais e quanto volume sem CEC por região e período
    t = d.assign(per=av.periodo(d.ano)).groupby(["reg", "per"]).agg(
        aih=("off", "size"), aih_off=("off", "sum"), hospitais=("CNES", "nunique"),
        hospitais_com_off=("CNES", lambda s: s[d.loc[s.index, "off"] == 1].nunique())).reset_index()
    t["pct_off"] = 100 * t.aih_off / t.aih
    salvar(t.round(2), "R7c_regiao_periodo_hospitais")


# ============================================================================ 8. conversão: análise quantitativa de viés
def conversao(d):
    """Valores publicados (conferidos no PubMed):
    Stevens 2017, CORONARY (PMID 28082464): conversão OFF→ON 7,9% (186/2356); emergente 3,2% (75/2356).
    Mukherjee 2012, metanálise (PMID 21684171): 920 conversões em 18.870 OPCAB (4,9%); OR de óbito, convertido ×
    não convertido, 6,18 (IC 4,65–8,20).
    Cenário: os convertidos foram faturados como com CEC. Recompõe-se o grupo 'intenção sem CEC' (observado + convertidos)
    e retira-se os convertidos do grupo com CEC; a mortalidade dos convertidos sai do OR aplicado à mortalidade dos não
    convertidos (sem CEC observado)."""
    n_off, k_off = int(d.off.sum()), int(d.loc[d.off == 1, "MORTE"].sum())
    n_on, k_on = int((d.off == 0).sum()), int(d.loc[d.off == 0, "MORTE"].sum())
    m_off = k_off / n_off
    linhas = []
    for c_nome, c in [("Emergent conversion, CORONARY (3.2%)", 75 / 2356), ("Meta-analysis pooled (4.9%)", 920 / 18870),
                      ("All conversions, CORONARY (7.9%)", 186 / 2356)]:
        for or_nome, orc in [("OR 4.65 (lower CI)", 4.65), ("OR 6.18 (pooled)", 6.18), ("OR 8.20 (upper CI)", 8.20)]:
            od = m_off / (1 - m_off) * orc
            m_c = od / (1 + od)
            n_c = n_off * c / (1 - c)
            k_c = n_c * m_c
            mi_off = (k_off + k_c) / (n_off + n_c)
            mi_on = (k_on - k_c) / (n_on - n_c)
            rd_obs = 100 * (m_off - k_on / n_on)
            rd_cor = 100 * (mi_off - mi_on)
            linhas.append({"taxa_conversao": c_nome, "or_obito_convertido": or_nome, "mort_convertidos_pct": 100 * m_c,
                           "convertidos_estimados": round(n_c), "mort_sem_cec_intencao_pct": 100 * mi_off,
                           "mort_com_cec_sem_convertidos_pct": 100 * mi_on, "dr_bruta_observada_pp": rd_obs,
                           "dr_bruta_corrigida_pp": rd_cor, "pct_da_diferenca_explicada": 100 * (rd_cor - rd_obs) / -rd_obs,
                           "or_bruto_observado": (m_off / (1 - m_off)) / ((k_on / n_on) / (1 - k_on / n_on)),
                           "or_bruto_corrigido": (mi_off / (1 - mi_off)) / (mi_on / (1 - mi_on))})
    salvar(pd.DataFrame(linhas).round(3), "R8_vies_conversao")


# ============================================================================ 9. especificação original lado a lado
def especificacao_original():
    pastas = [("Original pre-specified: 2008–2025 with admission type", "resultados_v2"),
              ("2008–2025 without admission type", "resultados_v2_2008_2025_sem_carater"),
              ("Main: 2003–2025 without admission type", "resultados_v2_2003_2025_sem_carater")]
    linhas = []
    for rot, p in pastas:
        base = os.path.join(RAIZ, p)
        js = json.load(open(os.path.join(base, "resultados_v2.json")))
        m4 = pd.read_csv(os.path.join(base, "M4_institucional.csv"))
        m6 = pd.read_csv(os.path.join(base, "M6_individual.csv"))
        iv = pd.read_csv(os.path.join(base, "IV1_variavel_instrumental.csv"))
        v1 = pd.read_csv(os.path.join(base, "V1_modelo_hierarquico.csv"))
        f = lambda r, a="or", b="lo", c="hi", k=3: f"{r[a]:.{k}f} ({r[b]:.{k}f}–{r[c]:.{k}f})"
        pri = m4[m4.modelo.str.contains("PRIMARY")].iloc[0]
        fe = m4[m4.modelo.str.contains("SENSITIVITY")].iloc[0]
        ip = m6[m6.modelo.str.contains("PRIMARY")].iloc[0]
        ife = m6[m6.modelo == "Adjusted + hospital FE"].iloc[0]
        ivfe = iv[(iv.instrumento.str.startswith("prior")) & (iv.efeitos_fixos_hospital)].iloc[0]
        ivn = iv[(iv.instrumento.str.startswith("prior")) & (~iv.efeitos_fixos_hospital)].iloc[0]
        cats = m4[m4.modelo == "Institutional, categories vs 0%, no hospital FE"]
        v = v1.iloc[-1]
        linhas += [
            {"especificacao": rot, "medida": "Adult admissions; hospitals", "valor": f"{js['n_adultos']:,}; {js['n_hospitais']}"},
            {"especificacao": rot, "medida": "OPCAB %; crude mortality %", "valor": f"{js['off_global_pct']:.2f}; {js['mort_global_pct']:.2f}"},
            {"especificacao": rot, "medida": "MOR (95% CI), year + case mix", "valor": f"{v.MOR:.1f} ({v.MOR_lo:.1f}–{v.MOR_hi:.1f})"},
            {"especificacao": rot, "medida": "Hospital strategy, OR per +10 pp (primary)", "valor": f(pri) + f"; boot p={pri.p_boot:.3f}"},
            {"especificacao": rot, "medida": "Hospital strategy, RD pp per +10 pp", "valor": f(pri, "dr_pp", "dr_lo", "dr_hi", 2)},
            {"especificacao": rot, "medida": "Hospital strategy categories vs 0% (>0–5 / >5–15 / >15–30 / >30)",
             "valor": " / ".join(f"{r['or']:.2f}" for _, r in cats.iterrows())},
            {"especificacao": rot, "medida": "Hospital strategy with hospital FE, OR per +10 pp", "valor": f(fe)},
            {"especificacao": rot, "medida": "Patient level adjusted OR (primary)", "valor": f(ip)},
            {"especificacao": rot, "medida": "Patient level RD pp", "valor": f(ip, "dr_pp", "dr_lo", "dr_hi", 2)},
            {"especificacao": rot, "medida": "Patient level with hospital FE, OR", "valor": f(ife)},
            {"especificacao": rot, "medida": "IV prior-year, no FE, pp", "valor": f(ivn, "iv_dr_pp", "iv_lo", "iv_hi", 2)},
            {"especificacao": rot, "medida": "IV prior-year, hospital FE, pp", "valor": f(ivfe, "iv_dr_pp", "iv_lo", "iv_hi", 2)},
        ]
        el = m6[m6.modelo.str.startswith("Elective only (")]
        if len(el):
            linhas.append({"especificacao": rot, "medida": "Patient level OR, elective only (admission type as coded)", "valor": f(el.iloc[0])})
    t = pd.DataFrame(linhas).pivot(index="medida", columns="especificacao", values="valor")
    t = t[[r for r, _ in pastas]].reset_index()
    salvar(t, "R9_especificacao_original_lado_a_lado")


# ============================================================================ 10. análises adicionais (23/out/2026)
def adicionais2(d, B):
    VOL, MAT = "cr(logvol, df=4, constraints='center')", "cr(anos_prog, df=3, constraints='center')"
    HUF = "C(natj, Treatment('Nonprofit/philanthropic')) + C(uf)"
    g = cel(d, ["expo", "hy_n", "logvol", "anos_prog", "x_lag1"])
    gi = g[g.expo.notna()].assign(e10=lambda z: z.expo / 10, algum=lambda z: (z.expo > 0).astype(int))
    R["n_celulas_modelo_principal"] = int(len(gi))
    form = f"{FORM} + {HOSP}"
    lin_ = []
    m = Modelo(f"{form} + e10", gi)
    lin_.append(lin("Primary (reference)", m, "e10", m.dr(dr_continua("e10", 1)), **m.boot_ic("e10", B)))
    m = Modelo(f"{form} + {VOL} + {MAT} + e10", gi)
    lin_.append(lin("+ volume + years since first CABG", m, "e10", m.dr(dr_continua("e10", 1)), or_6casas=m.orci("e10")))
    m = Modelo(f"{FORM} + {HUF} + {VOL} + {MAT} + e10", gi)
    lin_.append(lin("+ volume + years since first CABG + state", m, "e10", m.dr(dr_continua("e10", 1)), **m.boot_ic("e10", B)))
    m = Modelo(f"{form} + algum", gi)
    lin_.append(lin("Any vs no OPCAB activity (alone)", m, "algum", m.dr(dr_dummy("algum", ["algum"])), **m.boot_ic("algum", B)))
    # restrito aos hospitais que identificam o modelo com efeitos fixos
    p = av.pesos_identificacao(d)
    kh = gi.groupby("CNES").k.sum()
    nh = gi.groupby("CNES").n.sum()
    com_var = set(kh[(kh > 0) & (kh < nh)].index)
    ident = set(p[p > 1e-9].index) & com_var
    top = set(p.index[:int(round(1 / (p ** 2).sum()))])
    for nome, conj in [("Hospitals contributing to the fixed-effects model", ident),
                       ("Top 27 identifying hospitals", top),
                       ("Hospitals NOT contributing to the fixed-effects model", set(gi.CNES) - ident)]:
        gx = gi[gi.CNES.isin(conj)]
        if gx.k.sum() == 0:
            continue
        m = Modelo(f"{form} + e10", gx)
        lin_.append(lin(f"Between hospitals, restricted: {nome}", m, "e10", m.dr(dr_continua("e10", 1))))
    R["hospitais_contribuem_FE"] = len(ident)
    # sem os hospitais que puxaram a alta de 2024–25 e sem os maiores em volume sem CEC
    p1, p2 = d[d.ano.between(2020, 2023)], d[d.ano.between(2024, 2025)]
    cont = (p2.groupby("CNES").off.sum() / len(p2)).sub(p1.groupby("CNES").off.sum() / len(p1), fill_value=0).sort_values(ascending=False)
    top5 = list(cont.index[:5])
    offvol = d.groupby("CNES").off.sum().sort_values(ascending=False)
    for nome, fora in [("Without the 5 hospitals driving the 2024–2025 increase", top5),
                       ("Without the 10 largest OPCAB hospitals", list(offvol.index[:10]))]:
        m = Modelo(f"{form} + e10", gi[~gi.CNES.isin(fora)])
        lin_.append(lin(nome, m, "e10", m.dr(dr_continua("e10", 1))))
    for reg in ["North", "Northeast", "Southeast", "South", "Center-West"]:
        gx = gi[gi.reg != reg]
        fr = form.replace("Treatment('Southeast')", "Treatment('South')") if reg == "Southeast" else form
        m = Modelo(f"{fr} + e10", gx)
        lin_.append(lin(f"Leaving out region: {reg}", m, "e10", m.dr(dr_continua("e10", 1))))
    lu = []
    for uf in sorted(gi.uf.unique()):
        m = Modelo(f"{form} + e10", gi[gi.uf != uf])
        lu.append({"uf_retirada": uf, **{k: v for k, v in m.orci("e10").items() if k in ("or", "lo", "hi", "p")}})
    lu = pd.DataFrame(lu)
    lin_.append({"modelo": f"Leaving out one state at a time (min–max of {len(lu)} fits)", "termo": "e10",
                 "or": lu["or"].min(), "lo": lu.lo.min(), "hi": lu["or"].max(), "p": lu.p.max()})
    salvar(pd.DataFrame(lin_).round(4), "R10a_entre_hospitais_robustez")
    salvar(lu.round(4), "R10b_sem_uma_UF")
    log("10: entre hospitais")
    # contraste dentro de cada UF
    h = d.groupby("CNES").agg(uf=("uf", "first"), n=("off", "size"), n_off=("off", "sum"))
    t = h.groupby("uf").agg(hospitais=("n", "size"), hospitais_com_off=("n_off", lambda s: int((s > 0).sum())),
                            aih=("n", "sum"), aih_off=("n_off", "sum")).reset_index()
    t["pct_off"] = 100 * t.aih_off / t.aih
    t["pct_do_off_nacional"] = 100 * t.aih_off / t.aih_off.sum()
    t["contraste_dentro_UF"] = (t.hospitais >= 2) & (t.hospitais_com_off >= 1) & (t.hospitais_com_off < t.hospitais)
    salvar(t.round(2), "R10c_contraste_dentro_UF")
    R["UF_sem_contraste"] = {"ufs": int((~t.contraste_dentro_UF).sum()), "pct_aih": float(100 * t.loc[~t.contraste_dentro_UF, "aih"].sum() / t.aih.sum()),
                             "pct_off": float(100 * t.loc[~t.contraste_dentro_UF, "aih_off"].sum() / t.aih_off.sum())}
    # série anual dos diagnósticos por técnica
    x = d.assign(tec=np.where(d.off == 1, "Off-pump", "On-pump"))
    s = pd.crosstab([x.ano, x.tec], x.diag, normalize="index").mul(100).round(2).reset_index()
    salvar(s, "R10d_diagnostico_por_ano_tecnica")
    # sem CEC de 1 enxerto em 2024–25: hospitais
    y = d[(d.off == 0) & (d.ano >= 2020)]
    a1, a2 = y[y.ano <= 2023], y[y.ano >= 2024]
    c1 = (a2.groupby("CNES").graft1.sum() / len(a2)).sub(a1.groupby("CNES").graft1.sum() / len(a1), fill_value=0).sort_values(ascending=False)
    dl = 100 * (a2.graft1.mean() - a1.graft1.mean())
    mapa = {c: f"C{i + 1:02d}" for i, c in enumerate(cont.index[:15])}
    r = pd.DataFrame({"contrib_pp": 100 * c1.head(10)})
    r["pct_da_variacao"] = 100 * r.contrib_pp / dl
    r["entre_os_5_da_alta_sem_CEC"] = r.index.isin(top5)
    r["id_na_tabela_R6c"] = [mapa.get(c, "") for c in r.index]
    r = r.reset_index(drop=True)
    r.insert(0, "hospital", [f"G{i + 1:02d}" for i in range(len(r))])
    salvar(r.round(3), "R10e_alta_1enxerto_com_CEC_2024_25")
    R["alta_1enxerto_com_CEC"] = {"pct_2020_23": float(100 * a1.graft1.mean()), "pct_2024_25": float(100 * a2.graft1.mean()),
                                  "top5_pct_da_variacao": float(r.pct_da_variacao.head(5).sum()),
                                  "pct_da_variacao_nos_5_da_alta_sem_CEC": float(100 * (100 * c1[c1.index.isin(top5)]).sum() / dl)}
    log("10: descritivos")
    # o bootstrap de hospitais dos modelos com efeitos fixos (R10f) está em analise_adicional_bootfe.py


# ============================================================================ Excel e main
LEIA = [
    ("R1a_confiabilidade_pct_anual", "1. Confiabilidade da % anual sem CEC dentro do hospital: variância binomial esperada ÷ variância intra-hospital observada."),
    ("R1b_exposicao_predeterminada", "1. Estimando institucional com exposição predeterminada (ano anterior, média móvel 3 anos), encolhida (EB), sem leave-one-out e hospital-anos ≥30/≥50; com e sem efeitos fixos; wild cluster bootstrap nos efeitos fixos. Referência: −0,236 pp por +10 pp se o OR individual fosse causal."),
    ("R1c_hospital_anos_por_categoria", "1. AIH e hospital-anos por categoria da exposição."),
    ("R1d_transicoes_categoria", "1. Transições de categoria entre anos consecutivos dentro do hospital."),
    ("R2a_estudo_eventos", "2. Estudo de eventos (probabilidade linear, efeitos fixos de hospital e ano): mudança sustentada ≥10 pp na % sem CEC, definida antes de olhar a mortalidade; referência t−1."),
    ("R2b_hospitais_identificadores_anon", "2. Hospitais que identificam o modelo com efeitos fixos (top = nº efetivo), anonimizados."),
    ("R2c_identificadores_resumo", "2. Identificadores × demais hospitais."),
    ("R2d_eventos_anon", "2. Hospitais com evento (mudança sustentada ≥10 pp), anonimizados."),
    ("R3a_institucional_volume_maturidade", "3. Modelo institucional + volume anual (spline) + anos desde a 1ª CRVM + UF; 0% × algum sem CEC; categorias ajustadas."),
    ("R3b_volume_mortalidade", "3. Volume anual de CRVM do hospital × mortalidade (global e por técnica), referência ≥400/ano."),
    ("R3c_OR_individual_por_categoria_hospital", "3. Teste de seleção: OR individual sem × com CEC por categoria de estratégia e de volume sem CEC do hospital."),
    ("R4a_enxertos_por_ano_tecnica", "4. % de códigos de 1 enxerto por ano e técnica (2008–2025), só descritivo."),
    ("R4b_enxertos_por_categoria_hospital", "4. % de 1 enxerto por técnica e categoria do hospital (2008–2025)."),
    ("R4c_OR_individual_por_enxertos", "4. OR individual estratificado pelo código de enxertos (sem ajuste), todos e 130 hospitais estáveis; e só apresentações crônicas."),
    ("R5a_momento_obito", "5. Dias da internação ao óbito por técnica."),
    ("R5b_desfechos_alternativos", "5. Óbito ≤30 dias na própria AIH; óbito ou transferência (2008–2025)."),
    ("R5c_transferencias", "5. % de transferências (COBRANCA 31) por técnica e categoria de hospital, 2008–2025."),
    ("R5d_concordancia_MORTE_motivo_saida", "5. Concordância entre o campo MORTE e o motivo de saída."),
    ("R6a_serie_mensal_codigo_2020_2025", "6. Série mensal por código (mês de internação), % sem CEC, % 1 enxerto, % com complemento federal."),
    ("R6b_serie_interrompida_mai2023", "6. Série interrompida em mai/2023 (nível 'pos' e inclinação 'tpos', por mês)."),
    ("R6c_hospitais_alta_2024_2025_anon", "6. Hospitais que mais contribuíram para a variação da % sem CEC de 2020–23 para 2024–25 (anonimizados)."),
    ("R6d_preco_relativo_2003_2012", "6. Teste de coerência: preço relativo sem/com CEC e % sem CEC em torno de 2008."),
    ("R7a_serie_regiao_natureza", "7. % sem CEC por ano, região e natureza jurídica."),
    ("R7b_conferencia_numeros_regionais", "7. Conferência dos números regionais citados no texto."),
    ("R7c_regiao_periodo_hospitais", "7. Hospitais e volume sem CEC por região e período."),
    ("R8_vies_conversao", "8. Análise de viés da conversão com valores publicados (Stevens 2017; Mukherjee 2012)."),
    ("R9_especificacao_original_lado_a_lado", "9. Especificação original (2008–2025 com caráter) lado a lado com as demais."),
    ("R10a_entre_hospitais_robustez", "10. 2ª rodada: modelo entre hospitais restrito aos hospitais que identificam os efeitos fixos, sem regiões/UF/hospitais influentes; algum × nenhum; bootstrap."),
    ("R10b_sem_uma_UF", "10. Modelo principal retirando uma UF de cada vez."),
    ("R10c_contraste_dentro_UF", "10. Hospitais com e sem CEC por UF (há contraste dentro da UF?)."),
    ("R10d_diagnostico_por_ano_tecnica", "10. Distribuição do diagnóstico principal por ano e técnica."),
    ("R10e_alta_1enxerto_com_CEC_2024_25", "10. Hospitais que puxaram a alta do código com CEC de 1 enxerto em 2024–25 (anonimizados; C = mesmos IDs de R6c)."),
    ("R10f_FE_diferenca_risco_bootstrap", "10. Diferença de risco com efeitos fixos: IC por bootstrap de hospitais (inclui a incerteza dos interceptos)."),
]


def excel(caminho):
    with pd.ExcelWriter(caminho, engine="xlsxwriter") as xw:
        cab = xw.book.add_format({"bold": True, "bg_color": "#DDE6F0", "text_wrap": True, "valign": "top", "border": 1})
        leia = pd.DataFrame([("Leia-me", "Análises adicionais post hoc (artigo/analise_adicional.py); "
                              "janela 2003–2025 sem caráter, salvo indicação.")] + LEIA, columns=["aba", "conteúdo"])
        leia.to_excel(xw, sheet_name="Leia-me", index=False)
        xw.sheets["Leia-me"].set_column(0, 0, 38)
        xw.sheets["Leia-me"].set_column(1, 1, 140)
        for aba, _ in LEIA:
            arq = os.path.join(OUT, aba + ".csv")
            if not os.path.exists(arq):
                continue
            df = pd.read_csv(arq)
            nome = aba[:31]
            df.to_excel(xw, sheet_name=nome, index=False)
            ws = xw.sheets[nome]
            for j, c in enumerate(df.columns):
                ws.write(0, j, c, cab)
                larg = max(len(str(c)), *(len(str(v)) for v in df[c].head(200))) if len(df) else len(str(c))
                ws.set_column(j, j, min(max(10, larg + 1), 60))
            ws.freeze_panes(1, 1)
        res = pd.json_normalize(R, sep=".").T.reset_index()
        res.columns = ["resultado", "valor"]
        res["valor"] = res.valor.astype(str)
        res.to_excel(xw, sheet_name="Resultados_chave", index=False)
        xw.sheets["Resultados_chave"].set_column(0, 0, 70)
        xw.sheets["Resultados_chave"].set_column(1, 1, 120)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--boot", type=int, default=9999)
    ap.add_argument("--so", default="", help="itens a rodar (vírgulas): 1,...,10")
    a = ap.parse_args()
    os.makedirs(OUT, exist_ok=True)
    so = set(a.so.split(",")) if a.so else None
    rodar = lambda e: so is None or e in so
    jp = os.path.join(OUT, "resultados_adicionais.json")
    if so and os.path.exists(jp):
        R.update(json.load(open(jp)))
    R["configuracao"] = {"janela": f"{INI_REV}–2025", "carater": CAR_REV, "casemix": CASEMIX, "hospital": HOSP, "boot": a.boot}
    d = preparar()
    log(f"{len(d)} AIH")
    if rodar("1"):
        confiabilidade(d)
        expo_predeterminada(d, a.boot)
    if rodar("2"):
        ev = eventos(d)
        identificadores(d, ev)
    if rodar("3"):
        volume_maturidade(d)
        selecao(d)
    if rodar("4"):
        enxertos(d)
    if rodar("5"):
        desfecho(d)
    if rodar("6"):
        codificacao(d)
    if rodar("7"):
        regional(d)
    if rodar("8"):
        conversao(d)
    if rodar("9"):
        especificacao_original()
    if rodar("10"):
        adicionais2(d, a.boot)
    with open(jp, "w") as f:
        json.dump(R, f, indent=1, ensure_ascii=False,
                  default=lambda o: float(o) if isinstance(o, np.floating) else int(o) if isinstance(o, np.integer) else str(o))
    excel(os.path.join(RAIZ, f"tabelas_adicionais{SUF}.xlsx"))
    log("fim")


if __name__ == "__main__":
    main()
