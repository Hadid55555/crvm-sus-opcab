#!/usr/bin/env python3
"""Análises do plano v2 (artigo CRVM com × sem CEC no SUS, 2008–2025).

Especificação fixada antes de rodar: artigo/plano_artigo_cec_v2.md, seção "Especificação fixada antes de rodar".
Sem figuras. Saídas: artigo/resultados_v2/*.csv, resultados_v2.json e artigo/tabelas_v2_crvm_2008_2025.xlsx.

Uso: python artigo/analise_v2.py [--boot 9999] [--ini 2008 --fim 2025] [--sem-carater] [--micro CSV] [--tag _x]
Janela 2003–2025 (emenda de 15/out/2026): --micro saida_crvm/crvm_2000_2025_internacao_microdados.csv --ini 2003
O caráter da internação não entra em nenhuma análise (implausível clinicamente; decisão do autor, 15/out/2026).
"""
import argparse
import json
import os
import time
import warnings

import numpy as np
import pandas as pd
import patsy
import statsmodels.api as sm
from scipy import optimize, stats
from scipy.special import expit, logsumexp

warnings.filterwarnings("ignore")
RAIZ = os.path.dirname(os.path.abspath(__file__))
MICRO = os.path.join(RAIZ, "..", "saida_crvm", "crvm_2008_2025_internacao_microdados.csv")
OUT = os.path.join(RAIZ, "resultados_v2")
INI, FIM, CARATER = 2008, 2025, True

COD = {"0406010927": "On-pump, 1 graft", "0406010935": "On-pump, ≥2 grafts",
       "0406010943": "Off-pump, 1 graft", "0406010951": "Off-pump, ≥2 grafts",
       "32011016": "On-pump (old table 32, generic/1 graft)", "32039018": "On-pump, ≥2 grafts (old table 32)",
       "32038011": "Off-pump (old table 32, generic/1 graft)", "32040016": "Off-pump, ≥2 grafts (old table 32)",
       "48010073": "On-pump, 1 graft (old table 48)", "48010081": "On-pump, ≥2 grafts (old table 48)",
       "48010090": "Off-pump, 1 graft (old table 48)", "48010103": "Off-pump, ≥2 grafts (old table 48)"}
REG_EN = {"Norte": "North", "Nordeste": "Northeast", "Sudeste": "Southeast", "Sul": "South", "Centro-Oeste": "Center-West"}
EXPO_CAT = [-1, 0, 5, 15, 30, 101]
EXPO_LAB = ["0%", ">0-5%", ">5-15%", ">15-30%", ">30%"]
CASEMIX = "cr(idade, df=4, constraints='center') + fem + urg + C(diag, Treatment('I25 chronic IHD'))"
HOSP = "C(reg, Treatment('Southeast')) + C(natj, Treatment('Nonprofit/philanthropic'))"
R = {}
T0 = time.time()


def periodos():
    if INI < 2008:
        return [(INI, 2007), (2008, 2013), (2014, 2019), (2020, FIM)]
    return [(INI, 2013), (2014, 2019), (2020, FIM)]


def periodo(ano):
    ano = np.asarray(ano)
    out = np.empty(len(ano), dtype=object)
    for a, b in periodos():
        out[(ano >= a) & (ano <= b)] = f"{a}-{b}"
    return out


def log(msg):
    print(f"[{time.time() - T0:7.0f}s] {msg}", flush=True)


def salvar(df, nome):
    df.to_csv(os.path.join(OUT, nome + ".csv"), index=False)
    return df


def wilson(k, n, z=1.96):
    p = k / n
    d = 1 + z ** 2 / n
    c = (p + z ** 2 / (2 * n)) / d
    h = z * np.sqrt(p * (1 - p) / n + z ** 2 / (4 * n ** 2)) / d
    return c - h, c + h


# ============================================================================ dados
def natj_grupo(s):
    s = s.dropna().astype(str)
    if s.empty:
        return "Unknown"
    m = s.mode().iloc[0][0]
    return {"1": "Public", "2": "Private for-profit", "3": "Nonprofit/philanthropic"}.get(m, "Unknown")


def carregar():
    d = pd.read_csv(MICRO, low_memory=False, dtype={"N_AIH": str, "PROC_REA": str, "CNES": str, "SEXO": str,
                                                     "CAR_INT": str, "DIAG_PRINC": str, "COBRANCA": str,
                                                     "NAT_JUR": str, "DT_INTER": str, "ANO_CMPT": str, "MES_CMPT": str,
                                                     "IDENT": str, "CGC_HOSP": str, "CNES_ORIG": str, "grupo": str})
    d = d[d.ano_internacao.between(INI, FIM)].copy()
    if "grupo" in d:
        d = d[d.grupo.isin(["Com CEC", "Sem CEC"])].copy()  # "Não separável" (código único até fev/2002) fica fora
    R["fluxo"] = {f"registros_RD_ano_internacao_{INI}_{FIM}": int(len(d)), "menores_18": int((d.idade_anos < 18).sum()),
                  "aih_duplicadas": int(d.n_aih_duplicado.sum()),
                  "longa_permanencia_ident5": int((d.IDENT.astype(str).str.strip() == "5").sum())}
    d = d[d.idade_anos >= 18].copy()
    dup = d.N_AIH.duplicated()
    R["fluxo"]["aih_reapresentada_excluida"] = int(dup.sum())
    d = d[~dup].copy()
    R["fluxo"]["adultos_analisados"] = int(len(d))
    d["ano"] = d.ano_internacao.astype(int)
    d["idade"] = d.idade_anos.clip(upper=95).astype(int)
    d["off"] = (d.grupo == "Sem CEC").astype(int)
    d["fem"] = d.SEXO.str.strip().isin(["2", "3"]).astype(int)
    d["urg"] = (d.CAR_INT.str.strip().str.zfill(2) != "01").astype(int) if CARATER else 0
    cid = d.DIAG_PRINC.str.strip().str.upper()
    d["diag"] = np.select([cid.str.startswith(("I21", "I22")), cid.str.startswith("I200"), cid.str.startswith("I20"),
                           cid.str.startswith("I24"), cid.str.startswith("I25")],
                          ["AMI (I21-I22)", "Unstable angina (I20.0)", "Other angina (I20.1-9)", "Other acute IHD (I24)",
                           "I25 chronic IHD"], default="Other codes")
    d["reg"] = d.regiao.map(REG_EN)
    nat = d.groupby("CNES").NAT_JUR.apply(natj_grupo)
    d["natj"] = d.CNES.map(nat)
    d["cob"] = d.COBRANCA.str.strip()
    d["morte_transf"] = ((d.MORTE == 1) | (d.cob == "31")).astype(int)
    d["codigo"] = d.PROC_REA.map(COD)
    d["fed"] = d[["VAL_SH_FED", "VAL_SP_FED"]].fillna(0).sum(axis=1)
    ipca = pd.read_csv(os.path.join(RAIZ, "dados_ref", "ipca_fator_dez2025_2000_2025.csv")).set_index("ano")["fator_dez2025"]
    d["f_ipca"] = d.ano.map(ipca)
    # exposição institucional: % sem CEC do hospital no ano, sem o próprio paciente
    hy = d.groupby(["CNES", "ano"]).off.agg(hy_off="sum", hy_n="count")
    d = d.join(hy, on=["CNES", "ano"])
    d["expo"] = np.where(d.hy_n > 1, 100 * (d.hy_off - d.off) / (d.hy_n - 1).clip(lower=1), np.nan)
    d["expo_cat"] = pd.cut(d.expo, EXPO_CAT, labels=EXPO_LAB).astype(str)
    # instrumento defasado: % sem CEC do hospital no ano anterior (≥10 CRVM no ano anterior)
    lag = hy.reset_index()
    lag = lag[lag.hy_n >= 10].assign(ano=lambda x: x.ano + 1)
    lag["z_lag"] = 100 * lag.hy_off / lag.hy_n
    d = d.merge(lag[["CNES", "ano", "z_lag"]], on=["CNES", "ano"], how="left")
    # codificação estável de enxertos (arquivo A5 da auditoria)
    a5 = pd.read_csv(os.path.join(RAIZ, "resultados_diag", "A5_hospitais_mudanca_1enxerto.csv"), dtype={"CNES": str})
    d["cod_estavel"] = d.CNES.isin(a5.loc[a5.delta_pct1_on.abs() < 10, "CNES"]).astype(int)
    R["n_adultos"] = int(len(d))
    R["n_hospitais"] = int(d.CNES.nunique())
    R["n_obitos"] = int(d.MORTE.sum())
    R["n_hospital_anos_1_aih_excluidos_expo"] = int(d.expo.isna().sum())
    return d


CHAVES = ["CNES", "ano", "idade", "fem", "urg", "diag", "reg", "natj", "off", "expo", "expo_cat", "hy_n", "cod_estavel"]


def celulas(d, y="MORTE"):
    g = d.groupby(CHAVES, observed=True, dropna=False).agg(k=(y, "sum"), n=(y, "size")).reset_index()
    return g


# ============================================================================ modelos (GLM binomial em células)
def irls(X, k, n, offset=None, start=None, tol=1e-10, maxit=50):
    """Logística binomial por IRLS (numpy). Retorna beta, informação X'WX e probabilidades."""
    off = np.zeros(len(k)) if offset is None else offset
    if start is None:
        p0 = (k.sum() + 0.5) / (n.sum() + 1)
        b = np.zeros(X.shape[1])
        eta = np.full(len(k), np.log(p0 / (1 - p0)))
    else:
        b = start.copy()
        eta = X @ b + off
    ll_old = -np.inf
    for _ in range(maxit):
        p = expit(eta)
        w = n * p * (1 - p)
        z = eta - off + (k - n * p) / np.maximum(w, 1e-12)
        H = (X * w[:, None]).T @ X
        b = np.linalg.solve(H + 1e-10 * np.eye(len(H)), (X * w[:, None]).T @ z)
        eta = X @ b + off
        p = np.clip(expit(eta), 1e-15, 1 - 1e-15)
        ll = np.sum(k * np.log(p) + (n - k) * np.log(1 - p))
        if abs(ll - ll_old) < tol * (1 + abs(ll)):
            break
        ll_old = ll
    w = n * p * (1 - p)
    H = (X * w[:, None]).T @ X
    return b, H, p, ll


class Modelo:
    """Logística binomial em células; erros robustos por hospital; padronização marginal com método delta."""

    def __init__(self, formula, g, termo=None):
        self.g = g.reset_index(drop=True)
        self.X = patsy.dmatrix(formula, self.g, return_type="dataframe")
        self.cols = list(self.X.columns)
        self.grupos = pd.factorize(self.g.CNES)[0]
        X, k, n = self.X.values, self.g.k.values.astype(float), self.g.n.values.astype(float)
        self.b, H, p, self.ll = irls(X, k, n)
        Hi = np.linalg.pinv(H)
        S = np.zeros((self.grupos.max() + 1, X.shape[1]))
        np.add.at(S, self.grupos, X * (k - n * p)[:, None])
        G = S.shape[0]
        self.V = Hi @ (S.T @ S) @ Hi * G / (G - 1)
        self.n_aih, self.n_hosp, self.k_param = int(n.sum()), int(G), X.shape[1]
        self.termo = termo

    def idx(self, nome):
        return self.cols.index(nome)

    def orci(self, nome):
        j = self.idx(nome)
        b, se = self.b[j], np.sqrt(self.V[j, j])
        return {"or": float(np.exp(b)), "lo": float(np.exp(b - 1.96 * se)), "hi": float(np.exp(b + 1.96 * se)),
                "p": float(2 * stats.norm.sf(abs(b / se))), "beta": float(b), "se": float(se)}

    def dr(self, mudar):
        """Diferença de risco padronizada: mudar(X)->(X1, X0) em DataFrames de desenho."""
        X1, X0 = mudar(self.X.copy())
        w = self.g.n.values / self.g.n.sum()
        p1, p0 = expit(X1.values @ self.b), expit(X0.values @ self.b)
        est = np.sum(w * (p1 - p0))
        grad = (w * p1 * (1 - p1)) @ X1.values - (w * p0 * (1 - p0)) @ X0.values
        se = np.sqrt(grad @ self.V @ grad)
        return {"dr_pp": 100 * est, "lo": 100 * (est - 1.96 * se), "hi": 100 * (est + 1.96 * se),
                "risco1_pct": 100 * np.sum(w * p1), "risco0_pct": 100 * np.sum(w * p0)}

    def score_boot(self, nome, b0=0.0, B=9999, seed=1):
        """Wild cluster score bootstrap (Kline & Santos 2012), nula imposta, pesos Rademacher. Retorna p."""
        j = self.idx(nome)
        X = self.X.values
        Xr = np.delete(X, j, axis=1)
        k, n = self.g.k.values.astype(float), self.g.n.values.astype(float)
        off = b0 * X[:, j]
        _, _, p, _ = irls(Xr, k, n, offset=off, start=np.delete(self.b, j))
        resid = k - n * p
        S = np.zeros((self.grupos.max() + 1, X.shape[1]))
        np.add.at(S, self.grupos, X * resid[:, None])  # G × p
        Wt = n * p * (1 - p)
        H = (X * Wt[:, None]).T @ X
        o = [i for i in range(X.shape[1]) if i != j]
        coef = np.linalg.solve(H[np.ix_(o, o)], H[o, j])
        st = S[:, j] - S[:, o] @ coef
        T = st.sum() ** 2 / np.sum(st ** 2)
        rng = np.random.default_rng(seed)
        w = rng.choice([-1.0, 1.0], size=(B, len(st)))
        Tb = (w @ st) ** 2 / np.sum(st ** 2)
        return float((1 + np.sum(Tb >= T)) / (B + 1))

    def boot_ic(self, nome, B=9999, alfa=0.05):
        """IC por inversão do teste do score bootstrap (bisseção em cada lado)."""
        j = self.idx(nome)
        b, se = self.b[j], np.sqrt(self.V[j, j])
        lim = []
        for lado in (-1, 1):
            dentro, fora = b, b + lado * 2.5 * se
            k = 0
            while self.score_boot(nome, fora, B) >= alfa and k < 6:
                fora += lado * 1.5 * se
                k += 1
            for _ in range(12):
                meio = (dentro + fora) / 2
                if self.score_boot(nome, meio, B) >= alfa:
                    dentro = meio
                else:
                    fora = meio
            lim.append((dentro + fora) / 2)
        return {"p_boot": self.score_boot(nome, 0.0, B), "or_lo_boot": float(np.exp(lim[0])),
                "or_hi_boot": float(np.exp(lim[1]))}


class ModeloFE(Modelo):
    """Logística com efeitos fixos de hospital (absorvidos por complemento de Schur); erros robustos por hospital."""

    def __init__(self, formula, g, termo=None, maxit=60, tol=1e-10):
        # hospitais sem óbito (ou só com óbitos) não informam os coeficientes com efeitos fixos: excluídos
        kh = g.groupby("CNES").k.transform("sum")
        nh = g.groupby("CNES").n.transform("sum")
        self.hosp_sem_variacao = int(g.loc[(kh == 0) | (kh == nh), "CNES"].nunique())
        self.g = g[(kh > 0) & (kh < nh)].reset_index(drop=True)
        X = patsy.dmatrix(formula, self.g, return_type="dataframe").drop(columns="Intercept")
        self.X, self.cols = X, list(X.columns)
        gr = pd.factorize(self.g.CNES)[0]
        self.grupos, G = gr, gr.max() + 1
        W, k, n = X.values, self.g.k.values.astype(float), self.g.n.values.astype(float)
        kh, nh = np.bincount(gr, k, G), np.bincount(gr, n, G)
        a = np.log((kh + 0.5) / (nh - kh + 0.5))
        b = np.zeros(W.shape[1])
        ll_old = -np.inf
        for _ in range(maxit):
            eta = a[gr] + W @ b
            p = expit(eta)
            w = np.maximum(n * p * (1 - p), 1e-12)
            z = eta + (k - n * p) / w
            ah = np.maximum(np.bincount(gr, w, G), 1e-12)
            ch = np.zeros((G, W.shape[1]))
            np.add.at(ch, gr, W * w[:, None])
            zh = np.bincount(gr, w * z, G)
            Hs = (W * w[:, None]).T @ W - (ch / ah[:, None]).T @ ch
            rs = (W * w[:, None]).T @ z - (ch / ah[:, None]).T @ zh
            b = np.linalg.solve(Hs, rs)
            a = (zh - ch @ b) / ah
            p = np.clip(expit(a[gr] + W @ b), 1e-15, 1 - 1e-15)
            ll = np.sum(k * np.log(p) + (n - k) * np.log(1 - p))
            if abs(ll - ll_old) < tol * (1 + abs(ll)):
                break
            ll_old = ll
        w = n * p * (1 - p)
        ah = np.maximum(np.bincount(gr, w, G), 1e-12)
        ch = np.zeros((G, W.shape[1]))
        np.add.at(ch, gr, W * w[:, None])
        Hs = (W * w[:, None]).T @ W - (ch / ah[:, None]).T @ ch
        S = np.zeros((G, W.shape[1]))
        np.add.at(S, gr, W * (k - n * p)[:, None])
        Hi = np.linalg.inv(Hs)
        self.V = Hi @ (S.T @ S) @ Hi * G / (G - 1)
        self.b, self.a, self.ll = b, a, ll
        self.n_aih, self.n_hosp, self.k_param = int(n.sum()), int(G), W.shape[1] + G
        self.termo = termo
        self.peso_ident = None

    def dr(self, mudar):
        X1, X0 = mudar(self.X.copy())
        w = self.g.n.values / self.g.n.sum()
        off = self.a[self.grupos]
        p1, p0 = expit(off + X1.values @ self.b), expit(off + X0.values @ self.b)
        est = np.sum(w * (p1 - p0))
        grad = (w * p1 * (1 - p1)) @ X1.values - (w * p0 * (1 - p0)) @ X0.values
        se = np.sqrt(grad @ self.V @ grad)
        return {"dr_pp": 100 * est, "lo": 100 * (est - 1.96 * se), "hi": 100 * (est + 1.96 * se),
                "risco1_pct": 100 * np.sum(w * p1), "risco0_pct": 100 * np.sum(w * p0)}


def dr_continua(col, delta):
    def f(X):
        X1 = X.copy()
        X1[col] = X1[col] + delta
        return X1, X
    return f


def dr_dummy(col_on, cols_all):
    def f(X):
        X1, X0 = X.copy(), X.copy()
        for c in cols_all:
            X1[c], X0[c] = 0.0, 0.0
        if col_on is not None:
            X1[col_on] = 1.0
        return X1, X0
    return f


# ============================================================================ logística com intercepto aleatório (AGHQ)
class MistoAGHQ:
    """Logística binomial com intercepto aleatório de hospital, máxima verossimilhança por quadratura de
    Gauss-Hermite adaptativa (Q nós por hospital). Dados em células (k de n)."""

    def __init__(self, formula, g, Q=11):
        self.g = g.reset_index(drop=True)
        self.X = patsy.dmatrix(formula, self.g, return_type="dataframe")
        self.cols = list(self.X.columns)
        self.gr = pd.factorize(self.g.CNES)[0]
        self.hosp = pd.factorize(self.g.CNES)[1]
        self.G = self.gr.max() + 1
        self.k, self.n = self.g.k.values.astype(float), self.g.n.values.astype(float)
        self.Q = Q
        self.u = np.zeros(self.G)

    def _modo(self, eta, s2, iters=100):
        u = self.u.copy() if np.all(np.isfinite(self.u)) else np.zeros(self.G)
        for _ in range(iters):
            p = expit(eta + u[self.gr])
            g1 = np.bincount(self.gr, self.k - self.n * p, self.G) - u / s2
            h = -np.bincount(self.gr, self.n * p * (1 - p), self.G) - 1 / s2
            passo = np.clip(g1 / h, -2, 2)
            u = u - passo
            if np.max(np.abs(passo)) < 1e-9:
                break
        p = expit(eta + u[self.gr])
        h = -np.bincount(self.gr, self.n * p * (1 - p), self.G) - 1 / s2
        return u, 1 / np.sqrt(-h)

    def loglik_h(self, theta, Q=None, grad=False):
        """log-verossimilhança por hospital (AGHQ); com grad=True devolve também o gradiente da soma
        (esperanças a posteriori calculadas com os mesmos nós)."""
        Q = Q or self.Q
        b, s2 = theta[:-1], np.exp(2 * theta[-1])
        eta = self.X.values @ b
        u, s = self._modo(eta, s2)
        self.u = u
        z, w = np.polynomial.hermite.hermgauss(Q)
        L = np.empty((Q, self.G))
        res = [] if grad else None
        for q in range(Q):
            uq = u + np.sqrt(2) * s * z[q]
            e = eta + uq[self.gr]
            lp, l1p = -np.logaddexp(0, -e), -np.logaddexp(0, e)
            L[q] = np.bincount(self.gr, self.k * lp + (self.n - self.k) * l1p, self.G) - uq ** 2 / (2 * s2) + z[q] ** 2 + np.log(w[q])
            if grad:
                res.append(self.k - self.n * expit(e))
        lh = logsumexp(L, axis=0)
        out = lh + np.log(np.sqrt(2) * s) - 0.5 * np.log(2 * np.pi * s2)
        if not grad:
            return out
        pi = np.exp(L - lh)  # Q × G pesos a posteriori dos nós
        v = np.zeros(len(self.k))
        eu2 = np.zeros(self.G)
        for q in range(Q):
            v += pi[q][self.gr] * res[q]
            eu2 += pi[q] * (u + np.sqrt(2) * s * z[q]) ** 2
        gb = self.X.values.T @ v
        gs = np.sum(eu2 / s2 - 1)
        return out, np.r_[gb, gs]

    def nll(self, theta):
        return -np.sum(self.loglik_h(theta))

    def nll_grad(self, theta):
        ll, gr = self.loglik_h(theta, grad=True)
        return -np.sum(ll), -gr

    def ajustar(self):
        b0, _, _, _ = irls(self.X.values, self.k, self.n)
        th0 = np.r_[b0, 0.0]
        self.u = np.zeros(self.G)
        r = optimize.minimize(self.nll_grad, th0, jac=True, method="L-BFGS-B",
                              options={"maxiter": 5000, "gtol": 1e-6, "ftol": 1e-13})
        self.theta, self.ll, self.conv, self.otim_msg = r.x, -r.fun, bool(r.success), str(r.message)
        self.b, self.sigma = r.x[:-1], float(np.exp(r.x[-1]))
        self.grad_max = float(np.max(np.abs(r.jac)))
        # hessiana por diferenças finitas do gradiente analítico
        p = len(self.theta)
        H = np.empty((p, p))
        for i in range(p):
            hstep = 1e-4 * max(1, abs(self.theta[i]))
            tp, tm = self.theta.copy(), self.theta.copy()
            tp[i] += hstep
            tm[i] -= hstep
            H[:, i] = (self.nll_grad(tp)[1] - self.nll_grad(tm)[1]) / (2 * hstep)
        H = (H + H.T) / 2
        self.se_logsigma = float(np.sqrt(np.linalg.inv(H)[-1, -1]))
        self.loglik_h(self.theta)  # deixa self.u no modo final (preditores empíricos de Bayes)
        self.ll_laplace = -self.nll_q(1)
        self.ll_q21 = -self.nll_q(21)
        return self

    def nll_q(self, Q):
        return -np.sum(self.loglik_h(self.theta, Q))

    def resumo(self):
        s2 = self.sigma ** 2
        lo, hi = np.exp(np.log(self.sigma) - 1.96 * self.se_logsigma), np.exp(np.log(self.sigma) + 1.96 * self.se_logsigma)
        mor = lambda s: float(np.exp(np.sqrt(2) * s * stats.norm.ppf(0.75)))
        return {"sigma2": s2, "sigma2_lo": lo ** 2, "sigma2_hi": hi ** 2, "MOR": mor(self.sigma), "MOR_lo": mor(lo),
                "MOR_hi": mor(hi), "ICC_latente": s2 / (s2 + np.pi ** 2 / 3), "loglik": self.ll,
                "loglik_laplace": self.ll_laplace, "loglik_Q21": self.ll_q21, "convergiu": self.conv,
                "otimizador": self.otim_msg, "gradiente_max": self.grad_max,
                "n_celulas": len(self.g), "n_hospitais": int(self.G)}

    def pct_ajustada_hospital(self):
        """% sem CEC de cada hospital padronizada ao case-mix nacional (preditor empírico de Bayes)."""
        eta = self.X.values @ self.b
        wts = self.n / self.n.sum()
        out = np.array([np.sum(wts * expit(eta + uh)) for uh in self.u])
        return pd.DataFrame({"CNES": self.hosp, "u_eb": self.u, "pct_off_ajustada": 100 * out})


# ============================================================================ 0. fluxo, tabela 1 e matriz SIGTAP
def smd(a, b):
    a, b = np.asarray(a, float), np.asarray(b, float)
    s = np.sqrt((a.var(ddof=1) + b.var(ddof=1)) / 2)
    return float((b.mean() - a.mean()) / s) if s > 0 else 0.0


def tabela1(d):
    d = d.assign(_ge70=(d.idade_anos >= 70).astype(int), _obito=d.MORTE, _real=d.VAL_TOT * d.f_ipca)
    on, of = d[d.off == 0], d[d.off == 1]
    linhas = []

    def bin_(nome, col, cond=None):
        x = (d[col] == cond).astype(int) if cond is not None else d[col]
        xo, xf = x[d.off == 0], x[d.off == 1]
        linhas.append({"Characteristic": nome, "On-pump": f"{xo.sum():,} ({100 * xo.mean():.1f})",
                       "Off-pump": f"{xf.sum():,} ({100 * xf.mean():.1f})", "Overall": f"{x.sum():,} ({100 * x.mean():.1f})",
                       "SMD": round(smd(xo, xf), 3), "Missing, n": int(d[col].isna().sum())})

    def cont(nome, col, mediana=False):
        f = (lambda s: f"{s.median():.0f} ({s.quantile(.25):.0f}–{s.quantile(.75):.0f})") if mediana else \
            (lambda s: f"{s.mean():.1f} ({s.std():.1f})")
        linhas.append({"Characteristic": nome, "On-pump": f(on[col]), "Off-pump": f(of[col]), "Overall": f(d[col]),
                       "SMD": round(smd(on[col], of[col]), 3), "Missing, n": int(d[col].isna().sum())})

    linhas.append({"Characteristic": "AIH records, n", "On-pump": f"{len(on):,}", "Off-pump": f"{len(of):,}",
                   "Overall": f"{len(d):,}"})
    linhas.append({"Characteristic": "Hospitals, n", "On-pump": f"{on.CNES.nunique()}", "Off-pump": f"{of.CNES.nunique()}",
                   "Overall": f"{d.CNES.nunique()}"})
    cont("Age, years, mean (SD)", "idade_anos")
    bin_("Age ≥70 years, n (%)", "_ge70")
    bin_("Female, n (%)", "fem")
    if CARATER:
        bin_("Urgent admission, n (%)", "urg")
    for c in ["AMI (I21-I22)", "Unstable angina (I20.0)", "Other angina (I20.1-9)", "Other acute IHD (I24)", "I25 chronic IHD",
              "Other codes"]:
        bin_(f"Principal diagnosis: {c}, n (%)", "diag", c)
    for c in ["North", "Northeast", "Southeast", "South", "Center-West"]:
        bin_(f"Region of hospital: {c}, n (%)", "reg", c)
    for c in ["Public", "Nonprofit/philanthropic", "Private for-profit", "Unknown"]:
        bin_(f"Hospital legal status: {c}, n (%)", "natj", c)
    cont("Length of stay, days, median (IQR)", "DIAS_PERM", True)
    cont("ICU days, median (IQR)", "UTI_MES_TO", True)
    bin_("In-hospital death, n (%)", "_obito")
    if INI >= 2008:
        bin_("In-hospital death or transfer to another hospital, n (%)", "morte_transf")
    cont("AIH reimbursement, BRL Dec-2025, median (IQR)", "_real", True)
    salvar(pd.DataFrame(linhas), "T1_caracteristicas_por_tecnica")
    miss = pd.DataFrame({"variavel": ["DIAG_PRINC", "SEXO", "CAR_INT", "idade_anos", "NAT_JUR (2012–2025)", "UTI_MES_TO"],
                         "n_ausente_ou_invalido": [int(d.DIAG_PRINC.isna().sum()), int((~d.SEXO.str.strip().isin(["1", "2", "3"])).sum()),
                                                   int(d.CAR_INT.isna().sum()), int(d.idade_anos.isna().sum()),
                                                   int(d[d.ano >= 2012].NAT_JUR.isna().sum()), int(d.UTI_MES_TO.isna().sum())]})
    salvar(miss, "T1b_dados_ausentes")
    R["mort_global_pct"] = 100 * d.MORTE.mean()
    R["off_global_pct"] = 100 * d.off.mean()


def sigtap(d):
    moda = lambda s: float(s.round(2).mode().iloc[0]) if len(s) else np.nan
    a = d.groupby(["ano", "PROC_REA", "codigo"]).agg(
        n=("MORTE", "size"), obitos=("MORTE", "sum"), sh_modal=("VAL_SH", moda), sp_modal=("VAL_SP", moda),
        tot_medio_nominal=("VAL_TOT", "mean"), pct_com_complemento=("fed", lambda s: 100 * (s > 0).mean()),
        complemento_medio_nominal=("fed", "mean"), f=("f_ipca", "first")).reset_index()
    a["tot_medio_real_dez2025"] = a.tot_medio_nominal * a.f
    a["sh_sp_modal_real_dez2025"] = (a.sh_modal + a.sp_modal) * a.f
    a["mort_pct"] = 100 * a.obitos / a.n
    a["tecnica"] = np.where(a.codigo.str.startswith("Off-pump"), "Off-pump", "On-pump")
    a["pct_dentro_tecnica"] = 100 * a.n / a.groupby(["ano", "tecnica"]).n.transform("sum")
    a["pct_do_ano"] = 100 * a.n / a.groupby("ano").n.transform("sum")
    salvar(a.drop(columns="f").round(2), "S1_matriz_sigtap_empirica")


# ============================================================================ 1. adoção
def adocao(d):
    linhas = []
    for a, x in d.groupby("ano"):
        lo, hi = wilson(x.off.sum(), len(x))
        linhas.append({"ano": a, "n": len(x), "n_off": int(x.off.sum()), "pct_off": 100 * x.off.mean(), "lo": 100 * lo,
                       "hi": 100 * hi, "hospitais": x.CNES.nunique(),
                       "hospitais_com_off": x[x.off == 1].CNES.nunique()})
    s = pd.DataFrame(linhas)
    # % ajustada pelo case-mix (padronização marginal ao case-mix de todo o período)
    go = d.groupby(["CNES", "ano", "idade", "fem", "urg", "diag", "reg", "natj"]).agg(k=("off", "sum"), n=("off", "size")).reset_index()
    m = Modelo(f"C(ano) + {CASEMIX}", go)
    cols_ano = [c for c in m.cols if c.startswith("C(ano)")]
    adj = []
    for a in range(INI, FIM + 1):
        alvo = None if a == INI else f"C(ano)[T.{a}]"
        X = m.X.copy()
        for c in cols_ano:
            X[c] = 0.0
        if alvo:
            X[alvo] = 1.0
        w = m.g.n.values / m.g.n.sum()
        adj.append(100 * np.sum(w * expit(X.values @ m.b)))
    s["pct_off_ajustada_casemix"] = adj
    salvar(s.round(2), "A1_adocao_anual")
    # tendência: linear × spline (razão de verossimilhança)
    go["t"] = go.ano - INI
    X1 = patsy.dmatrix(f"t + {CASEMIX}", go, return_type="dataframe").values
    X2 = patsy.dmatrix(f"cr(t, df=3, constraints='center') + {CASEMIX}", go, return_type="dataframe").values
    k, n = go.k.values.astype(float), go.n.values.astype(float)
    _, _, _, l1 = irls(X1, k, n)
    _, _, _, l2 = irls(X2, k, n)
    ml = Modelo(f"t + {CASEMIX}", go)
    R["adocao_or_por_ano_linear"] = ml.orci("t")
    R["adocao_spline_vs_linear_LR"] = {"chi2": float(2 * (l2 - l1)), "gl": int(X2.shape[1] - X1.shape[1]),
                                       "p": float(stats.chi2.sf(2 * (l2 - l1), X2.shape[1] - X1.shape[1]))}
    # por região, natureza jurídica e período
    d = d.assign(periodo=periodo(d.ano))
    tabs = []
    for var, nome in [("reg", "Region"), ("natj", "Legal status")]:
        t = d.groupby([var, "periodo"]).agg(n=("off", "size"), n_off=("off", "sum")).reset_index()
        t["pct_off"] = 100 * t.n_off / t.n
        t = t.rename(columns={var: "grupo"}).assign(variavel=nome)
        tabs.append(t)
    # categoria do hospital pela % sem CEC em todo o período
    h = d.groupby("CNES").agg(n=("off", "size"), n_off=("off", "sum"))
    h["pct"] = 100 * h.n_off / h.n
    h["cat"] = pd.cut(h.pct, [-1, 0, 4.999999, 19.999999, 101], labels=["None (0%)", "Low (>0–<5%)", "Intermediate (5–<20%)",
                                                                       "High (≥20%)"]).astype(str)
    d = d.join(h.cat, on="CNES")
    t = d.groupby(["cat", "periodo"]).agg(n=("off", "size"), n_off=("off", "sum")).reset_index()
    t["pct_off"] = 100 * t.n_off / t.n
    tabs.append(t.rename(columns={"cat": "grupo"}).assign(variavel=f"Hospital off-pump category ({INI}–{FIM})"))
    tt = d.groupby("periodo").agg(n=("off", "size"), n_off=("off", "sum")).reset_index().assign(grupo="Brazil", variavel="Overall")
    tt["pct_off"] = 100 * tt.n_off / tt.n
    tabs.insert(0, tt)
    salvar(pd.concat(tabs)[["variavel", "grupo", "periodo", "n", "n_off", "pct_off"]].round(2), "A2_adocao_por_grupo_periodo")
    hc = h.groupby("cat").agg(hospitais=("n", "size"), aih=("n", "sum"), aih_off=("n_off", "sum")).reset_index()
    hc["pct_das_aih"] = 100 * hc.aih / hc.aih.sum()
    hc["pct_das_off"] = 100 * hc.aih_off / hc.aih_off.sum()
    salvar(hc.round(2), "A3_categorias_hospital")
    # T2: por período (hospitais, distribuição entre hospitais, concentração)
    linhas = []
    for per, x in d.groupby("periodo"):
        hh = x.groupby("CNES").agg(n=("off", "size"), n_off=("off", "sum"))
        hh["pct"] = 100 * hh.n_off / hh.n
        h10 = hh[hh.n >= 10]
        vol = hh.n_off.sort_values(ascending=False)
        sh = vol / vol.sum()
        linhas.append({"periodo": per, "aih": len(x), "pct_off": 100 * x.off.mean(), "hospitais": len(hh),
                       "hospitais_com_algum_off": int((hh.n_off > 0).sum()),
                       "hospitais_0pct_(>=10 AIH)": int((h10.pct == 0).sum()), "hospitais_>=10_aih": len(h10),
                       "pct_hosp_mediana_(>=10 AIH)": h10.pct.median(), "p25": h10.pct.quantile(.25),
                       "p75": h10.pct.quantile(.75), "p90": h10.pct.quantile(.9),
                       "top1_share_off_pct": 100 * sh.iloc[0], "top5_share_off_pct": 100 * sh.iloc[:5].sum(),
                       "top10_share_off_pct": 100 * sh.iloc[:10].sum(), "HHI_off": float((sh ** 2).sum()),
                       "n_efetivo_hosp_off": float(1 / (sh ** 2).sum())})
    salvar(pd.DataFrame(linhas).round(3), "T2_adocao_por_periodo")
    return h


# ============================================================================ 2. variação entre hospitais
def gini(x):
    x = np.sort(np.asarray(x, float))
    n = len(x)
    return float((2 * np.arange(1, n + 1) - n - 1) @ x / (n * x.sum()))


def variacao(d, h):
    go = d.groupby(["CNES", "ano", "idade", "fem", "urg", "diag"]).agg(k=("off", "sum"), n=("off", "size")).reset_index()
    log("modelo hierárquico M0")
    m0 = MistoAGHQ("C(ano)", go).ajustar()
    log("modelo hierárquico M1")
    m1 = MistoAGHQ(f"C(ano) + {CASEMIX}", go).ajustar()
    r0, r1 = m0.resumo(), m1.resumo()
    R["hier_M0_ano"], R["hier_M1_ano_casemix"] = r0, r1
    R["hier_pct_variancia_explicada_casemix"] = 100 * (r0["sigma2"] - r1["sigma2"]) / r0["sigma2"]
    tab = pd.DataFrame([dict(modelo="M0: year", **r0), dict(modelo="M1: year + case-mix", **r1)])
    salvar(tab.round(4), "V1_modelo_hierarquico")
    eb = m1.pct_ajustada_hospital().join(h, on="CNES")
    salvar(eb.round(3).sort_values("pct_off_ajustada", ascending=False), "V2_hospitais_pct_bruta_ajustada")
    linhas = []
    for nome, x in [("All hospitals", eb), ("Hospitals with ≥50 AIH", eb[eb.n >= 50]), ("Hospitals with ≥200 AIH", eb[eb.n >= 200])]:
        for med, col in [("Crude %", "pct"), ("Adjusted % (EB, national case-mix)", "pct_off_ajustada")]:
            q = x[col].quantile([.05, .1, .25, .5, .75, .9, .95])
            linhas.append({"hospitais": nome, "medida": med, "n_hosp": len(x), "P5": q[.05], "P10": q[.1], "P25": q[.25],
                           "mediana": q[.5], "P75": q[.75], "P90": q[.9], "P95": q[.95],
                           "pct_hosp_<1%": 100 * (x[col] < 1).mean(), "pct_hosp_>=20%": 100 * (x[col] >= 20).mean()})
    salvar(pd.DataFrame(linhas).round(2), "V3_distribuicao_entre_hospitais")
    # concentração
    conc = []
    for nome, col in [("Off-pump AIH", "n_off"), ("All CABG AIH", "n")]:
        v = h[col].sort_values(ascending=False)
        sh = v / v.sum()
        conc.append({"volume": nome, "hospitais_com_volume": int((v > 0).sum()), "top1_pct": 100 * sh.iloc[0],
                     "top5_pct": 100 * sh.iloc[:5].sum(), "top10_pct": 100 * sh.iloc[:10].sum(),
                     "top20_pct": 100 * sh.iloc[:20].sum(), "Gini": gini(v[v > 0]), "HHI": float((sh ** 2).sum()),
                     "n_efetivo": float(1 / (sh ** 2).sum())})
    nunca = h[h.n_off == 0]
    R["hosp_nunca_off"] = {"n": int(len(nunca)), "pct_aih": float(100 * nunca.n.sum() / h.n.sum()),
                           "n_com_>=50_aih": int((nunca.n >= 50).sum())}
    salvar(pd.DataFrame(conc).round(4), "V4_concentracao")
    # amplitude e estabilidade intra-hospital (hospital-anos com ≥10 CRVM)
    hy = d.groupby(["CNES", "ano"]).agg(n=("off", "size"), n_off=("off", "sum")).reset_index()
    hy["pct"] = 100 * hy.n_off / hy.n
    hy10 = hy[hy.n >= 10]
    amp = hy10.groupby("CNES").pct.agg(["min", "max", "count"])
    amp = amp[amp["count"] >= 2]
    amp["amplitude"] = amp["max"] - amp["min"]
    nx = hy10.assign(ano=hy10.ano - 1).rename(columns={"pct": "pct_seg", "n": "n_seg"})[["CNES", "ano", "pct_seg", "n_seg"]]
    par = hy10.merge(nx, on=["CNES", "ano"])
    est = {"hospitais_>=2_anos": int(len(amp)), "amplitude_mediana_pp": float(amp.amplitude.median()),
           "amplitude_p75": float(amp.amplitude.quantile(.75)), "amplitude_p90": float(amp.amplitude.quantile(.9)),
           "hosp_amplitude_>=5pp": int((amp.amplitude >= 5).sum()), "hosp_amplitude_>=10pp": int((amp.amplitude >= 10).sum()),
           "hosp_amplitude_>=20pp": int((amp.amplitude >= 20).sum()), "pares_anos_consecutivos": int(len(par)),
           "corr_pearson_anos_consecutivos": float(par[["pct", "pct_seg"]].corr().iloc[0, 1]),
           "corr_spearman_anos_consecutivos": float(par[["pct", "pct_seg"]].corr("spearman").iloc[0, 1]),
           "mudanca_absoluta_mediana_pp": float((par.pct_seg - par.pct).abs().median()),
           "pct_pares_mudanca_<5pp": float(100 * ((par.pct_seg - par.pct).abs() < 5).mean())}
    # decomposição da variância (ponderada) entre e dentro de hospitais
    w = hy10.n
    mg = np.average(hy10.pct, weights=w)
    mh = hy10.groupby("CNES").apply(lambda x: np.average(x.pct, weights=x.n), include_groups=False)
    tot = np.average((hy10.pct - mg) ** 2, weights=w)
    dentro = np.average((hy10.pct - hy10.CNES.map(mh)) ** 2, weights=w)
    est["pct_variancia_dentro_hospital"] = float(100 * dentro / tot)
    R["estabilidade_intra_hospital"] = est
    salvar(pd.DataFrame([est]).T.reset_index().rename(columns={"index": "medida", 0: "valor"}), "V5_estabilidade_intra_hospital")


# ============================================================================ 3. mortalidade no tempo
def wald(m, nomes):
    j = [m.idx(c) for c in nomes]
    b, V = m.b[j], m.V[np.ix_(j, j)]
    q = float(b @ np.linalg.solve(V, b))
    return {"chi2": q, "gl": len(j), "p": float(stats.chi2.sf(q, len(j)))}


def mortalidade_tempo(d, g):
    linhas = []
    for (a, off), x in d.groupby(["ano", "off"]):
        lo, hi = wilson(x.MORTE.sum(), len(x))
        linhas.append({"ano": a, "tecnica": "Off-pump" if off else "On-pump", "n": len(x), "obitos": int(x.MORTE.sum()),
                       "mort_pct": 100 * x.MORTE.mean(), "lo": 100 * lo, "hi": 100 * hi})
    for a, x in d.groupby("ano"):
        lo, hi = wilson(x.MORTE.sum(), len(x))
        linhas.append({"ano": a, "tecnica": "Overall", "n": len(x), "obitos": int(x.MORTE.sum()),
                       "mort_pct": 100 * x.MORTE.mean(), "lo": 100 * lo, "hi": 100 * hi})
    s = pd.DataFrame(linhas)
    # ajustada: efeitos fixos de ano + case-mix + hospital (região, natureza), padronizada ao case-mix do período todo
    aj = []
    for nome, gx in [("Overall", g), ("On-pump", g[g.off == 0]), ("Off-pump", g[g.off == 1])]:
        m = Modelo(f"C(ano) + {CASEMIX} + {HOSP}", gx)
        cols_ano = [c for c in m.cols if c.startswith("C(ano)")]
        Xb = patsy.build_design_matrices([m.X.design_info], g, return_type="dataframe")[0]
        w = g.n.values / g.n.sum()
        for a in range(INI, FIM + 1):
            X = Xb.copy()
            for c in cols_ano:
                X[c] = 0.0
            if a > INI:
                X[f"C(ano)[T.{a}]"] = 1.0
            aj.append({"ano": a, "tecnica": nome, "mort_ajustada_pct": 100 * np.sum(w * expit(X.values @ m.b))})
        gx = gx.assign(t=gx.ano - INI)
        ml = Modelo(f"t + {CASEMIX} + {HOSP}", gx)
        X1 = ml.X.values
        X2 = patsy.dmatrix(f"cr(t, df=3, constraints='center') + {CASEMIX} + {HOSP}", gx.reset_index(drop=True), return_type="dataframe").values
        _, _, _, l2 = irls(X2, gx.k.values.astype(float), gx.n.values.astype(float))
        R[f"mort_tendencia_{nome}"] = {"or_por_ano": ml.orci("t"), "LR_spline_vs_linear": float(2 * (l2 - ml.ll)),
                                       "gl": int(X2.shape[1] - X1.shape[1]),
                                       "p_nao_linear": float(stats.chi2.sf(2 * (l2 - ml.ll), X2.shape[1] - X1.shape[1]))}
    s = s.merge(pd.DataFrame(aj), on=["ano", "tecnica"])
    salvar(s.round(3), "M1_mortalidade_anual_por_tecnica")
    # OR sem × com CEC por período e interação
    gp = g.assign(periodo=periodo(g.ano))
    linhas = []
    for per, x in gp.groupby("periodo"):
        m = Modelo(f"C(ano) + {CASEMIX} + {HOSP} + off", x)
        o = m.orci("off")
        dr = m.dr(dr_dummy("off", ["off"]))
        linhas.append({"periodo": per, "aih": int(x.n.sum()), "pct_off": 100 * x.loc[x.off == 1, "n"].sum() / x.n.sum(),
                       "mort_on_pct": 100 * x.loc[x.off == 0, "k"].sum() / x.loc[x.off == 0, "n"].sum(),
                       "mort_off_pct": 100 * x.loc[x.off == 1, "k"].sum() / x.loc[x.off == 1, "n"].sum(),
                       "or_ajustado": o["or"], "lo": o["lo"], "hi": o["hi"], "p": o["p"], "dr_pp": dr["dr_pp"],
                       "dr_lo": dr["lo"], "dr_hi": dr["hi"]})
    salvar(pd.DataFrame(linhas).round(4), "M2_or_off_por_periodo")
    mi = Modelo(f"C(ano) + {CASEMIX} + {HOSP} + off * C(periodo)", gp)
    R["interacao_off_periodo_wald"] = wald(mi, [c for c in mi.cols if c.startswith("off:")])


def comparacao_brasil(d):
    def crude(x):
        return {"aih": len(x), "obitos": int(x.MORTE.sum()), "mort_pct": round(100 * x.MORTE.mean(), 2)}
    ref = [
        {"fonte": "Piegas 2009, Arq Bras Cardiol (PMID 20084319)", "periodo": "2005–2007",
         "dados": "SIH/SUS (TabNet), CRVM isolada, todas as idades, 16 hospitais de muito baixo volume excluídos",
         "n_publicado": 63272, "mort_publicada_pct": 6.22,
         **({"este_estudo": "2005–2007, adultos, ano de internação", **crude(d[d.ano.between(2005, 2007)])} if INI <= 2005 else
            {"este_estudo": "sem sobreposição; 2008–2010", **crude(d[d.ano <= 2010])})},
        {"fonte": "Khalil 2021, J Card Surg (PMID 33469979)", "periodo": "2008–2017",
         "dados": "SIH/SUS (TabNet), CRVM isolada, todas as idades, ano de processamento", "n_publicado": 226697,
         "mort_publicada_pct": 5.7, "este_estudo": "2008–2017, adultos, ano de internação", **crude(d[d.ano.between(2008, 2017)])},
        {"fonte": "Martins 2026, J Cardiothorac Surg (PMID 42260539)", "periodo": "2008–2024",
         "dados": "SIH/SUS, painel UF-ano (379 observações); resumo reporta mortalidade prevista 6,8% → 5,7% por +1 DP de "
                  "eficiência de desenvolvimento, não a mortalidade bruta global", "n_publicado": None,
         "mort_publicada_pct": None, "este_estudo": "2008–2024, adultos", **crude(d[d.ano.between(2008, 2024)])},
        {"fonte": "Gomes 2017, BYPASS, Braz J Cardiovasc Surg (PMID 28492786)", "periodo": "registro prospectivo",
         "dados": "17 centros, 83% SUS; mortalidade da CRVM até o 7º dia pós-operatório (não hospitalar total)",
         "n_publicado": None, "mort_publicada_pct": 2.6, "este_estudo": f"{INI}–{FIM} (não comparável: desfecho diferente)",
         **crude(d)},
    ]
    salvar(pd.DataFrame(ref), "M3_comparacao_literatura_brasileira")


# ============================================================================ 4. estimando institucional
def pesos_identificacao(d):
    """Peso de cada hospital na identificação do modelo com efeitos fixos: Σ n_hy (x_hy − x̄_h)²."""
    hy = d.groupby(["CNES", "ano"]).agg(n=("off", "size"), n_off=("off", "sum")).reset_index()
    hy["x"] = 100 * hy.n_off / hy.n
    hy["xm"] = hy.groupby("CNES").apply(lambda z: np.average(z.x, weights=z.n), include_groups=False).reindex(hy.CNES).values
    p = (hy.n * (hy.x - hy.xm) ** 2).groupby(hy.CNES).sum().sort_values(ascending=False)
    p = p / p.sum()
    return p


def institucional(d, g, B):
    gi = g[g.expo.notna()].assign(e10=lambda x: x.expo / 10)
    form = f"C(ano) + {CASEMIX} + {HOSP}"
    linhas = []
    log("institucional: contínuo")
    m = Modelo(f"{form} + e10", gi)
    o, dr = m.orci("e10"), m.dr(dr_continua("e10", 1))
    bo = m.boot_ic("e10", B)
    linhas.append({"modelo": "Institutional, continuous (+10 pp), no hospital FE [PRIMARY]", "termo": "+10 pp", **o, **bo,
                   "dr_pp": dr["dr_pp"], "dr_lo": dr["lo"], "dr_hi": dr["hi"], "aih": m.n_aih, "hospitais": m.n_hosp})
    log("institucional: categorias")
    cats = [c for c in EXPO_LAB if c != "0%"]
    mc = Modelo(f"{form} + C(expo_cat, Treatment('0%'))", gi)
    nomes = [f"C(expo_cat, Treatment('0%'))[T.{c}]" for c in cats]
    for c, nm in zip(cats, nomes):
        o = mc.orci(nm)
        dr = mc.dr(dr_dummy(nm, nomes))
        linhas.append({"modelo": "Institutional, categories vs 0%, no hospital FE", "termo": c, **o, "dr_pp": dr["dr_pp"],
                       "dr_lo": dr["lo"], "dr_hi": dr["hi"], "aih": int(gi.loc[gi.expo_cat == c, "n"].sum()),
                       "hospitais": int(gi.loc[gi.expo_cat == c, "CNES"].nunique())})
    R["institucional_categorias_wald"] = wald(mc, nomes)
    log("institucional: efeitos fixos de hospital")
    mf = ModeloFE(f"C(ano) + {CASEMIX} + e10", gi)
    o, dr = mf.orci("e10"), mf.dr(dr_continua("e10", 1))
    linhas.append({"modelo": "Institutional, continuous (+10 pp), hospital FE [SENSITIVITY]", "termo": "+10 pp", **o,
                   "dr_pp": dr["dr_pp"], "dr_lo": dr["lo"], "dr_hi": dr["hi"], "aih": mf.n_aih, "hospitais": mf.n_hosp})
    mfc = ModeloFE(f"C(ano) + {CASEMIX} + C(expo_cat, Treatment('0%'))", gi)
    for c, nm in zip(cats, nomes):
        o = mfc.orci(nm)
        linhas.append({"modelo": "Institutional, categories vs 0%, hospital FE", "termo": c, **o})
    R["fe_hosp_sem_variacao_desfecho_excluidos"] = mf.hosp_sem_variacao
    salvar(pd.DataFrame(linhas).round(4), "M4_institucional")
    # influência no modelo com efeitos fixos
    log("institucional: influência")
    p = pesos_identificacao(d)
    R["identificacao_n_efetivo_hospitais"] = float(1 / (p ** 2).sum())
    R["identificacao_top5_top10_pct"] = [float(100 * p.iloc[:5].sum()), float(100 * p.iloc[:10].sum())]
    inf = [{"retirado": "none", "peso_ident_pct": 0, **mf.orci("e10")}]
    for i, cnes in enumerate(p.index[:20]):
        mm = ModeloFE(f"C(ano) + {CASEMIX} + e10", gi[gi.CNES != cnes])
        inf.append({"retirado": f"rank {i + 1}: CNES {cnes}", "peso_ident_pct": 100 * p.iloc[i], **mm.orci("e10")})
    for k in (5, 10, 20):
        mm = ModeloFE(f"C(ano) + {CASEMIX} + e10", gi[~gi.CNES.isin(p.index[:k])])
        inf.append({"retirado": f"top {k} together", "peso_ident_pct": 100 * p.iloc[:k].sum(), **mm.orci("e10")})
    # e também no modelo principal (sem efeitos fixos): retirar os maiores hospitais por volume
    vol = d.CNES.value_counts()
    for k in (1, 5, 10):
        mm = Modelo(f"{form} + e10", gi[~gi.CNES.isin(vol.index[:k])])
        inf.append({"retirado": f"PRIMARY model without top {k} hospitals by CABG volume", "peso_ident_pct": None, **mm.orci("e10")})
    salvar(pd.DataFrame(inf).round(4), "M5_influencia")
    pi = p.reset_index()
    pi.columns = ["CNES", "peso_identificacao"]
    pi["acumulado"] = pi.peso_identificacao.cumsum()
    salvar(pi.round(5), "M5b_pesos_identificacao_hospital")


# ============================================================================ 5. estimando individual
def evalue(rr, lo, hi):
    def e(x):
        x = 1 / x if x < 1 else x
        return x + np.sqrt(x * (x - 1))
    lim = hi if rr < 1 else lo
    return {"E_ponto": float(e(rr)), "E_limite_IC": float(1.0 if (lo <= 1 <= hi) else e(lim))}


def individual(d, g, B):
    linhas = []
    for nome, form, gx, fe in [
        ("Crude", "off", g, False),
        ("Adjusted: year + case-mix", f"C(ano) + {CASEMIX} + off", g, False),
        ("Adjusted: + region + legal status [PRIMARY individual]", f"C(ano) + {CASEMIX} + {HOSP} + off", g, False),
        ("Adjusted + hospital FE", f"C(ano) + {CASEMIX} + off", g, True),
    ] + ([] if not CARATER else [
        ("Elective only (adjusted + region + legal status)", f"C(ano) + {CASEMIX.replace(' + urg', '')} + {HOSP} + off", g[g.urg == 0], False),
        ("Urgent only (adjusted + region + legal status)", f"C(ano) + {CASEMIX.replace(' + urg', '')} + {HOSP} + off", g[g.urg == 1], False),
        ("Elective only + hospital FE", f"C(ano) + {CASEMIX.replace(' + urg', '')} + off", g[g.urg == 0], True),
        ("Urgent only + hospital FE", f"C(ano) + {CASEMIX.replace(' + urg', '')} + off", g[g.urg == 1], True)]):
        m = (ModeloFE if fe else Modelo)(form, gx)
        o, dr = m.orci("off"), m.dr(dr_dummy("off", ["off"]))
        lin = {"modelo": nome, **o, "dr_pp": dr["dr_pp"], "dr_lo": dr["lo"], "dr_hi": dr["hi"],
               "risco_off_pct": dr["risco1_pct"], "risco_on_pct": dr["risco0_pct"], "aih": m.n_aih, "hospitais": m.n_hosp}
        if "PRIMARY" in nome:
            lin.update(m.boot_ic("off", B))
            lin.update(evalue(o["or"], o["lo"], o["hi"]))
        linhas.append(lin)
    salvar(pd.DataFrame(linhas).round(4), "M6_individual")


# ============================================================================ 6. sensibilidades
def sensibilidades(d, g):
    cen = [(f"Primary ({INI}–{FIM})", g),
           (f"{INI}–2022 only (excluding 2023–2025)", g[g.ano <= 2022])]
    if INI < 2008:
        cen.append(("2008–2025 only (excluding 2003–2007)", g[g.ano >= 2008]))
    cen.append(("Excluding 2020–2021", g[~g.ano.isin([2020, 2021])]))
    if CARATER:
        cen.append(("Elective only", g[g.urg == 0]))
    cen += [("Hospital-years with ≥10 CABG", g[g.hy_n >= 10]),
            ("Hospitals with stable graft coding (|Δ% 1-graft| <10 pp)", g[g.cod_estavel == 1])]
    if INI >= 2008:  # COBRANCA tem outra codificação antes de 2008
        cen.append(("Outcome: death or transfer (COBRANCA 31)", celulas(d, "morte_transf")))
    linhas = []
    for nome, gx in cen:
        cm = CASEMIX.replace(" + urg", "") if nome == "Elective only" else CASEMIX
        gi = gx[gx.expo.notna()].assign(e10=lambda x: x.expo / 10)
        mi = Modelo(f"C(ano) + {cm} + {HOSP} + e10", gi)
        mo = Modelo(f"C(ano) + {cm} + {HOSP} + off", gx)
        oi, di = mi.orci("e10"), mi.dr(dr_continua("e10", 1))
        oo, do = mo.orci("off"), mo.dr(dr_dummy("off", ["off"]))
        linhas.append({"cenario": nome, "aih": mo.n_aih, "hospitais": mo.n_hosp, "eventos": int(gx.k.sum()),
                       "inst_or_10pp": oi["or"], "inst_lo": oi["lo"], "inst_hi": oi["hi"], "inst_dr_pp": di["dr_pp"],
                       "inst_dr_lo": di["lo"], "inst_dr_hi": di["hi"], "indiv_or": oo["or"], "indiv_lo": oo["lo"],
                       "indiv_hi": oo["hi"], "indiv_dr_pp": do["dr_pp"], "indiv_dr_lo": do["lo"], "indiv_dr_hi": do["hi"]})
        log(f"sensibilidade: {nome}")
    salvar(pd.DataFrame(linhas).round(4), "M7_sensibilidades")


# ============================================================================ 7. variável instrumental (exploratória)
def ols_cluster(y, X, gr):
    XtX = X.T @ X
    b = np.linalg.solve(XtX, X.T @ y)
    e = y - X @ b
    S = np.zeros((gr.max() + 1, X.shape[1]))
    np.add.at(S, gr, X * e[:, None])
    G = S.shape[0]
    Ai = np.linalg.inv(XtX)
    V = Ai @ (S.T @ S) @ Ai * G / (G - 1)
    return b, V


def tsls(y, x, z, W, gr):
    Zm, Xm = np.column_stack([z, W]), np.column_stack([x, W])
    Pz = np.linalg.solve(Zm.T @ Zm, Zm.T @ Xm)
    Xh = Zm @ Pz
    Ai = np.linalg.inv(Xh.T @ Xm)
    b = Ai @ (Xh.T @ y)
    e = y - Xm @ b
    S = np.zeros((gr.max() + 1, Xm.shape[1]))
    np.add.at(S, gr, Xh * e[:, None])
    G = S.shape[0]
    V = Ai @ (S.T @ S) @ Ai.T * G / (G - 1)
    return float(b[0]), float(np.sqrt(V[0, 0]))


def demean(M, gr):
    G = gr.max() + 1
    cnt = np.bincount(gr, minlength=G)
    S = np.zeros((G, M.shape[1]))
    np.add.at(S, gr, M)
    return M - (S / cnt[:, None])[gr]


def instrumental(d):
    d = d.copy()
    d["z_loo"] = np.where(d.hy_n >= 10, d.expo, np.nan)
    linhas, fs_sub = [], []
    for inst in ["z_lag", "z_loo"]:
        x = d[d[inst].notna()].reset_index(drop=True)
        gr = pd.factorize(x.CNES)[0]
        Wdf = patsy.dmatrix(f"C(ano) + {CASEMIX} + {HOSP}", x, return_type="dataframe")
        for fe in (False, True):
            W = Wdf.values
            y, off, z = x.MORTE.values.astype(float), x.off.values.astype(float), x[inst].values / 10
            if fe:
                W = demean(Wdf.drop(columns="Intercept").values, gr)
                W = W[:, W.std(axis=0) > 1e-10]
                y, off, z = demean(np.column_stack([y, off, z]), gr).T
            bf, Vf = ols_cluster(off, np.column_stack([z, W]), gr)
            Fst = float(bf[0] ** 2 / Vf[0, 0])
            biv, seiv = tsls(y, off, z, W, gr)
            bo, Vo = ols_cluster(y, np.column_stack([off, W]), gr)
            linhas.append({"instrumento": "prior-year hospital % off-pump" if inst == "z_lag" else "same-year leave-one-out % (hospital-years ≥10)",
                           "efeitos_fixos_hospital": fe, "aih": len(x), "hospitais": int(gr.max() + 1),
                           "primeira_fase_coef_por_10pp": float(bf[0]), "primeira_fase_F_robusto": Fst,
                           "iv_dr_pp": 100 * biv, "iv_lo": 100 * (biv - 1.96 * seiv), "iv_hi": 100 * (biv + 1.96 * seiv),
                           "mqo_dr_pp": 100 * float(bo[0]), "mqo_lo": 100 * float(bo[0] - 1.96 * np.sqrt(Vo[0, 0])),
                           "mqo_hi": 100 * float(bo[0] + 1.96 * np.sqrt(Vo[0, 0]))})
            log(f"IV {inst} FE={fe}")
        if inst == "z_lag":
            # monotonicidade: 1ª fase por subgrupo (sem efeitos fixos)
            subs = {"Age <65": x.idade_anos < 65, "Age ≥65": x.idade_anos >= 65, "Male": x.fem == 0, "Female": x.fem == 1,
                    **({"Elective": x.urg == 0, "Urgent": x.urg == 1} if CARATER else {}), "AMI": x.diag == "AMI (I21-I22)",
                    "Non-AMI": x.diag != "AMI (I21-I22)"}
            for r in ["North", "Northeast", "Southeast", "South", "Center-West"]:
                subs[f"Region {r}"] = x.reg == r
            for nome, msk in subs.items():
                xs = x[msk.values]
                grs = pd.factorize(xs.CNES)[0]
                Ws = Wdf.values[msk.values]
                Ws = Ws[:, Ws.std(axis=0) > 1e-10]
                Ws = np.column_stack([np.ones(len(xs)), Ws])
                b, V = ols_cluster(xs.off.values.astype(float), np.column_stack([xs[inst].values / 10, Ws]), grs)
                fs_sub.append({"subgrupo": nome, "aih": len(xs), "coef_por_10pp": float(b[0]),
                               "lo": float(b[0] - 1.96 * np.sqrt(V[0, 0])), "hi": float(b[0] + 1.96 * np.sqrt(V[0, 0]))})
            # balanço por categoria do instrumento
            x["z_cat"] = pd.cut(x[inst], EXPO_CAT, labels=EXPO_LAB).astype(str)
            bal = x.groupby("z_cat").agg(aih=("off", "size"), hospitais=("CNES", "nunique"), pct_off=("off", "mean"),
                                         idade_media=("idade_anos", "mean"), pct_fem=("fem", "mean"), pct_urg=("urg", "mean"),
                                         pct_iam=("diag", lambda s: (s == "AMI (I21-I22)").mean()),
                                         pct_angina_instavel=("diag", lambda s: (s == "Unstable angina (I20.0)").mean()),
                                         pct_sudeste=("reg", lambda s: (s == "Southeast").mean()),
                                         mort_bruta=("MORTE", "mean")).reindex(EXPO_LAB).reset_index()
            for c in ["pct_off", "pct_fem", "pct_urg", "pct_iam", "pct_angina_instavel", "pct_sudeste", "mort_bruta"]:
                bal[c] *= 100
            salvar(bal.round(2), "IV2_balanco_por_categoria_instrumento")
            fv = x[inst].describe(percentiles=[.1, .25, .5, .75, .9])
            R["iv_distribuicao_instrumento_lag"] = {k: float(v) for k, v in fv.items()}
    salvar(pd.DataFrame(linhas).round(4), "IV1_variavel_instrumental")
    salvar(pd.DataFrame(fs_sub).round(4), "IV3_monotonicidade_primeira_fase")


# ============================================================================ Excel e main
LEIA = [
    ("Leia-me", "Conteúdo e especificação: artigo/plano_artigo_cec_v2.md (seção 'Especificação fixada antes de rodar')."),
    ("T1_caracteristicas_por_tecnica", "Tabela 1: características por técnica registrada, diferenças padronizadas (SMD) e ausentes."),
    ("T1b_dados_ausentes", "Dados ausentes ou inválidos por variável."),
    ("T2_adocao_por_periodo", "Tabela 2: adoção por período; distribuição entre hospitais (≥10 AIH) e concentração do volume sem CEC."),
    ("A1_adocao_anual", "% sem CEC por ano com IC de Wilson e % ajustada pelo case-mix (padronização marginal)."),
    ("A2_adocao_por_grupo_periodo", "% sem CEC por região, natureza jurídica e categoria de hospital, por período."),
    ("A3_categorias_hospital", "Hospitais por categoria de % sem CEC no período todo."),
    ("V1_modelo_hierarquico", "Logística com intercepto aleatório de hospital (AGHQ, 11 nós): variância, MOR, ICC; M0 ano, M1 ano + case-mix."),
    ("V2_hospitais_pct_bruta_ajustada", "% sem CEC bruta e ajustada (preditor EB, case-mix nacional) por hospital."),
    ("V3_distribuicao_entre_hospitais", "Distribuição da % sem CEC entre hospitais (bruta e ajustada)."),
    ("V4_concentracao", "Concentração do volume (top N, Gini, HHI, nº efetivo de hospitais)."),
    ("V5_estabilidade_intra_hospital", "Amplitude e estabilidade ano a ano da % sem CEC dentro dos hospitais."),
    ("M1_mortalidade_anual_por_tecnica", "Mortalidade hospitalar bruta (IC Wilson) e ajustada por ano, global e por técnica."),
    ("M2_or_off_por_periodo", "Associação individual sem × com CEC por período (OR e diferença de risco ajustados)."),
    ("M3_comparacao_literatura_brasileira", "Mortalidade bruta deste estudo nas janelas da literatura administrativa brasileira e BYPASS."),
    ("M4_institucional", "Estimando institucional (principal): % sem CEC do hospital no ano (leave-one-out); OR e diferença de risco; wild cluster bootstrap; efeitos fixos de hospital."),
    ("M5_influencia", "Influência: retirada dos hospitais de maior peso de identificação (modelo com efeitos fixos) e dos maiores por volume (modelo principal)."),
    ("M5b_pesos_identificacao_hospital", "Peso de identificação de cada hospital no modelo com efeitos fixos."),
    ("M6_individual", "Estimando individual (secundário): técnica registrada × óbito; E-value."),
    ("M7_sensibilidades", "Sensibilidades pré-especificadas para os estimandos institucional e individual."),
    ("IV1_variavel_instrumental", "Variável instrumental exploratória (2SLS, probabilidade linear), instrumentos defasado e mesmo ano."),
    ("IV2_balanco_por_categoria_instrumento", "Balanço de covariáveis por categoria do instrumento defasado."),
    ("IV3_monotonicidade_primeira_fase", "1ª fase por subgrupo (monotonicidade)."),
    ("S1_matriz_sigtap_empirica", "Matriz SIGTAP empírica por ano e código: valor modal de SP (≈ valor de tabela; o SH modal inclui diárias/UTI e é só aproximado), valores médios, complemento federal."),
    ("S2_mensal_codigo_complemento", "Série mensal (mês de internação, 2021–2025) por código, % 1 enxerto no com CEC, % sem CEC e % de AIH com complemento federal (auditoria A2)."),
    ("S3_contagens_por_codigo_ano", "Contagens anuais por código SIGTAP e % 1 enxerto dentro de cada técnica (auditoria A1b)."),
    ("S4_fora_SUS_por_ano", "EXPLORATÓRIO: CRVM fora do SUS (CIHA) × SUS por ano de internação, 2011–2025 (volume, % sem CEC, mortalidade). Ver ciha/LEIA-ME.md."),
    ("S5_fora_SUS_por_fonte", "EXPLORATÓRIO: SUS × fora do SUS por fonte de pagamento (plano privado, plano público, particular...)."),
    ("S6_fora_SUS_regiao", "EXPLORATÓRIO: SUS × fora do SUS por região do hospital."),
    ("S7_fora_SUS_modelos", "EXPLORATÓRIO: sem × com CEC fora do SUS e no SUS; fora do SUS × SUS; chance de sem CEC no mesmo hospital."),
    ("S8_Brasil_total_por_ano", "EXPLORATÓRIO: Brasil total (SUS + fora do SUS), 2011–2025."),
    ("S9_tendencias_SUS_x_fora", "EXPLORATÓRIO: séries anuais SUS × fora do SUS (todos os hospitais e painel estável ≥10 de 15 anos)."),
    ("S10_tendencias_modelos", "EXPLORATÓRIO: tendência por ano de % sem CEC e mortalidade em cada sistema e teste da diferença de tendência."),
    ("S11_tendencias_ajustadas", "EXPLORATÓRIO: % sem CEC e mortalidade ajustadas por ano e sistema."),
    ("S12_cobertura_CIHA_ANS", "EXPLORATÓRIO: CRVM CIHA por 100 mil beneficiários de plano (ANS) × CRVM SUS por 100 mil sem plano (captação máxima da CIHA)."),
]
DIAG = {"S2_mensal_codigo_complemento": "A2_mensal_por_internacao", "S3_contagens_por_codigo_ano": "A1b_contagens_por_codigo_wide"}
EXTERNO = {"S4_fora_SUS_por_ano": "ciha/resultados/C1_por_ano_e_base", "S5_fora_SUS_por_fonte": "ciha/resultados/C2_sus_x_fora_do_sus_por_fonte",
           "S6_fora_SUS_regiao": "ciha/resultados/C3_por_regiao", "S7_fora_SUS_modelos": "ciha/resultados/C5_modelos",
           "S8_Brasil_total_por_ano": "ciha/resultados/C6_brasil_total_por_ano",
           "S9_tendencias_SUS_x_fora": "ciha/resultados/T1_series_anuais", "S10_tendencias_modelos": "ciha/resultados/T2_modelos_tendencia",
           "S11_tendencias_ajustadas": "ciha/resultados/T3_ajustadas_por_ano", "S12_cobertura_CIHA_ANS": "ciha/resultados/C7_cobertura_ciha_ans"}


def excel(caminho):
    with pd.ExcelWriter(caminho, engine="xlsxwriter") as xw:
        wb = xw.book
        cab = wb.add_format({"bold": True, "bg_color": "#DDE6F0", "text_wrap": True, "valign": "top", "border": 1})
        leia = pd.DataFrame(LEIA, columns=["aba", "conteúdo"])
        leia.to_excel(xw, sheet_name="Leia-me", index=False)
        ws = xw.sheets["Leia-me"]
        ws.set_column(0, 0, 38)
        ws.set_column(1, 1, 120)
        res = pd.json_normalize(R, sep=".").T.reset_index()
        res.columns = ["resultado", "valor"]
        res["valor"] = res.valor.astype(str)
        for aba, _ in LEIA[1:]:
            arq = (os.path.join(RAIZ, "resultados_diag", DIAG[aba] + ".csv") if aba in DIAG else
                   os.path.join(RAIZ, "..", EXTERNO[aba] + ".csv") if aba in EXTERNO else os.path.join(OUT, aba + ".csv"))
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
        res.to_excel(xw, sheet_name="Resultados_chave", index=False)
        xw.sheets["Resultados_chave"].set_column(0, 0, 70)
        xw.sheets["Resultados_chave"].set_column(1, 1, 120)


def main():
    global INI, FIM, CARATER, CASEMIX, MICRO, OUT
    ap = argparse.ArgumentParser()
    ap.add_argument("--boot", type=int, default=9999)
    ap.add_argument("--ini", type=int, default=2008)
    ap.add_argument("--fim", type=int, default=2025)
    ap.add_argument("--sem-carater", action="store_true", help="(padrão; mantido por compatibilidade)")
    ap.add_argument("--com-carater", action="store_true",
                    help="só para reproduzir a rodada histórica; o caráter foi excluído de todas as análises (15/out/2026)")
    ap.add_argument("--micro", default=MICRO)
    ap.add_argument("--tag", default="", help="sufixo da pasta de resultados e da planilha")
    ap.add_argument("--so", default="", help="rodar só estas etapas (vírgulas): t1,adocao,variacao,tempo,inst,indiv,sens,iv")
    a = ap.parse_args()
    INI, FIM, CARATER, MICRO = a.ini, a.fim, a.com_carater, a.micro
    if INI < 2008 and CARATER:
        raise SystemExit("caráter da internação não é comparável antes de 2008: use --sem-carater")
    if not CARATER:
        CASEMIX = CASEMIX.replace(" + urg", "")
    OUT = os.path.join(RAIZ, "resultados_v2" + a.tag)
    os.makedirs(OUT, exist_ok=True)
    R["configuracao"] = {"janela": f"{INI}–{FIM}", "carater_no_ajuste": CARATER, "microdados": os.path.basename(MICRO),
                         "casemix": CASEMIX}
    so = set(a.so.split(",")) if a.so else None
    rodar = lambda e: so is None or e in so
    d = carregar()
    g = celulas(d)
    log(f"{len(d)} AIH, {len(g)} células")
    jp = os.path.join(OUT, "resultados_v2.json")
    if so and os.path.exists(jp):
        R.update({k: v for k, v in json.load(open(jp)).items() if k not in R})
    if rodar("t1"):
        tabela1(d)
        sigtap(d)
        comparacao_brasil(d)
        log("tabela 1, SIGTAP, comparação")
    h = adocao(d)
    log("adoção")
    if rodar("variacao"):
        variacao(d, h)
        log("variação")
    if rodar("tempo"):
        mortalidade_tempo(d, g)
        log("mortalidade no tempo")
    if rodar("inst"):
        institucional(d, g, a.boot)
        log("institucional")
    if rodar("indiv"):
        individual(d, g, a.boot)
        log("individual")
    if rodar("sens"):
        sensibilidades(d, g)
    if rodar("iv"):
        instrumental(d)
    R["extracao"] = ("SIH/SUS RD via espelho healthbr-data; competências jan/2008–jun/2026 (13/out/2026)"
                     + ("; jan/2000–dez/2007 das tabelas SIH antigas (15/out/2026)" if INI < 2008 else ""))
    with open(jp, "w") as f:
        json.dump(R, f, indent=1, ensure_ascii=False, default=lambda o: float(o) if isinstance(o, np.floating) else int(o) if isinstance(o, np.integer) else str(o))
    excel(os.path.join(RAIZ, f"tabelas_v2_crvm_{INI}_{FIM}{a.tag}.xlsx"))
    log("fim")


if __name__ == "__main__":
    main()
