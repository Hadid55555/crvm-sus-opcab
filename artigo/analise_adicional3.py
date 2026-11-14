#!/usr/bin/env python3
"""Análises adicionais post hoc (nov/2026).

F1. Modelo híbrido within–between (Mundlak): a % sem CEC do hospital-ano é decomposta em média do hospital (entre) e
    desvio anual em torno dela (dentro), no mesmo modelo logístico; teste de Wald da diferença entre os coeficientes.
F2. Estudo de eventos com os anos pós-evento agrupados em 0–1 (curva de aprendizado) e 2–5.
F3. Análise de viés de conversão (S16) aplicada em cada estrato de % sem CEC do hospital-ano (Tabela 3).
F4. P de interação da Tabela 3 nas três categorizações (lidos de resultados_adicionais*.json, item 3).

Mesma especificação de artigo/analise_adicional.py (janela e caráter por REV_INI / REV_CARATER).
Saídas: artigo/resultados_adicionais[SUF]/F1..F3_*.csv e F_resultados.json.
Uso: python artigo/analise_adicional3.py            (2003–2025 sem caráter)
     REV_INI=2008 REV_CARATER=1 python artigo/analise_adicional3.py   (pré-especificada)
"""
import json
import os
import sys

import numpy as np
import pandas as pd
import patsy
from scipy import stats

RAIZ = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, RAIZ)
import analise_adicional as ar  # noqa: E402
from analise_v2 import Modelo, dr_continua, dr_dummy, demean, ols_cluster  # noqa: E402

FORM, HOSP, OUT = ar.FORM, ar.HOSP, ar.OUT
F = {"configuracao": {"janela": f"{ar.INI_REV}–2025", "carater": ar.CAR_REV, "dr_individual_pp": ar.DR_IND}}


def salvar(df, nome):
    df.to_csv(os.path.join(OUT, nome + ".csv"), index=False)
    ar.log(f"salvo {nome}")


def contraste(m, a, b):
    """b_a − b_b com EP robusto; OR da razão e P."""
    i, j = m.idx(a), m.idx(b)
    c = np.zeros(len(m.b))
    c[i], c[j] = 1, -1
    est, se = float(c @ m.b), float(np.sqrt(c @ m.V @ c))
    return {"razao_or": float(np.exp(est)), "lo": float(np.exp(est - 1.96 * se)), "hi": float(np.exp(est + 1.96 * se)),
            "p": float(2 * stats.norm.sf(abs(est / se)))}


# ============================================================================ F1. Mundlak
def mundlak(d):
    linhas = []
    for x, rot in [("expo", "Same-year leave-one-out % (primary exposure)"), ("x_lag1", "Prior-year % (predetermined)")]:
        z = d[d[x].notna()].copy()
        z["hm"] = z.groupby("CNES")[x].transform("mean")  # média do hospital, ponderada pelas AIH
        z["dv"] = z[x] - z.hm
        z["hm10"], z["dv10"] = z.hm / 10, z.dv / 10
        g = z.groupby(ar.BASE + ["hm10", "dv10"], observed=True, dropna=False).agg(k=("MORTE", "sum"), n=("MORTE", "size")).reset_index()
        for am, gx in [("All hospitals", g), ("Hospitals with any OPCAB in the window", g[g.CNES.isin(z.loc[z.off == 1, "CNES"].unique())])]:
            m = Modelo(f"{FORM} + {HOSP} + hm10 + dv10", gx)
            for termo, comp in [("hm10", "Between (hospital mean), per +10 pp"), ("dv10", "Within (yearly deviation), per +10 pp")]:
                o, r = m.orci(termo), m.dr(dr_continua(termo, 1))
                linhas.append({"exposicao": rot, "amostra": am, "componente": comp, "or": o["or"], "lo": o["lo"], "hi": o["hi"],
                               "p": o["p"], "dr_pp": r["dr_pp"], "dr_lo": r["lo"], "dr_hi": r["hi"], "aih": m.n_aih,
                               "hospitais": m.n_hosp})
            c = contraste(m, "hm10", "dv10")
            linhas.append({"exposicao": rot, "amostra": am, "componente": "Between − within (ratio of ORs); Wald P",
                           "or": c["razao_or"], "lo": c["lo"], "hi": c["hi"], "p": c["p"], "aih": m.n_aih, "hospitais": m.n_hosp})
            F[f"mundlak_{x}_{am[:3]}"] = c
            ar.log(f"F1 {x} {am}")
    t = pd.DataFrame(linhas)
    t["esperado_dr_pp_se_OR_individual_causal"] = ar.DR_IND / 10
    salvar(t.round(4), "F1_mundlak_within_between")


# ============================================================================ F2. eventos 0–1 × 2–5
def eventos_agrupados(d):
    ev = ar.eventos(d)
    x = d.merge(ev[["CNES", "ano_evento", "delta_pp"]], on="CNES", how="left").reset_index(drop=True)
    x["rel"] = (x.ano - x.ano_evento).clip(-5, 5)
    gr = pd.factorize(x.CNES)[0]
    Wdf = patsy.dmatrix(FORM, x, return_type="dataframe").drop(columns="Intercept")
    janelas = [("Pre (−5 to −2)", [-5, -4, -3, -2]), ("Years 0–1", [0, 1]), ("Years 2–5", [2, 3, 4, 5])]
    linhas = []
    for nome, sel, escala in [("Signed (per +10 pp change)", None, True), ("Increases only (per event)", 1, False),
                              ("Decreases only (per event)", -1, False)]:
        msk = np.ones(len(x), bool) if sel is None else (x.delta_pp.isna() | (np.sign(x.delta_pp) == sel)).values
        E = np.column_stack([(x.rel.isin(rs) & x.delta_pp.notna()).astype(float) *
                             (x.delta_pp.fillna(0) / 10 if escala else 1.0) for _, rs in janelas])
        Xm = np.column_stack([E, Wdf.values])[msk]
        y = x.MORTE.values.astype(float)[msk]
        g2 = pd.factorize(gr[msk])[0]
        Xd = demean(Xm, g2)
        keep = Xd.std(axis=0) > 1e-12
        keep[:len(janelas)] = True
        yd = demean(y[:, None], g2)[:, 0]
        b, V = ols_cluster(yd, Xd[:, keep], g2)
        for i, (jn, _) in enumerate(janelas):
            se = np.sqrt(V[i, i])
            linhas.append({"versao": nome, "janela": jn, "coef_pp": 100 * b[i], "lo": 100 * (b[i] - 1.96 * se),
                           "hi": 100 * (b[i] + 1.96 * se), "p": float(2 * stats.norm.sf(abs(b[i] / se))),
                           "hospitais_evento": int(ev.CNES.nunique() if sel is None else (np.sign(ev.delta_pp) == sel).sum()),
                           "aih": int(msk.sum())})
        c = np.zeros(len(b))
        c[2], c[1] = 1, -1
        est, se = float(c @ b), float(np.sqrt(c @ V @ c))
        linhas.append({"versao": nome, "janela": "Difference (2–5 minus 0–1)", "coef_pp": 100 * est,
                       "lo": 100 * (est - 1.96 * se), "hi": 100 * (est + 1.96 * se), "p": float(2 * stats.norm.sf(abs(est / se)))})
        ar.log(f"F2 {nome}")
    t = pd.DataFrame(linhas)
    t["esperado_pp_se_individual_causal_por_10pp"] = ar.DR_IND / 10
    salvar(t.round(4), "F2_eventos_0a1_2a5")


# ============================================================================ F3. conversão por estrato
def conversao_estrato(d):
    """Mesma lógica de R8 (S16), estrato a estrato. Supõe a mesma taxa de conversão por tentativa sem CEC em todos os
    estratos (Mukherjee 2012: 4,9%; CORONARY: 3,2% emergente, 7,9% total) e OR de óbito do convertido 6,18. O OR ajustado
    'corrigido' é aproximado: OR ajustado observado × (OR bruto corrigido / OR bruto observado)."""
    r3 = pd.read_csv(os.path.join(OUT, "R3c_OR_individual_por_categoria_hospital.csv"))
    r3 = r3[(r3.modelo.str.startswith("Hospital-year OPCAB % (same year")) & (~r3.efeitos_fixos_hospital.astype(bool))]
    linhas = []
    for estr in ar.CAT4_LAB:
        x = d[d.cat_full == estr]
        n_off, k_off = int(x.off.sum()), int(x.loc[x.off == 1, "MORTE"].sum())
        n_on, k_on = int((x.off == 0).sum()), int(x.loc[x.off == 0, "MORTE"].sum())
        m_off, m_on = k_off / n_off, k_on / n_on
        or_obs = (m_off / (1 - m_off)) / (m_on / (1 - m_on))
        adj = r3[r3.estrato == estr]
        or_adj = float(adj["or"].iloc[0]) if len(adj) else np.nan
        for c_nome, c in [("3.2% (CORONARY, emergent)", 75 / 2356), ("4.9% (meta-analysis)", 920 / 18870),
                          ("7.9% (CORONARY, all)", 186 / 2356)]:
            orc = 6.18
            od = m_off / (1 - m_off) * orc
            m_c = od / (1 + od)
            n_c = n_off * c / (1 - c)
            k_c = n_c * m_c
            mi_off, mi_on = (k_off + k_c) / (n_off + n_c), (k_on - k_c) / (n_on - n_c)
            or_cor = (mi_off / (1 - mi_off)) / (mi_on / (1 - mi_on))
            linhas.append({"estrato": estr, "taxa_conversao": c_nome, "aih": len(x), "pct_off": 100 * n_off / len(x),
                           "mort_off_pct": 100 * m_off, "mort_on_pct": 100 * m_on, "convertidos_estimados": round(n_c, 1),
                           "pct_convertidos_no_pool_com_cec": 100 * n_c / n_on,
                           "mort_on_sem_convertidos_pct": 100 * mi_on, "inflacao_mort_on_pp": 100 * (m_on - mi_on),
                           "dr_bruta_obs_pp": 100 * (m_off - m_on), "dr_bruta_cor_pp": 100 * (mi_off - mi_on),
                           "or_bruto_obs": or_obs, "or_bruto_cor": or_cor, "or_ajustado_obs": or_adj,
                           "or_ajustado_cor_aprox": or_adj * or_cor / or_obs})
    t = pd.DataFrame(linhas)
    salvar(t.round(4), "F3_conversao_por_estrato")
    # inclinação do gradiente antes e depois da correção (4,9%): OR do estrato >30% / OR do estrato 0–5%
    q = t[t.taxa_conversao.str.startswith("4.9")].set_index("estrato")
    F["gradiente_or_ajustado_obs_0a5_vs_30"] = [float(q.loc["0-5%", "or_ajustado_obs"]), float(q.loc[">30%", "or_ajustado_obs"])]
    F["gradiente_or_ajustado_cor_0a5_vs_30"] = [float(q.loc["0-5%", "or_ajustado_cor_aprox"]), float(q.loc[">30%", "or_ajustado_cor_aprox"])]


def main():
    d = ar.preparar()
    ar.log(f"{len(d)} AIH ({F['configuracao']})")
    jr = os.path.join(OUT, "resultados_adicionais.json")
    if os.path.exists(jr):
        R = json.load(open(jr))
        F["interacao_tabela3"] = {k.replace("selecao_interacao_", ""): v for k, v in R.items() if k.startswith("selecao_interacao_")}
    mundlak(d)
    eventos_agrupados(d)
    conversao_estrato(d)
    with open(os.path.join(OUT, "F_resultados.json"), "w") as f:
        json.dump(F, f, indent=1, ensure_ascii=False, default=float)
    ar.log("fim F")


if __name__ == "__main__":
    main()
