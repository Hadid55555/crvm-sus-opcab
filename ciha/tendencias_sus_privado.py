#!/usr/bin/env python3
"""Tendências 2011–2025: SUS (SIH) × fora do SUS (CIHA) — % sem CEC e mortalidade hospitalar. Exploratório.

Entrada: ciha/crvm_sus_e_fora_do_sus_2011_2025.zip (analise_ciha.py).
Dois recortes de hospitais:
  - todos;
  - painel estável: hospitais com CRVM notificada naquele sistema em ≥10 dos 15 anos (controla a queda de
    notificação da CIHA ao longo do tempo).
Modelos (logística binomial em células, erros robustos por hospital, ajuste por idade em spline, sexo e diagnóstico):
  - % sem CEC: ano contínuo × sistema (a interação testa se as tendências diferem);
  - mortalidade: ano × sistema, ajustada também pela técnica; e tendência de cada técnica em cada sistema.
Saídas: ciha/resultados/T*.csv e ciha/tendencias_sus_x_fora_do_sus.xlsx
"""
import json
import os
import sys

import numpy as np
import pandas as pd
from scipy.special import expit

RAIZ = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(RAIZ, "..", "artigo"))
import analise_v2 as A  # noqa: E402

OUT = os.path.join(RAIZ, "resultados")
CM = "cr(idade, df=4, constraints='center') + fem + C(diag, Treatment('I25 chronic IHD'))"
R = {}


def salvar(df, nome):
    df.to_csv(os.path.join(OUT, nome + ".csv"), index=False)
    return df


def carregar():
    d = pd.read_csv(os.path.join(RAIZ, "crvm_sus_e_fora_do_sus_2011_2025.zip"), dtype={"CNES": str, "PROC_REA": str})
    d["fora"] = d.base.str.startswith("Fora").astype(int)
    d["sistema"] = np.where(d.fora == 1, "Fora do SUS", "SUS")
    d["fem"] = d.SEXO.astype(str).isin(["2", "3"]).astype(int)
    d["idade"] = d.idade.clip(upper=95).astype(int)
    d["t"] = d.ano - 2011
    anos = d.groupby(["sistema", "CNES"]).ano.nunique().rename("anos_sistema").reset_index()
    d = d.merge(anos, on=["sistema", "CNES"])
    d["estavel"] = (d.anos_sistema >= 10).astype(int)
    return d


def serie(d, rotulo):
    linhas = []
    for (a, s), x in d.groupby(["ano", "sistema"]):
        lo, hi = A.wilson(x.off.sum(), len(x))
        mlo, mhi = A.wilson(x.obito.sum(), len(x))
        linhas.append({"recorte": rotulo, "ano": a, "sistema": s, "n": len(x), "hospitais": x.CNES.nunique(),
                       "pct_sem_cec": 100 * x.off.mean(), "lo": 100 * lo, "hi": 100 * hi,
                       "mort_pct": 100 * x.obito.mean(), "mort_lo": 100 * mlo, "mort_hi": 100 * mhi,
                       "mort_com_cec": 100 * x[x.off == 0].obito.mean(), "mort_sem_cec": 100 * x[x.off == 1].obito.mean()})
    return pd.DataFrame(linhas)


def ajustada_por_ano(g, form, alvo_col, sistemas):
    """% ajustada (padronização marginal ao case-mix total) por ano e sistema, a partir de modelo com C(ano)*fora."""
    m = A.Modelo(form, g)
    out = []
    w = g.n.values / g.n.sum()
    import patsy
    for s, f in sistemas:
        for a in sorted(g.ano.unique()):
            X = patsy.build_design_matrices([m.X.design_info], g.assign(ano=a, fora=f), return_type="dataframe")[0]
            out.append({"ano": a, "sistema": s, alvo_col: 100 * np.sum(w * expit(X.values @ m.b))})
    return pd.DataFrame(out)


def modelos(d, rotulo):
    linhas = []
    go = d.groupby(["CNES", "ano", "t", "idade", "fem", "diag", "fora"]).agg(k=("off", "sum"), n=("off", "size")).reset_index()
    m = A.Modelo(f"{CM} + t * fora", go)
    for termo, nome in [("t", "SUS: OR por ano de receber sem CEC"), ("t:fora", "Diferença de tendência (fora do SUS ÷ SUS)")]:
        linhas.append({"recorte": rotulo, "desfecho": "sem CEC", "termo": nome, **m.orci(termo)})
    b, V = m.b, m.V
    i, j = m.idx("t"), m.idx("t:fora")
    est, se = b[i] + b[j], np.sqrt(V[i, i] + V[j, j] + 2 * V[i, j])
    linhas.append({"recorte": rotulo, "desfecho": "sem CEC", "termo": "Fora do SUS: OR por ano de receber sem CEC",
                   "or": float(np.exp(est)), "lo": float(np.exp(est - 1.96 * se)), "hi": float(np.exp(est + 1.96 * se)),
                   "p": float(2 * __import__("scipy.stats", fromlist=["norm"]).norm.sf(abs(est / se))), "beta": float(est), "se": float(se)})
    gm = d.groupby(["CNES", "ano", "t", "idade", "fem", "diag", "fora", "off"]).agg(k=("obito", "sum"), n=("obito", "size")).reset_index()
    m2 = A.Modelo(f"{CM} + off + t * fora", gm)
    for termo, nome in [("t", "SUS: OR por ano de óbito"), ("t:fora", "Diferença de tendência da mortalidade (fora ÷ SUS)"),
                        ("fora", "Fora do SUS × SUS em 2011 (óbito)")]:
        linhas.append({"recorte": rotulo, "desfecho": "óbito (ajust. técnica)", "termo": nome, **m2.orci(termo)})
    for s, f in [("SUS", 0), ("Fora do SUS", 1)]:
        for tec, o in [("com CEC", 0), ("sem CEC", 1)]:
            x = gm[(gm.fora == f) & (gm.off == o)]
            mm = A.Modelo(f"{CM} + t", x)
            linhas.append({"recorte": rotulo, "desfecho": f"óbito, {tec}", "termo": f"{s}: OR por ano", **mm.orci("t")})
        x = gm[gm.fora == f]
        mm = A.Modelo(f"{CM} + C(ano) + off", x)
        linhas.append({"recorte": rotulo, "desfecho": "óbito", "termo": f"{s}: OR sem × com CEC (ajustado)", **mm.orci("off")})
    return pd.DataFrame(linhas), go, gm


def main():
    d = carregar()
    R["hospitais_painel_estavel"] = d[d.estavel == 1].groupby("sistema").CNES.nunique().to_dict()
    R["aih_painel_estavel"] = d[d.estavel == 1].groupby("sistema").size().to_dict()
    s = pd.concat([serie(d, "Todos os hospitais"), serie(d[d.estavel == 1], "Painel estável (≥10 de 15 anos)")])
    salvar(s.round(2), "T1_series_anuais")
    ms, ajs = [], []
    for rot, x in [("Todos os hospitais", d), ("Painel estável (≥10 de 15 anos)", d[d.estavel == 1])]:
        m, go, gm = modelos(x, rot)
        ms.append(m)
        a1 = ajustada_por_ano(go, f"{CM} + C(ano) * fora", "pct_sem_cec_ajustada", [("SUS", 0), ("Fora do SUS", 1)])
        a2 = ajustada_por_ano(gm, f"{CM} + off + C(ano) * fora", "mort_ajustada", [("SUS", 0), ("Fora do SUS", 1)])
        ajs.append(a1.merge(a2, on=["ano", "sistema"]).assign(recorte=rot))
    salvar(pd.concat(ms).round(4), "T2_modelos_tendencia")
    salvar(pd.concat(ajs).round(2), "T3_ajustadas_por_ano")
    json.dump(R, open(os.path.join(OUT, "tendencias.json"), "w"), indent=1, ensure_ascii=False, default=str)
    with pd.ExcelWriter(os.path.join(RAIZ, "tendencias_sus_x_fora_do_sus.xlsx"), engine="xlsxwriter") as xw:
        pd.DataFrame([("Conteúdo", "Tendências 2011–2025 de % sem CEC e mortalidade hospitalar: SUS (SIH) × fora do SUS (CIHA)."),
                      ("Painel estável", "Hospitais com CRVM notificada no sistema em ≥10 dos 15 anos (controla a queda de notificação da CIHA)."),
                      ("Ajuste", "Idade (spline), sexo, diagnóstico; mortalidade também pela técnica. Erros robustos por hospital."),
                      ("Ajustadas", "Padronização marginal ao case-mix de todas as AIH do recorte."),
                      ("Natureza", "Exploratório (post hoc); CIHA com sub-registro de grau desconhecido.")],
                     columns=["item", "descrição"]).to_excel(xw, sheet_name="Leia-me", index=False)
        for n in ["T1_series_anuais", "T2_modelos_tendencia", "T3_ajustadas_por_ano"]:
            pd.read_csv(os.path.join(OUT, n + ".csv")).to_excel(xw, sheet_name=n, index=False)
    print(json.dumps(R, indent=1, ensure_ascii=False, default=str))


if __name__ == "__main__":
    main()
