#!/usr/bin/env python3
"""Compara as estimativas principais entre as rodadas de analise_v2.py (janela e caráter).

Principal = 2003–2025; sensibilidade = 2008–2025. Nenhuma usa o caráter da internação.
Saída: artigo/comparacao_janelas.csv e aba no artigo/tabelas_comparacao_janelas.xlsx
"""
import json
import os

import pandas as pd

RAIZ = os.path.dirname(os.path.abspath(__file__))
# Caráter da internação excluído de todas as análises (decisão do autor, 15/out/2026): a rodada com caráter
# (resultados_v2/) fica só como registro histórico e não entra na comparação.
RODADAS = [("Principal: 2003–2025", "resultados_v2_2003_2025_sem_carater"),
           ("Sensibilidade: 2008–2025", "resultados_v2_2008_2025_sem_carater")]


def ler(pasta, nome):
    p = os.path.join(RAIZ, pasta, nome + ".csv")
    return pd.read_csv(p) if os.path.exists(p) else None


def fmt(o, lo, hi, d=3):
    return f"{o:.{d}f} ({lo:.{d}f}–{hi:.{d}f})"


def linhas(rot, pasta):
    out = {}
    R = json.load(open(os.path.join(RAIZ, pasta, "resultados_v2.json")))
    out["AIH adultas analisadas"] = f"{R['fluxo']['adultos_analisados']:,}".replace(",", ".")
    out["Hospitais"] = R["n_hospitais"]
    out["Óbitos"] = f"{R['n_obitos']:,}".replace(",", ".")
    out["% sem CEC (período)"] = f"{R['off_global_pct']:.2f}"
    out["Mortalidade bruta (%)"] = f"{R['mort_global_pct']:.2f}"
    h = R["hier_M1_ano_casemix"]
    out["Variação entre hospitais: OR mediana (IC)"] = fmt(h["MOR"], h["MOR_lo"], h["MOR_hi"], 1)
    out["Variação: ICC latente"] = f"{h['ICC_latente']:.3f}"
    out["Variação dentro do hospital (% da variância)"] = f"{R['estabilidade_intra_hospital']['pct_variancia_dentro_hospital']:.1f}"
    out["Tendência da adoção: OR por ano (IC)"] = fmt(*[R["adocao_or_por_ano_linear"][k] for k in ("or", "lo", "hi")])
    for k, nome in [("Overall", "global"), ("On-pump", "com CEC"), ("Off-pump", "sem CEC")]:
        x = R[f"mort_tendencia_{k}"]["or_por_ano"]
        out[f"Tendência da mortalidade ({nome}): OR por ano (IC)"] = fmt(x["or"], x["lo"], x["hi"])
    m4 = ler(pasta, "M4_institucional")
    p = m4.iloc[0]
    out["Institucional (principal): OR por +10 pp (IC)"] = fmt(p["or"], p["lo"], p["hi"])
    out["Institucional: IC wild bootstrap"] = f"{p['or_lo_boot']:.3f}–{p['or_hi_boot']:.3f} (p={p['p_boot']:.3f})"
    out["Institucional: diferença de risco, pp por +10 pp (IC)"] = fmt(p["dr_pp"], p["dr_lo"], p["dr_hi"], 2)
    fe = m4[m4.modelo.str.contains("hospital FE \\[SENS", regex=True)].iloc[0]
    out["Institucional, efeitos fixos de hospital: OR (IC)"] = fmt(fe["or"], fe["lo"], fe["hi"])
    m6 = ler(pasta, "M6_individual")
    i = m6[m6.modelo.str.contains("PRIMARY")].iloc[0]
    out["Individual ajustado: OR sem × com CEC (IC)"] = fmt(i["or"], i["lo"], i["hi"])
    out["Individual: diferença de risco, pp (IC)"] = fmt(i["dr_pp"], i["dr_lo"], i["dr_hi"], 2)
    f = m6[m6.modelo == "Adjusted + hospital FE"].iloc[0]
    out["Individual com efeitos fixos de hospital: OR (IC)"] = fmt(f["or"], f["lo"], f["hi"])
    iv = ler(pasta, "IV1_variavel_instrumental")
    v = iv[(iv.instrumento.str.startswith("prior")) & (iv.efeitos_fixos_hospital)].iloc[0]
    out["Variável instrumental (ano anterior, efeitos fixos): pp (IC)"] = fmt(v["iv_dr_pp"], v["iv_lo"], v["iv_hi"], 2)
    out["Interação técnica × período (p)"] = f"{R['interacao_off_periodo_wald']['p']:.3f}"
    return out


def main():
    cols = {}
    for rot, pasta in RODADAS:
        if os.path.exists(os.path.join(RAIZ, pasta, "resultados_v2.json")):
            cols[rot] = linhas(rot, pasta)
    t = pd.DataFrame(cols)
    t.index.name = "Medida"
    t.to_csv(os.path.join(RAIZ, "comparacao_janelas.csv"))
    print(t.to_string())


if __name__ == "__main__":
    main()
