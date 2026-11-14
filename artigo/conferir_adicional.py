#!/usr/bin/env python3
"""Conferência independente de artigo/analise_adicional.py: refaz a coorte a partir do CSV bruto (sem usar analise_v2)
e reajusta modelos-chave com statsmodels (GLM binomial, efeitos fixos por dummies, EP agrupado por hospital).
Saída: artigo/resultados_adicionais/Z_conferencia_independente.csv"""
import os
import numpy as np
import pandas as pd
import statsmodels.api as sm
import patsy

RAIZ = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(RAIZ, "resultados_adicionais")
d = pd.read_csv(os.path.join(RAIZ, "..", "saida_crvm", "crvm_2000_2025_internacao_microdados.csv"), low_memory=False,
                dtype={"N_AIH": str, "PROC_REA": str, "CNES": str, "SEXO": str, "DIAG_PRINC": str, "NAT_JUR": str})
OFF = {"32038011", "32040016", "48010090", "48010103", "0406010943", "0406010951"}
ON = {"32011016", "32039018", "48010073", "48010081", "0406010927", "0406010935"}
d = d[d.ano_internacao.between(2003, 2025) & d.PROC_REA.isin(OFF | ON) & (d.idade_anos >= 18)]
d = d.drop_duplicates("N_AIH")
lin = [("coorte_n", len(d), 486080), ("hospitais", d.CNES.nunique(), 307)]
d["off"] = d.PROC_REA.isin(OFF).astype(int)
d["ano"] = d.ano_internacao.astype(int)
d["idade"] = d.idade_anos.clip(upper=95)
d["fem"] = d.SEXO.isin(["2", "3"]).astype(int)
c = d.DIAG_PRINC.fillna("").str.upper()
d["dg"] = np.where(c.str[:3].isin(["I21", "I22"]), "ami", np.where(c.str[:4] == "I200", "ua", np.where(c.str[:3] == "I20", "oa",
          np.where(c.str[:3] == "I24", "i24", np.where(c.str[:3] == "I25", "a_i25", "oth")))))
nj = d[d.ano >= 2012].groupby("CNES").NAT_JUR.agg(lambda s: s.dropna().astype(str).str[0].mode().iloc[0] if s.notna().any() else "u")
d["nj"] = d.CNES.map(nj).fillna("u")
# exposições, calculadas de forma independente
t = d.groupby(["CNES", "ano"]).off.agg(["sum", "size"])
p_prev = {(h, a + 1): 100 * s / n for (h, a), (s, n) in t.iterrows() if n >= 10}
d["lag"] = [p_prev.get(k, np.nan) for k in zip(d.CNES, d.ano)]
d["pcur"] = d.groupby(["CNES", "ano"]).off.transform("mean") * 100
# confiabilidade (n ≥ 10)
hy = t[t["size"] >= 10].reset_index()
hy = hy[hy.groupby("CNES").ano.transform("count") >= 2]
q = hy["sum"] / hy["size"]
pb = hy.groupby("CNES")["sum"].transform("sum") / hy.groupby("CNES")["size"].transform("sum")
Nh = hy.groupby("CNES")["size"].transform("sum")
num = (hy["size"] * pb * (1 - pb) * (1 / hy["size"] - 1 / Nh)).sum()
den = (hy["size"] * (q - pb) ** 2).sum()
r_ref = pd.read_csv(os.path.join(OUT, "R1a_confiabilidade_pct_anual.csv")).set_index("hospital_anos_min_crvm")
lin.append(("confiabilidade_n>=10", 1 - num / den, r_ref.loc[10, "confiabilidade_ponderada"]))


def glm(form, dd, fe=False, termo=None):
    g = dd.groupby([c for c in ["CNES", "ano", "idade", "fem", "dg", "reg", "nj", "off", "x", "cat"] if c in dd],
                   dropna=False).MORTE.agg(["sum", "size"]).reset_index()
    if fe:
        form += " + C(CNES)"
        tot = g.groupby("CNES")["sum"].transform("sum")
        g = g[(tot > 0) & (tot < g.groupby("CNES")["size"].transform("sum"))]
    X = patsy.dmatrix(form, g, return_type="dataframe")
    y = np.c_[g["sum"], g["size"] - g["sum"]]
    r = sm.GLM(y, X, family=sm.families.Binomial()).fit(cov_type="cluster", cov_kwds={"groups": pd.factorize(g.CNES)[0]})
    b, se = r.params[termo], r.bse[termo]
    return np.exp(b), np.exp(b - 1.96 * se), np.exp(b + 1.96 * se)


d["reg"] = d.regiao
CM = "C(ano) + cr(idade, df=4, constraints='center') + fem + C(dg)"
r1b = pd.read_csv(os.path.join(OUT, "R1b_exposicao_predeterminada.csv"))
ref = r1b[(r1b.amostra == "Own sample") & r1b.modelo.str.startswith("Prior-year")]
x = d[d.lag.notna()].assign(x=lambda z: z.lag / 10)
lin.append(("lag_semFE_OR", glm(f"{CM} + C(reg) + C(nj) + x", x, termo="x")[0], ref[~ref.efeitos_fixos_hospital]["or"].iloc[0]))
lin.append(("lag_FE_OR", glm(f"{CM} + x", x, fe=True, termo="x")[0], ref[ref.efeitos_fixos_hospital]["or"].iloc[0]))
r3c = pd.read_csv(os.path.join(OUT, "R3c_OR_individual_por_categoria_hospital.csv"))
r3c = r3c[r3c.modelo.str.startswith("Hospital-year OPCAB %") & ~r3c.efeitos_fixos_hospital]
d["cat"] = pd.cut(d.pcur, [-0.001, 5, 15, 30, 101], labels=["0-5%", ">5-15%", ">15-30%", ">30%"]).astype(str)
for cat in ["0-5%", ">30%"]:
    lin.append((f"OR_individual_{cat}", glm(f"{CM} + C(reg) + C(nj) + off", d[d.cat == cat], termo="off")[0],
                r3c[r3c.estrato == cat]["or"].iloc[0]))
# viés de conversão (meta-análise 4,9%, OR 6,18), aritmética direta
n1, k1 = d.off.sum(), d[d.off == 1].MORTE.sum()
n0, k0 = (1 - d.off).sum(), d[d.off == 0].MORTE.sum()
m1 = k1 / n1
mc = 6.18 * m1 / (1 - m1) / (1 + 6.18 * m1 / (1 - m1))
nc = 920 / (18870 - 920) * n1
rd = 100 * ((k1 + nc * mc) / (n1 + nc) - (k0 - nc * mc) / (n0 - nc))
r8 = pd.read_csv(os.path.join(OUT, "R8_vies_conversao.csv"))
lin.append(("vies_conversao_dr_corrigida_4.9%_OR6.18", rd, r8.dr_bruta_corrigida_pp.iloc[4]))
z = pd.DataFrame(lin, columns=["item", "conferencia_independente", "analise_adicional"])
z["dif_relativa"] = (z.conferencia_independente - z.analise_adicional).abs() / z.analise_adicional.abs()
z.to_csv(os.path.join(OUT, "Z_conferencia_independente.csv"), index=False)
print(z.to_string())
