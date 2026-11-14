#!/usr/bin/env python3
"""Diagnósticos de codificação (antes de qualquer modelo novo).

A. Auditoria da codificação da técnica (4 códigos SIGTAP): séries anuais e mensais por código, complemento
   federal, valores pagos, mortalidade por código, coerência de recursos, mudança por hospital.
B. Informação dentro dos hospitais para o modelo com efeitos fixos: decomposição da variância da % sem CEC,
   hospitais com mudança ≥5/10/20 pp, peso de identificação por hospital (concentração).

Sem modelos de desfecho. Saída: artigo/resultados_diag/*.csv, diagnosticos.json e diagnosticos.xlsx
"""
import json
import os

import numpy as np
import pandas as pd

RAIZ = os.path.dirname(os.path.abspath(__file__))
MICRO = os.path.join(RAIZ, "..", "saida_crvm", "crvm_2008_2025_internacao_microdados.csv")
OUT = os.path.join(RAIZ, "resultados_diag")
os.makedirs(OUT, exist_ok=True)
COD = {"0406010927": "Com CEC, 1 enxerto", "0406010935": "Com CEC, ≥2 enxertos",
       "0406010943": "Sem CEC, 1 enxerto", "0406010951": "Sem CEC, ≥2 enxertos"}
R = {}


def salvar(df, nome):
    df.to_csv(os.path.join(OUT, nome + ".csv"), index=False)
    return df


def carregar():
    d = pd.read_csv(MICRO, low_memory=False, dtype={"PROC_REA": str, "CNES": str, "CAR_INT": str, "COBRANCA": str,
                                                     "DT_INTER": str, "ANO_CMPT": str, "MES_CMPT": str})
    d = d[d.idade_anos >= 18].copy()
    d["ano"] = d.ano_internacao.astype(int)
    d["mes_inter"] = d.DT_INTER.str[:6]
    d["mes_cmpt"] = d.ANO_CMPT + d.MES_CMPT.str.zfill(2)
    d["codigo"] = d.PROC_REA.map(COD)
    d["off"] = d.PROC_REA.isin(["0406010943", "0406010951"]).astype(int)
    d["um_enxerto"] = d.PROC_REA.isin(["0406010927", "0406010943"]).astype(int)
    d["urg"] = (d.CAR_INT.str.strip().str.zfill(2) != "01").astype(int)
    d["fed"] = d[["VAL_SH_FED", "VAL_SP_FED"]].fillna(0).sum(axis=1)
    d["tem_fed"] = (d.fed > 0).astype(int)
    d["uti"] = d.UTI_MES_TO.fillna(0)
    d["sangue"] = (d.VAL_SANGUE.fillna(0) > 0).astype(int)
    ipca = pd.read_csv(os.path.join(RAIZ, "dados_ref", "ipca_fator_dez2025.csv")).set_index("ano")["fator_dez2025"]
    d["f"] = d.ano.map(ipca)
    return d


# ------------------------------------------------------------------ A. codificação
def auditoria(d):
    # A1. série anual por código
    a = d.groupby(["ano", "codigo"]).agg(n=("MORTE", "size"), obitos=("MORTE", "sum"),
                                          sp_mediano=("VAL_SP", "median"), sh_mediano=("VAL_SH", "median"),
                                          tot_medio=("VAL_TOT", "mean"), pct_com_fed=("tem_fed", "mean"),
                                          fed_medio=("fed", "mean"), uti_media=("uti", "mean"),
                                          perm_mediana=("DIAS_PERM", "median"), pct_sangue=("sangue", "mean"),
                                          pct_urg=("urg", "mean"), idade_media=("idade_anos", "mean")).reset_index()
    a["mort_pct"] = 100 * a.obitos / a.n
    a["pct_com_fed"] *= 100
    a["pct_sangue"] *= 100
    a["pct_urg"] *= 100
    tot = a.groupby("ano").n.transform("sum")
    a["pct_do_ano"] = 100 * a.n / tot
    salvar(a.round(2), "A1_serie_anual_por_codigo")
    w = a.pivot(index="ano", columns="codigo", values="n").fillna(0).astype(int)
    w["% sem CEC"] = 100 * (w["Sem CEC, 1 enxerto"] + w["Sem CEC, ≥2 enxertos"]) / w.sum(axis=1)
    w["% 1 enxerto entre com CEC"] = 100 * w["Com CEC, 1 enxerto"] / (w["Com CEC, 1 enxerto"] + w["Com CEC, ≥2 enxertos"])
    w["% 1 enxerto entre sem CEC"] = 100 * w["Sem CEC, 1 enxerto"] / (w["Sem CEC, 1 enxerto"] + w["Sem CEC, ≥2 enxertos"])
    salvar(w.round(2).reset_index(), "A1b_contagens_por_codigo_wide")
    # A2. mensal por competência e por mês de internação, 2021–2025
    for base, col in [("competencia", "mes_cmpt"), ("internacao", "mes_inter")]:
        m = d[d[col] >= "202101"].groupby([col, "codigo"]).agg(n=("MORTE", "size"), pct_fed=("tem_fed", "mean")).reset_index()
        mw = m.pivot(index=col, columns="codigo", values="n").fillna(0).astype(int)
        mw["% 1 enxerto entre com CEC"] = 100 * mw["Com CEC, 1 enxerto"] / (mw["Com CEC, 1 enxerto"] + mw["Com CEC, ≥2 enxertos"])
        mw["% sem CEC"] = 100 * (mw["Sem CEC, 1 enxerto"] + mw["Sem CEC, ≥2 enxertos"]) / mw[list(COD.values())].sum(axis=1)
        mw["% AIH com complemento federal"] = 100 * d[d[col] >= "202101"].groupby(col).tem_fed.mean()
        salvar(mw.round(2).reset_index(), f"A2_mensal_por_{base}")
    # A3. complemento federal: início e valor por código
    f = d[d.tem_fed == 1]
    R["fed_primeira_competencia"] = f.mes_cmpt.min() if len(f) else None
    R["fed_primeira_competencia_por_codigo"] = f.groupby("codigo").mes_cmpt.min().to_dict()
    R["fed_pct_aih_2024_2025_por_codigo"] = (100 * d[d.ano >= 2024].groupby("codigo").tem_fed.mean()).round(1).to_dict()
    R["fed_valor_mediano_por_codigo"] = f.groupby("codigo").fed.median().round(2).to_dict()
    R["fed_hospitais_com_complemento"] = int(f.CNES.nunique())
    # A4. mortalidade 1 vs ≥2 enxertos dentro de cada técnica, por período e caráter
    d["periodo"] = pd.cut(d.ano, [2007, 2013, 2019, 2022, 2025], labels=["2008–13", "2014–19", "2020–22", "2023–25"]).astype(str)
    mm = d.groupby(["periodo", "off", "urg", "um_enxerto"]).agg(n=("MORTE", "size"), mort=("MORTE", "mean")).reset_index()
    mm["mort"] *= 100
    mm["tecnica"] = np.where(mm.off == 1, "Sem CEC", "Com CEC")
    mm["carater"] = np.where(mm.urg == 1, "Urgência", "Eletiva")
    mm["enxertos"] = np.where(mm.um_enxerto == 1, "1", "≥2")
    salvar(mm[["periodo", "tecnica", "carater", "enxertos", "n", "mort"]].round(2), "A4_mortalidade_enxertos_periodo_carater")
    # A5. por hospital: % 1 enxerto entre com CEC, 2019–22 vs 2024–25
    def pct1(x):
        on = x[x.off == 0]
        return pd.Series({"n_on": len(on), "pct1_on": 100 * on.um_enxerto.mean() if len(on) else np.nan,
                          "n": len(x), "pct_off": 100 * x.off.mean()})
    a0 = d[d.ano.between(2019, 2022)].groupby("CNES").apply(pct1, include_groups=False).add_suffix("_19_22")
    a1 = d[d.ano.between(2024, 2025)].groupby("CNES").apply(pct1, include_groups=False).add_suffix("_24_25")
    h = a0.join(a1, how="inner")
    h = h[(h.n_on_19_22 >= 40) & (h.n_on_24_25 >= 20)].copy()
    h["delta_pct1_on"] = h.pct1_on_24_25 - h.pct1_on_19_22
    h["delta_pct_off"] = h.pct_off_24_25 - h.pct_off_19_22
    h["vol_anual_19_22"] = h.n_19_22 / 4
    h["vol_anual_24_25"] = h.n_24_25 / 2
    h = h.sort_values("delta_pct1_on", ascending=False)
    salvar(h.round(2).reset_index(), "A5_hospitais_mudanca_1enxerto")
    # contribuição de cada hospital ao aumento de casos "com CEC 1 enxerto"
    c = d[(d.PROC_REA == "0406010927")]
    inc = (c[c.ano.between(2024, 2025)].groupby("CNES").size() / 2 - c[c.ano.between(2019, 2022)].groupby("CNES").size().reindex(
        c[c.ano.between(2024, 2025)].CNES.unique()).fillna(0) / 4).sort_values(ascending=False)
    tot_inc = (c[c.ano.between(2024, 2025)].shape[0] / 2) - (c[c.ano.between(2019, 2022)].shape[0] / 4)
    R["com_cec_1enx_aumento_anual_19_22_para_24_25"] = float(tot_inc)
    R["com_cec_1enx_aumento_top5_hosp_pct"] = float(100 * inc.iloc[:5].sum() / tot_inc) if tot_inc > 0 else None
    R["com_cec_1enx_aumento_top10_hosp_pct"] = float(100 * inc.iloc[:10].sum() / tot_inc) if tot_inc > 0 else None
    R["hosp_salto_1enx_on_ge20pp"] = int((h.delta_pct1_on >= 20).sum())
    R["hosp_salto_1enx_on_ge10pp"] = int((h.delta_pct1_on >= 10).sum())
    R["hosp_analisados_A5"] = int(len(h))
    R["hosp_codificacao_estavel_abs_lt10pp"] = int((h.delta_pct1_on.abs() < 10).sum())
    # A6. coerência de recursos por técnica e período
    co = d.groupby(["periodo", "off"]).agg(n=("MORTE", "size"), uti_media=("uti", "mean"), pct_com_uti=("uti", lambda s: 100 * (s > 0).mean()),
                                           perm_mediana=("DIAS_PERM", "median"), pct_sangue=("sangue", "mean"),
                                           sh_real_medio=("VAL_SH", lambda s: np.nan), ).reset_index()
    sh = d.assign(sh_r=d.VAL_SH * d.f, sp_r=d.VAL_SP * d.f).groupby(["periodo", "off"])[["sh_r", "sp_r"]].mean().reset_index()
    co = co.drop(columns="sh_real_medio").merge(sh, on=["periodo", "off"])
    co["pct_sangue"] *= 100
    co["tecnica"] = np.where(co.off == 1, "Sem CEC", "Com CEC")
    salvar(co.drop(columns="off").round(2), "A6_coerencia_recursos")


# ------------------------------------------------------------------ B. informação dentro dos hospitais
def identificacao(d, ano_max, rotulo):
    x = d[d.ano <= ano_max]
    hy = x.groupby(["CNES", "ano"]).agg(n=("off", "size"), n_off=("off", "sum")).reset_index()
    hy["p"] = hy.n_off / hy.n
    hy = hy[hy.n >= 10]  # hospital-anos com ≥10 CRVM
    anos_h = hy.groupby("CNES").ano.nunique()
    hy = hy[hy.CNES.isin(anos_h[anos_h >= 2].index)].copy()
    # média do hospital ponderada por n
    hy["p_h"] = hy.groupby("CNES").apply(lambda g: np.average(g.p, weights=g.n), include_groups=False).reindex(hy.CNES).values
    p_tot = np.average(hy.p, weights=hy.n)
    within = np.sum(hy.n * (hy.p - hy.p_h) ** 2)
    between = np.sum(hy.n * (hy.p_h - p_tot) ** 2)
    hs = hy.groupby("CNES").apply(lambda g: pd.Series({
        "anos": len(g), "aih": g.n.sum(), "p_media": np.average(g.p, weights=g.n),
        "p_min": g.p.min(), "p_max": g.p.max(), "dp_intra": np.sqrt(np.average((g.p - np.average(g.p, weights=g.n)) ** 2, weights=g.n)),
        "peso_identificacao": np.sum(g.n * (g.p - np.average(g.p, weights=g.n)) ** 2)}), include_groups=False).reset_index()
    hs["amplitude_pp"] = 100 * (hs.p_max - hs.p_min)
    hs["dp_intra_pp"] = 100 * hs.dp_intra
    hs["pct_peso"] = 100 * hs.peso_identificacao / hs.peso_identificacao.sum()
    hs = hs.sort_values("pct_peso", ascending=False)
    salvar(hs.round(4), f"B1_hospitais_identificacao_{rotulo}")
    res = {
        "hospitais": int(len(hs)), "hospital_anos": int(len(hy)), "aih": int(hy.n.sum()),
        "pct_variancia_intra": float(100 * within / (within + between)),
        "pct_variancia_entre": float(100 * between / (within + between)),
        "hosp_sempre_0pct": int((hs.p_max == 0).sum()),
        "aih_hosp_sempre_0pct_pct": float(100 * hs.loc[hs.p_max == 0, "aih"].sum() / hs.aih.sum()),
        "dp_intra_mediano_pp": float(hs.dp_intra_pp.median()),
    }
    for k in (5, 10, 20):
        sel = hs.amplitude_pp >= k
        res[f"hosp_amplitude_ge{k}pp"] = int(sel.sum())
        res[f"aih_pct_hosp_amplitude_ge{k}pp"] = float(100 * hs.loc[sel, "aih"].sum() / hs.aih.sum())
    for k in (1, 5, 10, 20):
        res[f"peso_top{k}_pct"] = float(hs.pct_peso.iloc[:k].sum())
    # nº efetivo de hospitais (inverso do Herfindahl dos pesos)
    w = hs.peso_identificacao / hs.peso_identificacao.sum()
    res["n_efetivo_hospitais"] = float(1 / np.sum(w ** 2))
    R[f"identificacao_{rotulo}"] = res


def main():
    d = carregar()
    auditoria(d)
    identificacao(d, 2025, "2008_2025")
    identificacao(d, 2022, "2008_2022")
    with open(os.path.join(OUT, "diagnosticos.json"), "w") as f:
        json.dump(R, f, indent=1, ensure_ascii=False, default=str)
    with pd.ExcelWriter(os.path.join(OUT, "diagnosticos.xlsx"), engine="xlsxwriter") as w:
        for fn in sorted(os.listdir(OUT)):
            if fn.endswith(".csv"):
                t = pd.read_csv(os.path.join(OUT, fn))
                t.to_excel(w, sheet_name=fn[:-4][:31], index=False)
                w.sheets[fn[:-4][:31]].freeze_panes(1, 0)
        pd.DataFrame([(k, json.dumps(v, ensure_ascii=False)) for k, v in R.items()], columns=["indicador", "valor"]).to_excel(
            w, sheet_name="resumo", index=False)
    print(json.dumps(R, indent=1, ensure_ascii=False, default=str))


if __name__ == "__main__":
    main()
