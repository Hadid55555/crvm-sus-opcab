#!/usr/bin/env python3
"""CRVM fora do SUS (CIHA) × SUS (SIH), internações de 2011–2025 — análise exploratória.

Entradas: CSV de extrair_ciha.py e saida_crvm/crvm_2000_2025_internacao_microdados.csv.
Regras:
  - Ano = ano da internação (CIHA: DT_ATEND; SIH: DT_INTER); adultos ≥18 anos.
  - Óbito = motivo de saída 41–43 nas duas bases. O campo MORTE da CIHA também marca o motivo 51 (encerramento
    administrativo) como óbito — erro de origem (ex.: um hospital com 242/242 "óbitos" em 2014).
  - CIHA que casa exatamente com uma AIH do SIH (CNES, datas, procedimento, sexo, idade) é excluída (duplicata).
  - FONTE (Portaria GM/MS 1.171/2011, TP_FREMU): 1 plano privado; 2 particular PF; 3 gratuito; 4 recurso próprio SES;
    5 recurso próprio SMS; 6 plano público; 8 DPVAT; 9 particular PJ.
Saídas: ciha/resultados/*.csv, ciha/ciha_crvm_2011_2025.xlsx e ciha/crvm_sus_e_fora_do_sus_2011_2025.zip (base única)
Uso: python ciha/analise_ciha.py <ciha_crvm.csv>
"""
import json
import os
import sys

import numpy as np
import pandas as pd
import patsy
from scipy.special import expit

RAIZ = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(RAIZ, "..", "artigo"))
OUT = os.path.join(RAIZ, "resultados")
os.makedirs(OUT, exist_ok=True)
import analise_v2 as A  # noqa: E402  (Modelo, irls, wilson)

OFF = ["0406010943", "0406010951"]
FONTE = {"01": "Plano privado", "02": "Particular (pessoa física)", "03": "Gratuito", "04": "Recurso próprio SES",
         "05": "Recurso próprio SMS", "06": "Plano público", "08": "DPVAT", "09": "Particular (pessoa jurídica)"}
GRUPO_FONTE = {"01": "Plano de saúde", "06": "Plano de saúde", "02": "Particular", "09": "Particular",
               "03": "Outros não SUS", "04": "Outros não SUS", "05": "Outros não SUS", "08": "Outros não SUS"}
REG = {"1": "Norte", "2": "Nordeste", "3": "Sudeste", "4": "Sul", "5": "Centro-Oeste"}
R = {}


def salvar(df, nome):
    df.to_csv(os.path.join(OUT, nome + ".csv"), index=False)
    return df


def idade(d):
    cod, ida = d.COD_IDADE.str.strip(), pd.to_numeric(d.IDADE, errors="coerce")
    return np.where(cod == "4", ida, np.where(cod == "5", 100 + ida, np.where(cod.isin(["0", "1", "2", "3"]), 0, np.nan)))


def carregar(arq_ciha):
    c = pd.read_csv(arq_ciha, dtype=str, keep_default_na=False)
    for x in c.columns:
        c[x] = c[x].str.strip()
    R["ciha_registros_crvm"] = len(c)
    s = pd.read_csv(os.path.join(RAIZ, "..", "saida_crvm", "crvm_2000_2025_internacao_microdados.csv"), dtype=str,
                    keep_default_na=False, usecols=["CNES", "DT_INTER", "DT_SAIDA", "PROC_REA", "SEXO", "IDADE", "COD_IDADE",
                                                    "COBRANCA", "MUNIC_MOV", "DIAG_PRINC", "DIAS_PERM", "UTI_MES_TO",
                                                    "N_AIH", "grupo"])
    s = s[s.grupo.isin(["Com CEC", "Sem CEC"])]
    k = ["CNES", "DT_INTER", "DT_SAIDA", "PROC_REA", "SEXO", "IDADE"]
    c = c.rename(columns={"DT_ATEND": "DT_INTER"})
    dup = c.merge(s[k].drop_duplicates(), on=k, how="left", indicator=True)["_merge"].eq("both").values
    R["ciha_duplicata_exata_sih_excluida"] = int(dup.sum())
    c = c[~dup].copy()
    c["base"] = "Fora do SUS (CIHA)"
    c["fonte"] = c.FONTE.map(FONTE).fillna("Ignorada")
    c["fonte_grupo"] = c.FONTE.map(GRUPO_FONTE).fillna("Ignorada")
    s["base"] = "SUS (SIH)"
    s["fonte"] = s["fonte_grupo"] = "SUS"
    cols = ["base", "fonte", "fonte_grupo", "CNES", "DT_INTER", "DT_SAIDA", "PROC_REA", "SEXO", "IDADE", "COD_IDADE",
            "COBRANCA", "MUNIC_MOV", "DIAG_PRINC", "DIAS_PERM", "UTI_MES_TO"]
    d = pd.concat([c[cols], s[cols]], ignore_index=True)
    d["ano"] = pd.to_numeric(d.DT_INTER.str[:4], errors="coerce")
    d["idade"] = idade(d)
    d = d[d.ano.between(2011, 2025) & (d.idade >= 18)].copy()
    d["off"] = d.PROC_REA.isin(OFF).astype(int)
    d["obito"] = d.COBRANCA.str[:1].eq("4").astype(int)
    d["fem"] = d.SEXO.isin(["2", "3"]).astype(int)
    d["uf"] = d.MUNIC_MOV.str[:2]
    d["reg"] = d.MUNIC_MOV.str[:1].map(REG)
    d["perm"] = pd.to_numeric(d.DIAS_PERM, errors="coerce")
    d["uti"] = pd.to_numeric(d.UTI_MES_TO, errors="coerce").fillna(0)
    cid = d.DIAG_PRINC.str.upper()
    d["diag"] = np.select([cid.str.startswith(("I21", "I22")), cid.str.startswith("I200"), cid.str.startswith("I20"),
                           cid.str.startswith("I24"), cid.str.startswith("I25")],
                          ["AMI (I21-I22)", "Unstable angina (I20.0)", "Other angina (I20.1-9)", "Other acute IHD (I24)",
                           "I25 chronic IHD"], default="Other codes")
    d["idade_i"] = d.idade.clip(upper=95).astype(int)
    return d


def resumo(x):
    lo, hi = A.wilson(x.obito.sum(), len(x)) if len(x) else (np.nan, np.nan)
    return pd.Series({"n": len(x), "pct_sem_cec": 100 * x.off.mean(), "mort_pct": 100 * x.obito.mean(), "mort_lo": 100 * lo,
                      "mort_hi": 100 * hi, "mort_com_cec": 100 * x[x.off == 0].obito.mean(),
                      "mort_sem_cec": 100 * x[x.off == 1].obito.mean(), "idade_media": x.idade.mean(),
                      "pct_fem": 100 * x.fem.mean(), "pct_iam": 100 * (x.diag == "AMI (I21-I22)").mean(),
                      "perm_mediana": x.perm.median(), "pct_com_uti": 100 * (x.uti > 0).mean(), "hospitais": x.CNES.nunique()})


def main():
    d = carregar(sys.argv[1])
    ci, su = d[d.base.str.startswith("Fora")], d[d.base.str.startswith("SUS")]
    R["ciha_adultos_2011_2025"], R["sih_adultos_2011_2025"] = len(ci), len(su)
    R["ciha_pct_do_total"] = 100 * len(ci) / len(d)
    # 1. por ano
    a = d.groupby(["ano", "base"]).apply(resumo, include_groups=False).reset_index()
    salvar(a.round(2), "C1_por_ano_e_base")
    # 2. SUS × fora do SUS × fonte (período todo)
    t = pd.concat([d.groupby("base").apply(resumo, include_groups=False).reset_index().rename(columns={"base": "grupo"}),
                   ci.groupby("fonte_grupo").apply(resumo, include_groups=False).reset_index().rename(columns={"fonte_grupo": "grupo"}),
                   ci.groupby("fonte").apply(resumo, include_groups=False).reset_index().rename(columns={"fonte": "grupo"})])
    salvar(t.round(2), "C2_sus_x_fora_do_sus_por_fonte")
    # 3. região
    rg = d.groupby(["reg", "base"]).apply(resumo, include_groups=False).reset_index()
    salvar(rg.round(2), "C3_por_regiao")
    # 4. hospitais: concentração e hospitais que atendem SUS e fora do SUS
    h = d.groupby(["CNES", "base"]).agg(n=("off", "size"), n_off=("off", "sum")).reset_index()
    hw = h.pivot(index="CNES", columns="base", values=["n", "n_off"]).fillna(0)
    hw.columns = [f"{a}_{'ciha' if 'CIHA' in b else 'sih'}" for a, b in hw.columns]
    hw = hw.reset_index()
    ambos = hw[(hw.n_ciha >= 20) & (hw.n_sih >= 20)].copy()
    ambos["pct_off_ciha"] = 100 * ambos.n_off_ciha / ambos.n_ciha
    ambos["pct_off_sih"] = 100 * ambos.n_off_sih / ambos.n_sih
    R["hospitais_ciha"] = int((hw.n_ciha > 0).sum())
    R["hospitais_ciha_tambem_sih"] = int(((hw.n_ciha > 0) & (hw.n_sih > 0)).sum())
    R["pct_aih_ciha_em_hosp_que_tambem_atendem_sus"] = float(100 * hw.loc[hw.n_sih > 0, "n_ciha"].sum() / hw.n_ciha.sum())
    R["hosp_ambos_>=20"] = int(len(ambos))
    R["pct_off_mesmo_hospital_ciha_vs_sih"] = {"ciha": float(100 * ambos.n_off_ciha.sum() / ambos.n_ciha.sum()),
                                               "sih": float(100 * ambos.n_off_sih.sum() / ambos.n_sih.sum()),
                                               "correlacao_entre_hospitais": float(ambos[["pct_off_ciha", "pct_off_sih"]].corr().iloc[0, 1])}
    salvar(ambos.round(2).sort_values("n_ciha", ascending=False), "C4_hospitais_sus_e_fora_do_sus")
    v = h[h.base.str.startswith("Fora")].n.sort_values(ascending=False)
    R["ciha_top10_hospitais_pct"] = float(100 * v.iloc[:10].sum() / v.sum())
    # 4b. Brasil total (SUS + fora do SUS) por ano
    tot = d.groupby("ano").apply(resumo, include_groups=False).reset_index()
    tot = tot.merge(d.groupby("ano").apply(lambda x: pd.Series({"n_sus": int(x.base.str.startswith("SUS").sum()),
                                                                 "n_fora_sus": int(x.base.str.startswith("Fora").sum())}),
                                           include_groups=False).reset_index(), on="ano")
    tot["pct_fora_sus"] = 100 * tot.n_fora_sus / tot.n
    salvar(tot.round(2), "C6_brasil_total_por_ano")
    # base única SUS + fora do SUS (adultos, 2011–2025)
    cols = ["base", "fonte", "fonte_grupo", "ano", "CNES", "DT_INTER", "DT_SAIDA", "PROC_REA", "off", "SEXO", "idade",
            "DIAG_PRINC", "diag", "COBRANCA", "obito", "MUNIC_MOV", "uf", "reg", "perm", "uti"]
    arq = os.path.join(RAIZ, "crvm_sus_e_fora_do_sus_2011_2025.csv")
    d[cols].to_csv(arq, index=False)
    import zipfile
    with zipfile.ZipFile(arq.replace(".csv", ".zip"), "w", zipfile.ZIP_DEFLATED) as z:
        z.write(arq, os.path.basename(arq))
    os.remove(arq)
    # 5. modelos: técnica e base (fora do SUS × SUS), ajuste por idade, sexo, diagnóstico, ano
    g = d.groupby(["CNES", "ano", "idade_i", "fem", "diag", "off", "base"]).agg(k=("obito", "sum"), n=("obito", "size")).reset_index()
    g = g.rename(columns={"idade_i": "idade"})
    g["fora"] = g.base.str.startswith("Fora").astype(int)
    cm = "C(ano) + cr(idade, df=4, constraints='center') + fem + C(diag, Treatment('I25 chronic IHD'))"
    linhas = []
    for nome, gx, form, termo in [
            ("Fora do SUS: sem × com CEC (ajustado)", g[g.fora == 1], f"{cm} + off", "off"),
            ("SUS 2011–2025: sem × com CEC (ajustado)", g[g.fora == 0], f"{cm} + off", "off"),
            ("Fora do SUS × SUS, mortalidade (ajustado, + técnica)", g, f"{cm} + off + fora", "fora"),
            ("Interação técnica × base", g, f"{cm} + off * fora", "off:fora")]:
        m = A.Modelo(form, gx)
        o = m.orci(termo)
        linhas.append({"modelo": nome, "aih": m.n_aih, "hospitais": m.n_hosp, **o})
    # sem × com CEC: chance de receber sem CEC fora do SUS × SUS no mesmo hospital (efeitos fixos de hospital)
    go = d.groupby(["CNES", "ano", "idade_i", "fem", "diag", "base"]).agg(k=("off", "sum"), n=("off", "size")).reset_index()
    go = go.rename(columns={"idade_i": "idade"})
    go["fora"] = go.base.str.startswith("Fora").astype(int)
    mf = A.ModeloFE(f"{cm} + fora", go)
    o = mf.orci("fora")
    linhas.append({"modelo": "Receber sem CEC: fora do SUS × SUS no MESMO hospital (efeitos fixos)", "aih": mf.n_aih,
                   "hospitais": mf.n_hosp, **o})
    salvar(pd.DataFrame(linhas).round(4), "C5_modelos")
    json.dump(R, open(os.path.join(OUT, "resultados_ciha.json"), "w"), indent=1, ensure_ascii=False, default=float)
    with pd.ExcelWriter(os.path.join(RAIZ, "ciha_crvm_2011_2025.xlsx"), engine="xlsxwriter") as xw:
        pd.DataFrame([
            ("Fonte", "CIHA (internações não pagas pelo SUS, Portaria GM/MS 1.171/2011), compilação pública em Parquet "
                      "(Kaggle rafatrindade/brazilian-kaggle-datahub, a partir dos .dbc do FTP do DATASUS); SUS = SIH/RD."),
            ("Período", "Internações de 2011–2025, adultos; RR ausente na CIHA."),
            ("Óbito", "Motivo de saída 41–43 nas duas bases (o campo MORTE da CIHA inclui o código 51 por erro)."),
            ("Duplicatas", f"{R['ciha_duplicata_exata_sih_excluida']} registros CIHA idênticos a AIH do SIH foram excluídos."),
            ("Uso", "Exploratório: a CIHA é de notificação obrigatória mas sub-registrada; não representa todo o setor privado."),
        ], columns=["item", "descrição"]).to_excel(xw, sheet_name="Leia-me", index=False)
        for nome in ["C1_por_ano_e_base", "C2_sus_x_fora_do_sus_por_fonte", "C3_por_regiao", "C4_hospitais_sus_e_fora_do_sus",
                     "C5_modelos", "C6_brasil_total_por_ano"]:
            pd.read_csv(os.path.join(OUT, nome + ".csv")).to_excel(xw, sheet_name=nome[:31], index=False)
        pd.json_normalize(R, sep=".").T.reset_index().rename(columns={"index": "resultado", 0: "valor"}).to_excel(
            xw, sheet_name="Resultados_chave", index=False)
    print(json.dumps(R, indent=1, ensure_ascii=False, default=float))


if __name__ == "__main__":
    main()
