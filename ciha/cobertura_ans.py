#!/usr/bin/env python3
"""Plausibilidade da CIHA: CRVM por 100 mil beneficiários de plano médico (ANS) × CRVM SUS por 100 mil sem plano.

Beneficiários: ANS, beneficiários por UF (competência de dezembro; planos médico-hospitalares, NR_BENEF_M, vigência
anterior + posterior à Lei 9.656), compilação pública (Kaggle rafatrindade/brazilian-kaggle-datahub, ans/), agregada em
ciha/dados_ref/ans_beneficiarios_medicos_dezembro_uf.csv. População: IBGE (artigo/dados_ref/populacao_uf_ano_faixa.csv).
Taxas brutas (sem padronização etária). Saída: ciha/resultados/C7_cobertura_ciha_ans.csv
"""
import os

import pandas as pd

RAIZ = os.path.dirname(os.path.abspath(__file__))
UF = {"11": "RO", "12": "AC", "13": "AM", "14": "RR", "15": "PA", "16": "AP", "17": "TO", "21": "MA", "22": "PI", "23": "CE",
      "24": "RN", "25": "PB", "26": "PE", "27": "AL", "28": "SE", "29": "BA", "31": "MG", "32": "ES", "33": "RJ", "35": "SP",
      "41": "PR", "42": "SC", "43": "RS", "50": "MS", "51": "MT", "52": "GO", "53": "DF"}
b = pd.read_csv(os.path.join(RAIZ, "dados_ref", "ans_beneficiarios_medicos_dezembro_uf.csv"))
b = b.groupby(["ano", "SG_UF"]).NR_BENEF_M.sum().rename("benef").reset_index()
p = pd.read_csv(os.path.join(RAIZ, "..", "artigo", "dados_ref", "populacao_uf_ano_faixa.csv"))
p = p.groupby(["ano", "uf"])["pop"].sum().rename("pop").reset_index().rename(columns={"uf": "SG_UF"})
d = pd.read_csv(os.path.join(RAIZ, "crvm_sus_e_fora_do_sus_2011_2025.zip"), dtype={"uf": str})
d["SG_UF"] = d.uf.str.zfill(2).map(UF)
c = d.groupby(["ano", "SG_UF", "base"]).size().unstack(fill_value=0).reset_index()
c = c.rename(columns={"Fora do SUS (CIHA)": "crvm_ciha", "SUS (SIH)": "crvm_sus"})
t = c.merge(b, on=["ano", "SG_UF"]).merge(p, on=["ano", "SG_UF"])
br = t.groupby("ano")[["crvm_ciha", "crvm_sus", "benef", "pop"]].sum().reset_index()
br["cobertura_planos_pct"] = 100 * br.benef / br["pop"]
br["ciha_por_100mil_beneficiarios"] = 1e5 * br.crvm_ciha / br.benef
br["sus_por_100mil_sem_plano"] = 1e5 * br.crvm_sus / (br["pop"] - br.benef)
br["razao_ciha_sus"] = br.ciha_por_100mil_beneficiarios / br.sus_por_100mil_sem_plano
br["captacao_maxima_ciha_pct"] = 100 * br.razao_ciha_sus  # se a taxa real no setor privado fosse igual à do SUS
br.round(3).to_csv(os.path.join(RAIZ, "resultados", "C7_cobertura_ciha_ans.csv"), index=False)
print(br.round(2).to_string(index=False))
