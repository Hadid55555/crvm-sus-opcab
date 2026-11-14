#!/usr/bin/env python3
"""Emenda a extração pré-SIGTAP (competências 2000–2007) com a base 2008–2025 → base 2000–2025 por ano de internação.

Entradas: pre2008/saida/parciais (sih_crvm_pre2008.py), saida_crvm/parciais e
          saida_crvm/crvm_2008_2025_internacao_microdados.csv (não são alterados).
Regras:
  - Internações 2000–2007: AIH das tabelas antigas (competências 2000–2007) + AIH com internação ≤2007 processadas
    em 2008 ou depois (já com código SIGTAP, lidas dos parciais da base 2008–2025).
  - Técnica: 32011016 processada antes de mar/2002 → "Não separável" (único código de RVM; não existia código
    sem CEC). Demais códigos: Com CEC / Sem CEC (pre2008/VIABILIDADE.md).
  - Hospital antes de 2004 (sem CNES): CNPJ (CGC_HOSP) → CNES por (1) pares CNPJ–CNES nas próprias AIH de CRVM
    2003–2007, (2) arquivos brutos de jan/2004, jun/2004 e jan/2005 da UF e (3) troca de identificador em ago/2003
    (CNPJ ativo até jun–jul/2003 e um único CNES novo, sem CNPJ real, no mesmo município; candidatos disputados por
    dois CNPJ ficam sem correspondência), (4) CNPJ idêntico no cadastro do CNES e (5) nome na Receita + continuidade
    do faturamento (pre2008/mapa_cgc_cnes_manual.csv, com a evidência de cada caso). CNES '0000000' = não informado.
    Sem correspondência: 'CGC:<cnpj>'.
Saídas: saida_crvm/crvm_2000_2025_internacao_microdados.csv (+ .zip), pre2008/saida/serie_anual_2000_2025.csv,
        pre2008/saida/log_emenda.md, pre2008/saida/mapa_cgc_cnes.csv
"""
import json
import os
import sys
import zipfile
from types import SimpleNamespace

import numpy as np
import pandas as pd

AQUI = os.path.dirname(os.path.abspath(__file__))
RAIZ = os.path.dirname(AQUI)
sys.path.insert(0, RAIZ)
sys.path.insert(0, AQUI)
import sih_crvm as S  # noqa: E402
import sih_crvm_pre2008 as P  # noqa: E402

SAIDA = os.path.join(AQUI, "saida")
BASE08 = os.path.join(RAIZ, "saida_crvm", "crvm_2008_2025_internacao_microdados.csv")
OUT = os.path.join(RAIZ, "saida_crvm", "crvm_2000_2025_internacao_microdados.csv")
SIGTAP = {"0406010927": "Com CEC", "0406010935": "Com CEC", "0406010943": "Sem CEC", "0406010951": "Sem CEC"}
LOG = []


def log(t=""):
    print(t, flush=True)
    LOG.append(t)


def antigos():
    P.configurar()
    d = S.consolidar(SimpleNamespace(saida=SAIDA))
    d["CGC_HOSP"] = d["CGC_HOSP"].fillna("").astype(str).str.strip()
    d["CNES"] = d["CNES"].fillna("").astype(str).str.strip()
    d.loc[d.CNES.str.strip("0") == "", "CNES"] = ""  # '0000000' = CNES não informado (placeholder do SIH antigo)
    cmpt = d.ANO_CMPT.str.strip() + d.MES_CMPT.str.strip().str.zfill(2)
    d["tecnica_separavel"] = np.where((d.PROC_REA == "32011016") & (cmpt < "200203"), 0, 1)
    d.loc[d.tecnica_separavel == 0, "grupo"] = "Não separável"
    d["tabela_procedimento"] = np.where(d.PROC_REA.str.startswith("32"), "SIH antiga (32xxxxxx, até jun/2004)",
                                        "SIH antiga (48xxxxxx, jul/2004–dez/2007)")
    return d


def sigtap_2007():
    """Internações ≤2007 processadas em 2008+ (já SIGTAP), dos parciais da base 2008–2025."""
    S.PROCS = dict(SIGTAP)
    S.PROC_NOME = {
        "0406010927": "Revascularização miocárdica c/ uso de extracorpórea",
        "0406010935": "Revascularização miocárdica c/ uso de extracorpórea (c/ 2 ou mais enxertos)",
        "0406010943": "Revascularização miocárdica s/ uso de extracorpórea",
        "0406010951": "Revascularização miocárdica s/ uso de extracorpórea (c/ 2 ou mais enxertos)"}
    pasta = os.path.join(RAIZ, "saida_crvm", "parciais")
    arqs = [f for f in sorted(os.listdir(pasta)) if f.startswith("RD") and f.endswith(".csv")]
    partes = []
    for f in arqs:
        x = pd.read_csv(os.path.join(pasta, f), dtype=str, keep_default_na=False)
        x = x[x.DT_INTER.str.strip().str[:4] <= "2007"]
        if len(x):
            partes.append(x)
    tmp = os.path.join(SAIDA, "_tmp_sigtap2007")
    os.makedirs(os.path.join(tmp, "parciais"), exist_ok=True)
    pd.concat(partes).to_csv(os.path.join(tmp, "parciais", "RDXX0000.csv"), index=False)
    d = S.consolidar(SimpleNamespace(saida=tmp))
    os.remove(os.path.join(tmp, "parciais", "RDXX0000.csv"))
    os.rmdir(os.path.join(tmp, "parciais"))
    os.rmdir(tmp)
    d["tecnica_separavel"] = 1
    d["tabela_procedimento"] = "SIGTAP (internação ≤2007 processada em 2008+)"
    d["CGC_HOSP"] = ""
    d["CNES"] = d["CNES"].fillna("").astype(str).str.strip()
    return d


def mapa_cgc(d):
    import pyarrow.parquet as pq
    mp = d[(d.CNES != "") & (d.CGC_HOSP != "")].groupby("CGC_HOSP").CNES.agg(lambda s: s.mode().iat[0]).to_dict()
    regra = {k: "1: pares CNPJ–CNES nas AIH de CRVM 2003–2007" for k in mp}
    manifesto = json.load(open(os.path.join(SAIDA, "manifest_sih_rd.json")))["partitions"]
    sem = d[(d.CNES == "") & (d.CGC_HOSP != "") & ~d.CGC_HOSP.isin(mp)]
    cache = {}
    for cgc, x in sem.groupby("CGC_HOSP"):
        uf = S.UF_IBGE.get(x.MUNIC_MOV.str[:2].mode().iat[0])
        for k in [f"2004-01-{uf}", f"2004-06-{uf}", f"2005-01-{uf}"]:
            if k not in cache:
                info = manifesto.get(k)
                if not info:
                    cache[k] = pd.DataFrame(columns=["CGC_HOSP", "CNES"])
                    continue
                tabs = []
                for of in info["output_files"]:
                    dst = os.path.join(SAIDA, f"_bruto_{k}_{os.path.basename(of['path'])}")
                    S.baixar(f"{S.ESPELHO}/{of['path'][len('sih/rd/'):]}", dst)
                    if S.sha256(dst) != of["sha256"]:
                        raise ValueError(f"SHA-256 divergente em {k}")
                    tabs.append(pq.read_table(dst, columns=["CGC_HOSP", "CNES"]).to_pandas(ignore_metadata=True))
                    os.remove(dst)
                t = pd.concat(tabs)
                cache[k] = t.assign(CGC_HOSP=t.CGC_HOSP.str.strip(), CNES=t.CNES.str.strip())
            v = cache[k][(cache[k].CGC_HOSP == cgc) & (cache[k].CNES != "")].CNES.value_counts()
            if len(v):
                mp[cgc] = v.index[0]
                regra[cgc] = f"2: arquivo bruto {k} ({v.iloc[0]}/{v.sum()} AIH com esse CNPJ)"
                break
    # (3) troca de identificador em ago/2003: CNPJ ativo até jul/2003 e, no mesmo município, um único CNES que começa
    #     em ago/2003 ou depois e nunca aparece com CNPJ real (só provisório '77777…' ou vazio) nas AIH de CRVM 2003–2007
    cm = d.ANO_CMPT.str.strip() + d.MES_CMPT.str.strip().str.zfill(2)
    real = set(d[(d.CNES != "") & (d.CGC_HOSP != "") & ~d.CGC_HOSP.str.startswith("77777")].CNES)
    sem = d[(d.CNES == "") & (d.CGC_HOSP != "") & ~d.CGC_HOSP.isin(mp)]
    cand_por_cgc = {}
    for cgc, x in sem.groupby("CGC_HOSP"):
        mun = x.MUNIC_MOV.mode().iat[0]
        y = d[(d.MUNIC_MOV == mun) & (d.CNES != "") & ~d.CNES.isin(real)]
        ini = cm[y.index].groupby(y.CNES).min()
        cand_por_cgc[cgc] = (sorted(ini[ini >= "200308"].index), cm[x.index].max())
    usados = pd.Series([c for v, _ in cand_por_cgc.values() for c in v]).value_counts()
    for cgc, (cand, ultimo) in cand_por_cgc.items():
        if len(cand) == 1 and ultimo >= "200306" and usados[cand[0]] == 1:
            mp[cgc] = cand[0]
            regra[cgc] = "3: troca de identificador em ago/2003 (único CNES novo, sem CNPJ real, no município)"
    # (4) CNPJ idêntico no cadastro do CNES e (5) nome na Receita + continuidade: tabela documentada
    man = pd.read_csv(os.path.join(AQUI, "mapa_cgc_cnes_manual.csv"), dtype=str)
    for r in man.itertuples(index=False):
        if r.CGC_HOSP not in mp:
            mp[r.CGC_HOSP] = r.CNES
            regra[r.CGC_HOSP] = r.regra
    m = pd.DataFrame([{"CGC_HOSP": k, "CNES": v, "regra": regra[k]} for k, v in mp.items()])
    return mp, m


def main():
    d0 = antigos()
    log(f"Tabelas antigas (competências 2000–2007): {len(d0):,} AIH")
    d1 = sigtap_2007()
    log(f"SIGTAP com internação ≤2007 (processadas em 2008+): {len(d1):,} AIH")
    d = pd.concat([d0, d1], ignore_index=True)
    fora = d[~d.ano_internacao.between(2000, 2007)]
    log(f"Excluídas por internação fora de 2000–2007 (ex.: internações de 1999): {len(fora):,}")
    d = d[d.ano_internacao.between(2000, 2007)].copy()
    # hospital
    mp, m = mapa_cgc(d)
    m.to_csv(os.path.join(SAIDA, "mapa_cgc_cnes_local.csv"), index=False)  # identifica hospitais: fica fora do git
    d["CNES_ORIG"] = d.CNES
    d["CNES"] = np.where(d.CNES != "", d.CNES, d.CGC_HOSP.map(mp).fillna("CGC:" + d.CGC_HOSP))
    sem_cnes = d.CNES.str.startswith("CGC:")
    sem_id = sem_cnes & ~d.CGC_HOSP.isin(mp)
    log(f"AIH sem nenhum identificador de hospital resolvido: {int(sem_id.sum())}")
    log(f"CNPJ→CNES por regra: {m.regra.str[:1].value_counts().sort_index().to_dict()}")
    log(f"Hospital: CNES original em {(d.CNES_ORIG != '').mean():.1%} das AIH; após CNPJ→CNES, sem CNES em "
        f"{sem_cnes.sum():,} AIH ({sem_cnes.mean():.2%}) de {d[sem_cnes].CNES.nunique()} CNPJ")
    # base 2008–2025
    b = pd.read_csv(BASE08, dtype=str, keep_default_na=False)
    b["tecnica_separavel"] = "1"
    b["tabela_procedimento"] = "SIGTAP"
    b["CGC_HOSP"] = ""
    b["CNES_ORIG"] = b.CNES
    dup = set(d.N_AIH) & set(b.N_AIH)
    log(f"N_AIH presentes nas duas bases: {len(dup)}")
    d = d.astype(str).replace({"nan": "", "<NA>": "", "None": ""})
    cols = list(b.columns)
    for c in d.columns:
        if c not in cols:
            cols.append(c)
    t = pd.concat([d.reindex(columns=cols), b.reindex(columns=cols)], ignore_index=True).fillna("")
    t["n_aih_duplicado"] = t.N_AIH.duplicated(keep=False).astype(int)
    t.to_csv(OUT, index=False)
    with zipfile.ZipFile(OUT.replace(".csv", ".zip"), "w", zipfile.ZIP_DEFLATED) as z:
        z.write(OUT, os.path.basename(OUT))
    log(f"Base 2000–2025: {len(t):,} AIH ({len(d):,} de 2000–2007 + {len(b):,} de 2008–2025); "
        f"N_AIH repetidos: {int(t.n_aih_duplicado.sum())}")
    # série anual
    t["ano"] = t.ano_internacao.astype(int)
    t["morte"] = pd.to_numeric(t.MORTE, errors="coerce").fillna(0)
    t["adulto"] = pd.to_numeric(t.idade_anos, errors="coerce") >= 18
    g = t.groupby("ano")
    se = pd.DataFrame({
        "Nº de AIH (todas)": g.size(),
        "Com CEC": g.grupo.apply(lambda s: (s == "Com CEC").sum()),
        "Sem CEC": g.grupo.apply(lambda s: (s == "Sem CEC").sum()),
        "Não separável": g.grupo.apply(lambda s: (s == "Não separável").sum()),
        "Óbitos": g.morte.sum(), "Adultos (≥18 anos)": g.adulto.sum(),
        "Hospitais": g.CNES.nunique()})
    se["% sem CEC (entre separáveis)"] = (100 * se["Sem CEC"] / (se["Com CEC"] + se["Sem CEC"])).round(2)
    se["Mortalidade (%)"] = (100 * se["Óbitos"] / se["Nº de AIH (todas)"]).round(2)
    tab = t.groupby("ano").tabela_procedimento.agg(lambda s: "; ".join(sorted(s.unique())))
    se["Tabela de procedimentos"] = tab
    se = se.reset_index().rename(columns={"ano": "Ano"})
    se.to_csv(os.path.join(SAIDA, "serie_anual_2000_2025.csv"), index=False)
    log("")
    log("```\n" + se.to_string(index=False) + "\n```")
    cob = pd.read_csv(os.path.join(SAIDA, "cobertura_arquivos.csv"))
    falt = cob[cob.status != "lido"].arquivo.tolist()
    log(f"\nCobertura pré-2008: {len(cob)} arquivos esperados, {int((cob.status == 'lido').sum())} lidos; "
        f"faltando no espelho: {', '.join(falt) if falt else 'nenhum'}")
    open(os.path.join(SAIDA, "log_emenda.md"), "w").write("# Emenda 2000–2007 + 2008–2025\n\n" + "\n".join(LOG) + "\n")


if __name__ == "__main__":
    main()
