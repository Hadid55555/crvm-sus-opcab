#!/usr/bin/env python3
"""CRVM no SIH/SUS antes do SIGTAP: extração das competências jan/2000–dez/2007 (tabelas SIH antigas).

Reaproveita sih_crvm.py (download do espelho healthbr-data com SHA-256, filtro por PROC_REA, parciais por arquivo,
cobertura). Não altera a base 2008–2025. Códigos e fontes: pre2008/VIABILIDADE.md.

  tabela "32" (até jun/2004): 32011016 RVM com CEC (até fev/2002 é o único código de RVM → técnica não separável),
                              32039018 com CEC ≥2 enxertos, 32038011 sem CEC (desde mar/2002), 32040016 sem CEC ≥2
  tabela "48" (jul/2004–dez/2007): 48010073/48010081 com CEC; 48010090/48010103 sem CEC

Uso: python pre2008/sih_crvm_pre2008.py [--workers 8]
"""
import argparse
import os
import sys
from types import SimpleNamespace

AQUI = os.path.dirname(os.path.abspath(__file__))
RAIZ = os.path.dirname(AQUI)
sys.path.insert(0, RAIZ)
import sih_crvm as S  # noqa: E402

PROCS_ANTIGOS = {"32011016": "Com CEC", "32039018": "Com CEC", "32038011": "Sem CEC", "32040016": "Sem CEC",
                 "48010073": "Com CEC", "48010081": "Com CEC", "48010090": "Sem CEC", "48010103": "Sem CEC"}
NOMES_ANTIGOS = {
    "32011016": "RVM com circulação extracorpórea (tabela SIH até jun/2004)",
    "32039018": "RVM com CEC, dois ou mais enxertos (tabela SIH até jun/2004)",
    "32038011": "RVM sem circulação extracorpórea (tabela SIH até jun/2004)",
    "32040016": "RVM sem CEC, dois ou mais enxertos (tabela SIH até jun/2004)",
    "48010073": "Revascularização miocárdica com uso de extracorpórea (tabela SIH jul/2004–2007)",
    "48010081": "Revascularização miocárdica com uso de extracorpórea, 2 ou mais enxertos (tabela SIH jul/2004–2007)",
    "48010090": "Revascularização miocárdica sem uso de extracorpórea (tabela SIH jul/2004–2007)",
    "48010103": "Revascularização miocárdica sem uso de extracorpórea, 2 ou mais enxertos (tabela SIH jul/2004–2007)",
}


def configurar():
    S.PROCS = dict(PROCS_ANTIGOS)
    S.PROC_NOME = dict(NOMES_ANTIGOS)
    if "CGC_HOSP" not in S.COLUNAS:
        S.COLUNAS = S.COLUNAS + ["CGC_HOSP"]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--workers", type=int, default=8)
    a = ap.parse_args()
    configurar()
    args = SimpleNamespace(ini=2000, fim=2007, cmpt_fim=200712, saida=os.path.join(AQUI, "saida"), fonte="espelho",
                           workers=a.workers)
    tarefas = S.extrair(args)
    cob = S.cobertura(args, tarefas)
    cob.to_csv(os.path.join(args.saida, "cobertura_arquivos.csv"), index=False)
    print(cob.status.value_counts().to_dict())


if __name__ == "__main__":
    main()
