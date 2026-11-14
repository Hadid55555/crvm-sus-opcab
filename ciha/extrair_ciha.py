#!/usr/bin/env python3
"""CRVM fora do SUS: CIHA (Comunicação de Internação Hospitalar e Ambulatorial), 2011–2026.

Fonte: compilação pública da CIHA em Parquet (Kaggle rafatrindade/brazilian-kaggle-datahub, ciha/, CC0), feita a
partir dos CIHA{UF}{AAMM}.dbc do FTP do DATASUS (CIHA/201101_/Dados). Sem SHA-256 por arquivo (só tamanhos no
manifesto); o FTP do DATASUS não é alcançável deste ambiente. Filtra os 4 códigos SIGTAP de CRVM isolada.
Uso: python ciha/extrair_ciha.py <ciha.parquet> <saida.csv>
"""
import sys

import pyarrow as pa
import pyarrow.compute as pc
import pyarrow.parquet as pq

CODIGOS = ["0406010927", "0406010935", "0406010943", "0406010951"]
f = pq.ParquetFile(sys.argv[1])
partes, total = [], 0
for i in range(f.metadata.num_row_groups):
    t = f.read_row_group(i)
    total += t.num_rows
    t = t.filter(pc.is_in(pc.utf8_trim_whitespace(t["PROC_REA"]), value_set=pa.array(CODIGOS)))
    if t.num_rows:
        partes.append(t)
r = pa.concat_tables(partes)
r.to_pandas(ignore_metadata=True).to_csv(sys.argv[2], index=False)
print("registros lidos:", total, "| CRVM:", r.num_rows)
