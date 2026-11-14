#!/usr/bin/env python3
"""Sensibilidade pedida na 2ª rodada: modelos sem o diagnóstico principal (a fração de I20.0 salta na troca de tabela
de 2008, mais no sem CEC). Saída: artigo/resultados_adicionais/R10g_sem_diagnostico.csv"""
import os
import sys

import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import analise_adicional as ar  # noqa: E402
from analise_v2 import Modelo, ModeloFE, dr_continua, dr_dummy  # noqa: E402

d = ar.preparar()
g = ar.cel(d, ["expo"])
gi = g[g.expo.notna()].assign(e10=lambda z: z.expo / 10)
SEM = "C(ano) + cr(idade, df=4, constraints='center') + fem"
lin = []
for nome, form in [("With principal diagnosis (primary)", ar.FORM), ("Without principal diagnosis", SEM)]:
    m = Modelo(f"{form} + {ar.HOSP} + off", g)
    lin.append(ar.lin(f"Patient level — {nome}", m, "off", m.dr(dr_dummy("off", ["off"]))))
    m = ModeloFE(f"{form} + off", g)
    lin.append(ar.lin(f"Patient level, hospital FE — {nome}", m, "off", m.dr(dr_dummy("off", ["off"]))))
    m = Modelo(f"{form} + {ar.HOSP} + e10", gi)
    lin.append(ar.lin(f"Between hospitals — {nome}", m, "e10", m.dr(dr_continua("e10", 1))))
    m = ModeloFE(f"{form} + e10", gi)
    lin.append(ar.lin(f"Within hospitals — {nome}", m, "e10", m.dr(dr_continua("e10", 1))))
t = pd.DataFrame(lin).round(4)
t.to_csv(os.path.join(ar.OUT, "R10g_sem_diagnostico.csv"), index=False)
print(t[["modelo", "or", "lo", "hi", "dr_pp", "dr_lo", "dr_hi"]].to_string())
