#!/usr/bin/env bash
# Reproduces every table and figure from the public SIH/SUS files. Run from the repository root.
set -euo pipefail
pip install -r requirements.txt
# 1. Extraction (downloads the monthly RD files; several hours)
python sih_crvm.py --ini 2008 --fim 2025 --ano-ref internacao --saida saida_crvm
python pre2008/sih_crvm_pre2008.py
python pre2008/montar_2000_2025.py          # needs pre2008/mapa_cgc_cnes_manual.csv (see README)
M=saida_crvm/crvm_2000_2025_internacao_microdados.csv
# 2. Coding diagnostics
python artigo/diagnostico_codificacao.py
# 3. Main analyses: primary (pre-specified, 2008-2025 with admission type) and sensitivity windows
python artigo/analise_v2.py --com-carater
python artigo/analise_v2.py --micro $M --ini 2003 --tag _2003_2025_sem_carater
python artigo/analise_v2.py --micro $M --ini 2008 --tag _2008_2025_sem_carater
python artigo/comparar_janelas.py
# 4. Post hoc analyses: primary specification and 2003-2025 sensitivity
REV_INI=2008 REV_CARATER=1 python artigo/analise_adicional.py
REV_INI=2008 REV_CARATER=1 python artigo/analise_adicional_bootfe.py
REV_INI=2008 REV_CARATER=1 python artigo/analise_adicional_semdiag.py
REV_INI=2008 REV_CARATER=1 python artigo/analise_adicional3.py
python artigo/analise_adicional.py
python artigo/analise_adicional_bootfe.py
python artigo/analise_adicional_semdiag.py
python artigo/analise_adicional3.py
python artigo/conferir_adicional.py          # independent re-computation of key estimates
# 5. Figures
(cd artigo/manuscrito && python figuras_manuscrito.py --v2)
