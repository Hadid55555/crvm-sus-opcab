# Off-pump CABG in Brazil's public health system, 2003–2025

Code for the article:

> Hadid L, Salek L, Passos FS, Caldonazo T. Trends, centre variation and in-hospital mortality of off-pump coronary artery bypass grafting in Brazil's public health system, 2003–2025.

The code downloads the public hospital discharge files of the Brazilian Unified Health System (SIH/SUS, DATASUS), selects every adult isolated CABG admission, and reproduces all tables and figures of the article and its Supplementary Material. The analysis plan and its dated amendments are in `ANALYSIS_PLAN.md`.

## Data

| Source | Use | Access |
|---|---|---|
| SIH/SUS reduced admission files (RD), DATASUS | Admissions, technique, hospital, in-hospital death | Public. Downloaded by `sih_crvm.py` and `pre2008/sih_crvm_pre2008.py` from the healthbr-data mirror of the official files (checked against published SHA-256 checksums) |
| CIHA (non-SUS admissions), DATASUS | Exploratory analysis of non-SUS admissions | Public; scripts in `ciha/` |
| IBGE population, IPCA price index | Rates and real values | Public; copies in `artigo/dados_ref/` |
| ANS health-plan beneficiaries | Coverage of non-SUS admissions | Public; copy in `ciha/dados_ref/` |

**No record-level data are included.** The SIH files are public, but the files built from them (one row per admission) are created locally by the extraction step and excluded by `.gitignore`. Hospitals are not named: hospital-level intermediate files are not distributed, and published tables use anonymous codes. Records from 2003–2004 that carry only the hospital's tax number are assigned to a hospital with documented rules; the manual correspondence file (`pre2008/mapa_cgc_cnes_manual.csv`) contains hospital names and is available from the corresponding author on request. Without it, those records remain unassigned.

## Requirements

Python 3.11 or later: `pip install -r requirements.txt`.

## How to reproduce

```bash
bash run_all.sh
```

Steps (see `run_all.sh`): extraction of the monthly RD files of the 27 states (January 2000–June 2026; several hours); coding diagnostics; main analyses (`artigo/analise_v2.py`) in the primary specification (2008–2025 with admission type, `artigo/resultados_v2/`) and the sensitivity windows; post hoc analyses (`artigo/analise_adicional*.py`) in both specifications; an independent re-computation of key estimates (`artigo/conferir_adicional.py`); figures (`artigo/manuscrito/figuras_manuscrito.py`).

## Outputs included

The CSV and JSON results of every analysis are included (`artigo/resultados_*`, `ciha/resultados/`), except files that list hospitals. `SHA256SUMS.txt` lists the hashes of all distributed files. Column names and script comments are in Portuguese.

| Folder | Content |
|---|---|
| `artigo/resultados_v2/` | Primary specification (2008–2025, with admission type): Tables 1–2, adoption, variation, mortality |
| `artigo/resultados_v2_2003_2025_sem_carater/` | 2003–2025 without admission type (adoption and variation; mortality sensitivity) |
| `artigo/resultados_v2_2008_2025_sem_carater/` | 2008–2025 without admission type |
| `artigo/resultados_adicionais_2008_2025_carater/` | Post hoc analyses, primary specification (Tables 2–3, S10–S14, S16, S19–S21) |
| `artigo/resultados_adicionais/` | Post hoc analyses, 2003–2025 (Tables S9, S15, S21) |
| `artigo/resultados_diag/` | Coding diagnostics |
| `ciha/resultados/` | Non-SUS admissions (exploratory) |

## Licence

MIT. If you use this code, please cite the article (`CITATION.cff`).
