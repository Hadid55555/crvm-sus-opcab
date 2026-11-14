# CRVM fora do SUS — CIHA (exploratório, 19/out/2026)

- **Fonte:** CIHA (Comunicação de Internação Hospitalar e Ambulatorial; notificação obrigatória de internações não
  pagas pelo SUS, Portaria GM/MS 1.171/2011). O FTP do DATASUS não é alcançável deste ambiente; usada a compilação
  pública em Parquet do projeto kaggle-datahub (Kaggle `rafatrindade/brazilian-kaggle-datahub`, `ciha/`, CC0, 4.798
  arquivos CIHA{UF}{AAMM}.dbc de jan/2011–2026, sem RR). Só há tamanhos de arquivo no manifesto (sem SHA-256).
- **Extração:** `extrair_ciha.py` (289.317.133 registros lidos → 34.200 com os 4 códigos SIGTAP de CRVM).
  Extrato em `ciha_crvm_extrato_2011_2026.zip`.
- **Análise:** `analise_ciha.py` → `resultados/` e `ciha_crvm_2011_2025.xlsx`.
- **Cuidados de qualidade encontrados:** (1) o campo MORTE da CIHA marca o motivo 51 (encerramento administrativo)
  como óbito — usar motivo de saída 41–43; (2) dias de UTI não vêm preenchidos (0 em 100%); (3) 172 registros
  idênticos a AIH do SIH (excluídos); (4) volume anual cai de ~3.100 (2011) para ~1.500 (2025), compatível com
  sub-registro crescente; (5) não há como saber a cobertura real do setor privado.
- **Uso:** exploratório/contexto; não substitui o SIH nem representa todo o setor privado.
