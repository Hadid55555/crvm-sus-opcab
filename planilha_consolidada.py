#!/usr/bin/env python3
"""Gera a planilha consolidada (formatada, com fórmulas) a partir dos microdados de sih_crvm.py.

Mesmo layout da consolidação anterior (Resumo, Por_ano, Por_UF, Por_regiao, Ano_x_regiao,
Por_sexo, Por_faixa_etaria, Por_carater, Base_cruzada, Leia-me), acrescido de UTI, custo por
componente, hospitais (CNES, com nome do estabelecimento) e comparação de anos de referência.

As tabelas são fórmulas SUMIFS sobre a aba Base_cruzada; cada fórmula é gravada com o valor já
calculado (o arquivo abre com números mesmo em visualizadores que não recalculam).

Uso:
  python planilha_consolidada.py --micro saida_crvm/crvm_2008_2025_internacao_microdados.csv \
      --parciais saida_crvm/parciais --saida saida_crvm/crvm_2008_2025_consolidado.xlsx
"""
import argparse
import glob
import json
import os
import time
import urllib.request

import numpy as np
import pandas as pd
import xlsxwriter
from xlsxwriter.utility import xl_col_to_name as COL

REGIOES = ["Norte", "Nordeste", "Sudeste", "Sul", "Centro-Oeste"]
UF_REGIAO = {"Norte": ["AC", "AM", "AP", "PA", "RO", "RR", "TO"],
             "Nordeste": ["AL", "BA", "CE", "MA", "PB", "PE", "PI", "RN", "SE"],
             "Sudeste": ["ES", "MG", "RJ", "SP"], "Sul": ["PR", "RS", "SC"],
             "Centro-Oeste": ["DF", "GO", "MS", "MT"]}
FAIXAS = ["Até 39", "40–49", "50–59", "60–69", "70–79", "80+"]
CARATER = ["Eletivo", "Urgência", "Outros"]
GRUPOS = ["Com CEC", "Sem CEC"]
OUTROS_VAL = ["VAL_RN", "VAL_ACOMP", "VAL_SADTSR", "VAL_TRANSP", "VAL_OBSANG", "VAL_PED1AC"]
COMPONENTES = [("VAL_SH", "Serviços hospitalares (VAL_SH)"),
               ("VAL_SP", "Serviços profissionais (VAL_SP)"),
               ("VAL_SADT", "SADT (VAL_SADT)"),
               ("VAL_ORTP", "Órteses e próteses (VAL_ORTP)"),
               ("VAL_SANGUE", "Sangue (VAL_SANGUE)"),
               ("VAL_OUTROS", "Outros (RN, acomp., SADT s/ rem., transp., OPM sangue, ped.)"),
               ("VAL_FED", "Complemento federal (VAL_SH_FED + VAL_SP_FED)"),
               ("VAL_GES", "Complemento do gestor (VAL_SH_GES + VAL_SP_GES)")]

# Nomes das colunas da Base_cruzada (ordem = letras A, B, C...)
BASE_COLS = ["Ano", "Região", "UF", "Grupo", "Sexo", "Faixa etária", "Caráter",
             "Nº de AIH", "Óbitos", "Soma dias perm.", "Soma idade (anos)", "Soma dias UTI",
             "AIH com UTI", "Custo total (R$)", "VAL_SH", "VAL_SP", "VAL_SADT", "VAL_ORTP",
             "VAL_SANGUE", "VAL_OUTROS", "VAL_FED", "VAL_GES", "VAL_UTI (incluso em VAL_SH)"]
B = {n: COL(i) for i, n in enumerate(BASE_COLS)}
SOMA = {"n": "Nº de AIH", "ob": "Óbitos", "dp": "Soma dias perm.", "id": "Soma idade (anos)",
        "uti": "Soma dias UTI", "cuti": "AIH com UTI", "ct": "Custo total (R$)"}

AZUL_ESC, AZUL, VERDE, CINZA = "#1F3864", "#2E75B6", "#548235", "#595959"


def baixar_json(url, tentativas=4):
    req = urllib.request.Request(url, headers={"User-Agent": "sih-crvm/1.0"})
    for i in range(tentativas):
        try:
            with urllib.request.urlopen(req, timeout=60) as r:
                return json.load(r)
        except Exception:
            if i == tentativas - 1:
                return None
            time.sleep(2 ** (i + 1))


def referencias(cnes_lista, cache_dir):
    """Nomes de estabelecimentos (API CNES do Ministério da Saúde) e municípios (API IBGE)."""
    pc = os.path.join(cache_dir, "ref_cnes.json")
    cn = json.load(open(pc)) if os.path.exists(pc) else {}
    for c in cnes_lista:
        if c not in cn and str(c).isdigit():
            j = baixar_json(f"https://apidadosabertos.saude.gov.br/cnes/estabelecimentos/{int(c)}")
            cn[c] = {"nome_fantasia": (j or {}).get("nome_fantasia"),
                     "razao_social": (j or {}).get("nome_razao_social"),
                     "esfera": (j or {}).get("descricao_esfera_administrativa")}
    json.dump(cn, open(pc, "w"), ensure_ascii=False)
    pm = os.path.join(cache_dir, "ref_municipios.json")
    if not os.path.exists(pm):
        j = baixar_json("https://servicodados.ibge.gov.br/api/v1/localidades/municipios")
        json.dump({str(m["id"])[:6]: m["nome"] for m in (j or [])}, open(pm, "w"), ensure_ascii=False)
    return cn, json.load(open(pm))


def preparar(df):
    df = df.copy()
    df["Ano"] = df["ano_ref"].astype(int)
    df["Região"] = df["regiao"]
    df["UF"] = df["uf"]
    df["Grupo"] = df["grupo"]
    df["Sexo"] = df["SEXO"].astype(str).str.strip().map({"1": "Masculino", "2": "Feminino", "3": "Feminino"}).fillna("Ignorado")
    df["Faixa etária"] = pd.cut(df["idade_anos"], [-1, 39, 49, 59, 69, 79, 999], labels=FAIXAS).astype(str)
    car = df["CAR_INT"].astype(str).str.strip().str.zfill(2)
    df["Caráter"] = car.map({"01": "Eletivo", "02": "Urgência"}).fillna("Outros")
    df["VAL_OUTROS"] = df[OUTROS_VAL].fillna(0).sum(axis=1)
    df["VAL_FED"] = df[["VAL_SH_FED", "VAL_SP_FED"]].fillna(0).sum(axis=1)
    df["VAL_GES"] = df[["VAL_SH_GES", "VAL_SP_GES"]].fillna(0).sum(axis=1)
    df["com_uti"] = (df["UTI_MES_TO"].fillna(0) > 0).astype(int)
    return df


def base_cruzada(df):
    dims = BASE_COLS[:7]
    g = df.groupby(dims, observed=True).agg(**{
        "Nº de AIH": ("N_AIH", "size"), "Óbitos": ("MORTE", "sum"),
        "Soma dias perm.": ("DIAS_PERM", "sum"), "Soma idade (anos)": ("idade_anos", "sum"),
        "Soma dias UTI": ("UTI_MES_TO", "sum"), "AIH com UTI": ("com_uti", "sum"),
        "Custo total (R$)": ("VAL_TOT", "sum"), "VAL_SH": ("VAL_SH", "sum"), "VAL_SP": ("VAL_SP", "sum"),
        "VAL_SADT": ("VAL_SADT", "sum"), "VAL_ORTP": ("VAL_ORTP", "sum"), "VAL_SANGUE": ("VAL_SANGUE", "sum"),
        "VAL_OUTROS": ("VAL_OUTROS", "sum"), "VAL_FED": ("VAL_FED", "sum"), "VAL_GES": ("VAL_GES", "sum"),
        "VAL_UTI (incluso em VAL_SH)": ("VAL_UTI", "sum")}).reset_index()
    for c in BASE_COLS[7:]:
        g[c] = g[c].astype(float).round(2)
    return g.sort_values(dims).reset_index(drop=True)


class Livro:
    def __init__(self, path, base):
        self.wb = xlsxwriter.Workbook(path)
        self.base = base
        self.n = len(base) + 1  # última linha da base (1-indexada, com cabeçalho)
        w = self.wb
        F = dict(font_name="Arial", font_size=10)
        self.f = {
            "titulo": w.add_format(dict(F, bold=True, font_size=14, font_color=AZUL_ESC)),
            "sub": w.add_format(dict(F, font_size=9, font_color=CINZA)),
            "hdr": w.add_format(dict(F, bold=True, font_color="white", bg_color=AZUL_ESC, align="center", valign="vcenter", text_wrap=True)),
            "hdr_com": w.add_format(dict(F, bold=True, font_color="white", bg_color=AZUL, align="center", valign="vcenter")),
            "hdr_sem": w.add_format(dict(F, bold=True, font_color="white", bg_color=VERDE, align="center", valign="vcenter")),
            "hdr_tot": w.add_format(dict(F, bold=True, font_color="white", bg_color=CINZA, align="center", valign="vcenter")),
            "hdr2": w.add_format(dict(F, bold=True, bg_color="#DDEBF7", align="center", valign="vcenter", text_wrap=True)),
            "rot": w.add_format(dict(F, align="left")),
            "rotb": w.add_format(dict(F, bold=True, bg_color="#DDEBF7")),
            "tot_rot": w.add_format(dict(F, bold=True, bg_color="#F2F2F2")),
            "conf": w.add_format(dict(F, font_size=9, font_color=CINZA, num_format='0;\\-0;"OK"', align="center")),
            "conf_rot": w.add_format(dict(F, font_size=9, font_color=CINZA)),
            "nota": w.add_format(dict(F, font_size=9, font_color=CINZA, text_wrap=True, valign="top")),
            "alerta": w.add_format(dict(F, bold=True, font_color="#C00000")),
            "secao": w.add_format(dict(F, bold=True, font_color="white", bg_color=AZUL_ESC)),
            "texto": w.add_format(dict(F, text_wrap=True, valign="top")),
        }
        nf = {"int": "#,##0", "pct": "0.0%", "dec": "0.0", "rs": "#,##0", "rs2": "#,##0.00"}
        for k, v in nf.items():
            self.f[k] = w.add_format(dict(F, num_format=v))
            self.f[k + "_b"] = w.add_format(dict(F, num_format=v, bold=True, bg_color="#F2F2F2"))
            self.f[k + "_fix"] = w.add_format(dict(F, num_format=v, font_color="#0000FF"))
            self.f[k + "_fixb"] = w.add_format(dict(F, num_format=v, font_color="#0000FF", bold=True, bg_color="#F2F2F2"))

    def rng(self, nome):
        c = B[nome]
        return f"Base_cruzada!${c}$2:${c}${self.n}"

    def sumifs(self, nome, crits):
        """crits: lista de (coluna_base, ref_celula, valor). Retorna (fórmula, valor)."""
        if not crits:
            return f"SUM({self.rng(nome)})", float(self.base[nome].sum())
        m = np.ones(len(self.base), bool)
        partes = []
        for col, ref, val in crits:
            partes.append(f"{self.rng(col)},{ref}")
            m &= (self.base[col] == val).to_numpy()
        return f"SUMIFS({self.rng(nome)},{','.join(partes)})", float(self.base.loc[m, nome].sum())

    def cabecalho(self, ws, titulo, sub):
        ws.hide_gridlines(2)
        ws.write(0, 0, titulo, self.f["titulo"])
        ws.write(1, 0, sub, self.f["sub"])
        ws.set_row(0, 18)

    # Bloco de métricas: (rótulo, chave, formato)
    METR = [("Nº de AIH\n(procedimentos)", "n", "int"), ("Óbitos", "ob", "int"),
            ("Mortalidade\n(%)", "mort", "pct"), ("Idade média\n(anos)", "id", "dec"),
            ("Permanência\nmédia (dias)", "dp", "dec"), ("UTI média\n(dias)", "uti", "dec"),
            ("% com UTI", "cuti", "pct"), ("Custo total\n(R$)", "ct", "rs"),
            ("Custo médio\npor AIH (R$)", "cm", "rs")]

    def bloco(self, ws, r, c0, crits, grupo, bold=False):
        """Escreve as métricas na linha r a partir da coluna c0; grupo = célula do cabeçalho do grupo."""
        cr = list(crits) + ([("Grupo", grupo[0], grupo[1])] if grupo else [])
        sfx = "_b" if bold else ""
        ncell = f"{COL(c0)}{r + 1}"
        fn, vn = self.sumifs(SOMA["n"], cr)
        vals = {}
        for j, (_, k, fmt) in enumerate(self.METR):
            col = c0 + j
            if k == "n":
                f, v = fn, vn
            elif k in ("ob", "ct"):
                f, v = self.sumifs(SOMA[k], cr)
            elif k == "mort":
                f = f'IF({ncell}=0,"",{COL(c0 + 1)}{r + 1}/{ncell})'
                v = vals["ob"] / vn if vn else ""
            elif k == "cm":
                f = f'IF({ncell}=0,"",{COL(c0 + 7)}{r + 1}/{ncell})'
                v = vals["ct"] / vn if vn else ""
            else:
                f0, v0 = self.sumifs(SOMA[k], cr)
                f = f'IF({ncell}=0,"",{f0}/{ncell})'
                v = v0 / vn if vn else ""
            vals[k] = v
            ws.write_formula(r, col, "=" + f, self.f[fmt + sfx], v)
        return vals

    def tabela_dim(self, nome, titulo, sub, dims, linhas, notas=()):
        """dims: [(rótulo, coluna_base)]; linhas: lista de tuplas de valores das dimensões."""
        ws = self.wb.add_worksheet(nome)
        self.cabecalho(ws, titulo, sub)
        nd, nm = len(dims), len(self.METR)
        r_g, r_h = 3, 4
        blocos = [("Com CEC", "hdr_com"), ("Sem CEC", "hdr_sem"), ("Total", "hdr_tot")]
        for b, (g, fm) in enumerate(blocos):
            c0 = nd + b * nm
            ws.merge_range(r_g, c0, r_g, c0 + nm - 1, g, self.f[fm])
            for j, (lab, _, _) in enumerate(self.METR):
                ws.write(r_h, c0 + j, lab, self.f["hdr2"])
        for i, (lab, _) in enumerate(dims):
            ws.write(r_h, i, lab, self.f["hdr"])
        c_pct = nd + 3 * nm
        ws.write(r_h, c_pct, "% Sem CEC\nno total de AIH", self.f["hdr"])
        ws.set_row(r_h, 42)
        r = r_h + 1
        for valores in linhas:
            crits = []
            for i, ((_, bc), v) in enumerate(zip(dims, valores)):
                ws.write(r, i, v, self.f["rot"])
                crits.append((bc, f"${COL(i)}{r + 1}", v))
            res = []
            for b, (g, _) in enumerate(blocos):
                c0 = nd + b * nm
                grp = (f"{COL(c0)}${r_g + 1}", g) if g != "Total" else None
                res.append(self.bloco(ws, r, c0, crits, grp))
            nt, ns = res[2]["n"], res[1]["n"]
            ws.write_formula(r, c_pct, f'=IF({COL(nd + 2 * nm)}{r + 1}=0,"",{COL(nd + nm)}{r + 1}/{COL(nd + 2 * nm)}{r + 1})',
                             self.f["pct"], ns / nt if nt else "")
            r += 1
        r_tot = r
        ws.write(r_tot, 0, "TOTAL", self.f["tot_rot"])
        for i in range(1, nd):
            ws.write_blank(r_tot, i, None, self.f["tot_rot"])
        res = []
        for b, (g, _) in enumerate(blocos):
            c0 = nd + b * nm
            grp = (f"{COL(c0)}${r_g + 1}", g) if g != "Total" else None
            res.append(self.bloco(ws, r_tot, c0, [], grp, bold=True))
        ws.write_formula(r_tot, c_pct, f'=IF({COL(nd + 2 * nm)}{r_tot + 1}=0,"",{COL(nd + nm)}{r_tot + 1}/{COL(nd + 2 * nm)}{r_tot + 1})',
                         self.f["pct_b"], res[1]["n"] / res[2]["n"])
        rc = r_tot + 1
        ws.write(rc, 0, "Conferência (soma das linhas − total da base; deve ser 0)", self.f["conf_rot"])
        for b in range(3):
            for j, (_, k, _) in enumerate(self.METR):
                if k in ("n", "ob", "ct"):
                    c = COL(nd + b * nm + j)
                    ws.write_formula(rc, nd + b * nm + j, f"=ROUND(SUM({c}{r_h + 2}:{c}{r_tot})-{c}{r_tot + 1},2)", self.f["conf"], 0)
        for i, nt in enumerate(notas):
            ws.write(rc + 2 + i, 0, nt, self.f["sub"])
        ws.set_column(0, nd - 1, 14)
        ws.set_column(nd, c_pct, 12.5)
        ws.freeze_panes(r_h + 1, nd)
        return ws


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--micro", default="saida_crvm/crvm_2008_2025_internacao_microdados.csv")
    ap.add_argument("--parciais", nargs="+", default=["saida_crvm/parciais"])
    ap.add_argument("--fonte-txt", default="Fonte: 5.859 arquivos RD<UF><AA><MM> (27 UFs, competências jan/2008 a jan/2026), lidos do espelho público healthbr-data "
                    "(Parquet convertido 1:1 dos .dbc do FTP do DATASUS, SHA-256 conferido por arquivo). Nenhum arquivo faltante.")
    ap.add_argument("--codigos-txt", default="Com CEC = procedimentos 0406010927 e 0406010935.  Sem CEC = 0406010943 e 0406010951.")
    ap.add_argument("--sem-carater", action="store_true", help="não gera a aba Por_carater")
    ap.add_argument("--ano-ini", type=int, default=None)
    ap.add_argument("--ano-fim", type=int, default=None)
    ap.add_argument("--serie", default=None, help="CSV opcional com a série anual completa (aba Serie_anual)")
    ap.add_argument("--extra", nargs="*", default=[], help="abas extras 'Nome|Título|arquivo.csv' (tabelas fixas)")
    ap.add_argument("--notas-extra", nargs="*", default=[], help="parágrafos extras no Leia-me")
    ap.add_argument("--saida", default="saida_crvm/crvm_2008_2025_consolidado.xlsx")
    ap.add_argument("--ano-col", default="ano_internacao")
    args = ap.parse_args()

    df = pd.read_csv(args.micro, dtype={"N_AIH": str, "PROC_REA": str, "CNES": str, "MUNIC_MOV": str,
                                        "MUNIC_RES": str, "SEXO": str, "CAR_INT": str, "IDENT": str,
                                        "CID_MORTE": str}, low_memory=False)
    df["ano_ref"] = df[args.ano_col]
    if args.ano_ini or args.ano_fim:
        df = df[df["ano_ref"].between(args.ano_ini or 0, args.ano_fim or 9999)]
    df = df[df["grupo"].isin(GRUPOS)].copy()  # 'Não separável' (código único de RVM até fev/2002) fica só na Serie_anual
    df = preparar(df)
    ini, fim = int(df["Ano"].min()), int(df["Ano"].max())
    anos = list(range(ini, fim + 1))
    base = base_cruzada(df)
    L = Livro(args.saida, base)
    f = L.f
    N = len(df)
    ref_txt = {"ano_internacao": "ano da internação (DT_INTER)", "ano_saida": "ano da saída (DT_SAIDA)",
               "ano_cmpt": "ano de competência (processamento)"}[args.ano_col]

    # ---------------- Resumo ----------------
    ws = L.wb.add_worksheet("Resumo")
    L.cabecalho(ws, f"Revascularização miocárdica no SIH/SUS, {ini}–{fim}: com CEC × sem CEC",
                f"Resumo geral por {ref_txt}. Fórmulas sobre 'Base_cruzada'; valores em azul (medianas) vêm "
                "direto dos registros individuais. Mortalidade bruta, sem ajuste de risco.")
    cols = [("Grupo", None), ("Nº de AIH\n(procedimentos)", "int"), ("% do total", "pct"), ("Óbitos", "int"),
            ("Mortalidade\n(%)", "pct"), ("Idade média\n(anos)", "dec"), ("Idade mediana\n(anos)", "dec"),
            ("Permanência\nmédia (dias)", "dec"), ("Permanência\nmediana (dias)", "dec"),
            ("UTI média\n(dias)", "dec"), ("UTI mediana\n(dias)", "dec"), ("% com UTI", "pct"),
            ("Custo total\n(R$)", "rs"), ("Custo médio\npor AIH (R$)", "rs"), ("Custo mediano\npor AIH (R$)", "rs")]
    for j, (lab, _) in enumerate(cols):
        ws.write(3, j, lab, f["hdr"])
    ws.set_row(3, 42)
    linhas = [("Com CEC", df[df.Grupo == "Com CEC"]), ("Sem CEC", df[df.Grupo == "Sem CEC"]), ("Total", df)]
    for i, (g, sub) in enumerate(linhas):
        r = 4 + i
        bold = g == "Total"
        sfx = "_b" if bold else ""
        ws.write(r, 0, g, f["tot_rot"] if bold else f["rotb"])
        cr = [] if bold else [("Grupo", f"$A{r + 1}", g)]
        fn, vn = L.sumifs("Nº de AIH", cr)
        fo, vo = L.sumifs("Óbitos", cr)
        fi, vi = L.sumifs("Soma idade (anos)", cr)
        fd, vd = L.sumifs("Soma dias perm.", cr)
        fu, vu = L.sumifs("Soma dias UTI", cr)
        fcu, vcu = L.sumifs("AIH com UTI", cr)
        fc, vc = L.sumifs("Custo total (R$)", cr)
        R = r + 1
        ws.write_formula(r, 1, "=" + fn, f["int" + sfx], vn)
        ws.write_formula(r, 2, f"=B{R}/$B$7", f["pct" + sfx], vn / N)
        ws.write_formula(r, 3, "=" + fo, f["int" + sfx], vo)
        ws.write_formula(r, 4, f"=D{R}/B{R}", f["pct" + sfx], vo / vn)
        ws.write_formula(r, 5, f"={fi}/B{R}", f["dec" + sfx], vi / vn)
        ws.write_number(r, 6, float(sub["idade_anos"].median()), f["dec_fix" + ("b" if bold else "")])
        ws.write_formula(r, 7, f"={fd}/B{R}", f["dec" + sfx], vd / vn)
        ws.write_number(r, 8, float(sub["DIAS_PERM"].median()), f["dec_fix" + ("b" if bold else "")])
        ws.write_formula(r, 9, f"={fu}/B{R}", f["dec" + sfx], vu / vn)
        ws.write_number(r, 10, float(sub["UTI_MES_TO"].median()), f["dec_fix" + ("b" if bold else "")])
        ws.write_formula(r, 11, f"={fcu}/B{R}", f["pct" + sfx], vcu / vn)
        ws.write_formula(r, 12, "=" + fc, f["rs" + sfx], vc)
        ws.write_formula(r, 13, f"=M{R}/B{R}", f["rs" + sfx], vc / vn)
        ws.write_number(r, 14, float(sub["VAL_TOT"].median()), f["rs_fix" + ("b" if bold else "")])
    ws.write(7, 0, "Conferência (Com CEC + Sem CEC − Total; deve ser 0)", f["conf_rot"])
    for c in ("B", "D", "M"):
        ws.write_formula(7, ord(c) - 65, f"=ROUND({c}5+{c}6-{c}7,2)", f["conf"], 0)

    # Composição do custo
    r0 = 10
    ws.write(r0 - 1, 0, "Composição do custo médio por AIH (R$) — componentes somam o custo total (VAL_TOT)", f["secao"])
    for j in range(1, 8):
        ws.write_blank(r0 - 1, j, None, f["secao"])
    hd = ["Componente", "Com CEC\n(R$/AIH)", "Sem CEC\n(R$/AIH)", "Total\n(R$/AIH)", "% do custo\n(Total)",
          "Com CEC\ntotal (R$)", "Sem CEC\ntotal (R$)", "Total\n(R$)"]
    for j, h in enumerate(hd):
        ws.write(r0, j, h, f["hdr"])
    ws.set_row(r0, 30)
    comp_rows = COMPONENTES + [("VAL_UTI (incluso em VAL_SH)", "dentro de VAL_SH: UTI (VAL_UTI) — informativo, já somado acima")]
    rr = r0 + 1
    acum = {j: 0.0 for j in range(1, 8)}
    for k, lab in comp_rows:
        is_info = k.startswith("VAL_UTI")
        ws.write(rr, 0, lab, f["sub"] if is_info else f["rot"])
        R = rr + 1
        for j, (g, nref) in enumerate([("Com CEC", "$B$5"), ("Sem CEC", "$B$6"), (None, "$B$7")]):
            cr = [("Grupo", f'"{g}"', g)] if g else []
            ft, vt = L.sumifs(k, cr)
            vn = {"Com CEC": linhas[0][1], "Sem CEC": linhas[1][1], None: df}[g]
            ws.write_formula(rr, 5 + j, "=" + ft, f["rs"], vt)
            ws.write_formula(rr, 1 + j, f"={COL(5 + j)}{R}/{nref}", f["rs"], vt / len(vn))
            if not is_info:
                acum[5 + j] += vt
                acum[1 + j] += vt / len(vn)
        pc = float(base[k].sum()) / float(base["Custo total (R$)"].sum())
        ws.write_formula(rr, 4, f"=H{R}/$M$7", f["pct"], pc)
        if not is_info:
            acum[4] += pc
        rr += 1
    tot_ref = {1: linhas[0][1]["VAL_TOT"].mean(), 2: linhas[1][1]["VAL_TOT"].mean(), 3: df["VAL_TOT"].mean(),
               5: linhas[0][1]["VAL_TOT"].sum(), 6: linhas[1][1]["VAL_TOT"].sum(), 7: df["VAL_TOT"].sum()}
    n_comp = len(COMPONENTES)
    ws.write(rr, 0, "Soma dos componentes", f["tot_rot"])
    for j in range(1, 8):
        c = COL(j)
        ws.write_formula(rr, j, f"=SUM({c}{r0 + 2}:{c}{r0 + 1 + n_comp})", f["pct_b"] if j == 4 else f["rs_b"],
                         acum[j])
    ws.write(rr + 1, 0, "Custo total (VAL_TOT)", f["tot_rot"])
    for j, ref in zip(range(1, 8), ["N5", "N6", "N7", None, "M5", "M6", "M7"]):
        if ref:
            ws.write_formula(rr + 1, j, f"={ref}", f["rs_b"], float(tot_ref[j]))
    ws.write(rr + 2, 0, "Diferença (componentes − VAL_TOT)", f["conf_rot"])
    for j in (1, 2, 3, 5, 6, 7):
        c = COL(j)
        ws.write_formula(rr + 2, j, f"=ROUND({c}{rr + 1}-{c}{rr + 2},2)", f["rs2"], round(acum[j] - float(tot_ref[j]), 2))
    notas = [
        "VAL_UTI está CONTIDO em VAL_SH (teste empírico na aba Qualidade: VAL_UTI ≤ VAL_SH em 99,99% das AIH com UTI; "
        "VAL_TOT = soma dos componentes SEM VAL_UTI em 99,1%). Não some VAL_UTI ao total.",
        "A partir de 2023, parte das AIH recebe complemento federal (VAL_SH_FED + VAL_SP_FED), que entra no VAL_TOT mas não em VAL_SH/VAL_SP; "
        "por isso o custo médio sobe em 2023–2025. Para comparar com anos anteriores, olhe também SH + SP sem complemento.",
        "Valores em R$ nominais (sem correção pela inflação). Para comparar anos, deflacione (ex.: IPCA).",
    ]
    for i, t in enumerate(notas):
        ws.merge_range(rr + 4 + i, 0, rr + 4 + i, 14, t, f["nota"])
        ws.set_row(rr + 4 + i, 26)
    ws.set_column(0, 0, 44)
    ws.set_column(1, 14, 13)
    ws.freeze_panes(4, 1)

    # ---------------- Tabelas por dimensão ----------------
    rotulo_ano = {"ano_internacao": "internação", "ano_saida": "saída", "ano_cmpt": "competência"}[args.ano_col]
    sub_pad = (f"Unidade de análise: AIH com um dos 4 procedimentos de revascularização miocárdica como procedimento "
               f"realizado. Ano = {ref_txt}. Valores nominais. UTI = UTI_MES_TO. Fórmulas sobre 'Base_cruzada'.")
    L.tabela_dim("Por_ano", f"Por ano de {rotulo_ano}", sub_pad, [("Ano", "Ano")], [(a,) for a in anos])
    ufs_presentes = set(base["UF"])
    linhas_uf = [(rg, uf) for rg in REGIOES for uf in UF_REGIAO[rg] if uf in ufs_presentes]
    ausentes = [uf for rg in REGIOES for uf in UF_REGIAO[rg] if uf not in ufs_presentes]
    L.tabela_dim("Por_UF", "Por estado (local de atendimento)", sub_pad, [("Região", "Região"), ("UF", "UF")], linhas_uf,
                 [f"UF sem nenhuma AIH de revascularização como local de atendimento: {', '.join(ausentes)}."] if ausentes else [])
    L.tabela_dim("Por_regiao", "Por região (local de atendimento)", sub_pad, [("Região", "Região")], [(r,) for r in REGIOES])
    L.tabela_dim("Ano_x_regiao", f"Ano de {rotulo_ano} × região", sub_pad, [("Ano", "Ano"), ("Região", "Região")],
                 [(a, r) for a in anos for r in REGIOES])
    sexos = [s for s in ["Masculino", "Feminino", "Ignorado"] if s in set(base["Sexo"])]
    L.tabela_dim("Por_sexo", "Por sexo", sub_pad, [("Sexo", "Sexo")], [(s,) for s in sexos])
    L.tabela_dim("Por_faixa_etaria", "Por faixa etária", sub_pad, [("Faixa etária", "Faixa etária")], [(x,) for x in FAIXAS])
    if not args.sem_carater:
      L.tabela_dim("Por_carater", "Por caráter da internação", sub_pad, [("Caráter", "Caráter")], [(x,) for x in CARATER],
                 ["Eletivo = CAR_INT 01; Urgência = 02; Outros = demais códigos (03–06, acidentes/lesões)."]
                 + (["ATENÇÃO: antes de 2008 o caráter seguia outra codificação (código 05 em ~64% das AIH, 02 quase ausente); "
                     "a proporção de urgência de 2003–2007 não é clinicamente plausível e não deve ser interpretada nem comparada."]
                    if ini < 2008 else []))

    # ---------------- Custos por componente × ano ----------------
    ws = L.wb.add_worksheet("Custo_componentes")
    L.cabecalho(ws, f"Custo médio por AIH (R$) por componente, por ano de {rotulo_ano}",
                "Fórmulas sobre 'Base_cruzada'. Os componentes somam o custo total (VAL_TOT). VAL_UTI já está incluído em VAL_SH.")
    comps = [k for k, _ in COMPONENTES]
    nomes = ["Serv. hospitalares\n(VAL_SH)", "Serv. profissionais\n(VAL_SP)", "SADT\n(VAL_SADT)", "Órteses/próteses\n(VAL_ORTP)",
             "Sangue\n(VAL_SANGUE)", "Outros", "Compl. federal\n(SH+SP _FED)", "Compl. gestor\n(SH+SP _GES)",
             "Custo total\n(VAL_TOT)", "dos quais UTI\n(VAL_UTI)"]
    ks = comps + ["Custo total (R$)", "VAL_UTI (incluso em VAL_SH)"]
    nm = len(ks)
    for b, (g, fm) in enumerate([("Com CEC", "hdr_com"), ("Sem CEC", "hdr_sem"), ("Total", "hdr_tot")]):
        c0 = 1 + b * nm
        ws.merge_range(3, c0, 3, c0 + nm - 1, g, f[fm])
        for j, lab in enumerate(nomes):
            ws.write(4, c0 + j, lab, f["hdr2"])
    ws.write(4, 0, "Ano", f["hdr"])
    ws.set_row(4, 42)
    for i, a in enumerate(anos + ["TOTAL"]):
        r = 5 + i
        tot = a == "TOTAL"
        ws.write(r, 0, a, f["tot_rot"] if tot else f["rot"])
        for b, g in enumerate(GRUPOS + [None]):
            c0 = 1 + b * nm
            cr = ([] if tot else [("Ano", f"$A{r + 1}", a)]) + ([("Grupo", f"{COL(c0)}$4", g)] if g else [])
            fn, vn = L.sumifs("Nº de AIH", cr)
            for j, k in enumerate(ks):
                fk, vk = L.sumifs(k, cr)
                ws.write_formula(r, c0 + j, f'=IF({fn}=0,"",{fk}/{fn})', f["rs_b" if tot else "rs"], vk / vn if vn else "")
    ws.set_column(0, 0, 10)
    ws.set_column(1, 3 * nm, 13)
    ws.freeze_panes(5, 1)

    # ---------------- Hospitais ----------------
    cn, mun = referencias(sorted(df["CNES"].astype(str).unique()), os.path.dirname(args.saida) or ".")
    h = df.groupby("CNES").agg(UF=("UF", lambda s: s.mode().iat[0]), Região=("Região", lambda s: s.mode().iat[0]),
                               mun=("MUNIC_MOV", lambda s: s.mode().iat[0]), primeiro=("Ano", "min"),
                               ultimo=("Ano", "max"), anos=("Ano", "nunique"), n=("N_AIH", "size"),
                               n_com=("Grupo", lambda s: (s == "Com CEC").sum()),
                               n_sem=("Grupo", lambda s: (s == "Sem CEC").sum()), ob=("MORTE", "sum"),
                               dp=("DIAS_PERM", "mean"), uti=("UTI_MES_TO", "mean"), cm=("VAL_TOT", "mean"),
                               ct=("VAL_TOT", "sum")).reset_index().sort_values("n", ascending=False)
    ws = L.wb.add_worksheet("Por_hospital")
    L.cabecalho(ws, "Hospitais que realizaram revascularização miocárdica pelo SUS",
                f"Um estabelecimento (CNES) por linha, {ini}–{fim}. Nome: API de dados abertos do CNES (Ministério da Saúde); "
                "município: IBGE. Valores fixos (azul) calculados dos microdados.")
    hh = ["CNES", "Nome fantasia", "Razão social", "Esfera", "UF", "Região", "Município", "Primeiro\nano", "Último\nano",
          "Anos\nativos", "Nº de AIH", "Com CEC", "Sem CEC", "Óbitos", "Mortalidade\n(%)", "Permanência\nmédia (dias)",
          "UTI média\n(dias)", "Custo médio\npor AIH (R$)", "Custo total\n(R$)"]
    for j, x in enumerate(hh):
        ws.write(3, j, x, f["hdr"])
    ws.set_row(3, 42)
    for i, x in enumerate(h.itertuples(index=False)):
        r = 4 + i
        info = cn.get(str(x.CNES), {})
        vals = [str(x.CNES), info.get("nome_fantasia") or "", info.get("razao_social") or "", info.get("esfera") or "",
                x.UF, x.Região, mun.get(str(x.mun)[:6], str(x.mun))]
        for j, v in enumerate(vals):
            ws.write_string(r, j, v, f["rot"])
        nums = [(x.primeiro, "int_fix"), (x.ultimo, "int_fix"), (x.anos, "int_fix"), (x.n, "int_fix"),
                (x.n_com, "int_fix"), (x.n_sem, "int_fix"), (x.ob, "int_fix")]
        for j, (v, fm) in enumerate(nums):
            ws.write_number(r, 7 + j, float(v), f[fm])
        ws.write_formula(r, 14, f"=N{r + 1}/K{r + 1}", f["pct"], x.ob / x.n)
        ws.write_number(r, 15, float(x.dp), f["dec_fix"])
        ws.write_number(r, 16, float(x.uti), f["dec_fix"])
        ws.write_number(r, 17, float(x.cm), f["rs_fix"])
        ws.write_number(r, 18, float(x.ct), f["rs_fix"])
    rt = 4 + len(h)
    ws.write(rt, 0, "TOTAL", f["tot_rot"])
    for j in range(1, 10):
        ws.write_blank(rt, j, None, f["tot_rot"])
    for j, v in zip((10, 11, 12, 13, 18), (h.n.sum(), h.n_com.sum(), h.n_sem.sum(), h.ob.sum(), h.ct.sum())):
        c = COL(j)
        ws.write_formula(rt, j, f"=SUM({c}5:{c}{rt})", f["rs_b" if j == 18 else "int_b"], float(v))
    ws.write_formula(rt, 14, f"=N{rt + 1}/K{rt + 1}", f["pct_b"], h.ob.sum() / h.n.sum())
    ws.write(rt + 1, 0, "Conferência (Nº de AIH − total da Base_cruzada; deve ser 0)", f["conf_rot"])
    ws.write_formula(rt + 1, 10, f"=ROUND(K{rt + 1}-SUM({L.rng('Nº de AIH')}),2)", f["conf"], 0)
    ws.autofilter(3, 0, rt - 1, len(hh) - 1)
    ws.set_column(0, 0, 9)
    ws.set_column(1, 2, 38)
    ws.set_column(3, 6, 12)
    ws.set_column(6, 6, 22)
    ws.set_column(7, 18, 11)
    ws.freeze_panes(4, 2)

    # ---------------- Comparação de anos de referência ----------------
    ws = L.wb.add_worksheet("Comparacao_anos")
    L.cabecalho(ws, "Nº de AIH por ano de referência: internação × saída × competência",
                f"Mesmo conjunto de arquivos. Cada coluna recorta {ini}–{fim} pelo respectivo ano.")
    p = pd.concat([pd.read_csv(x, dtype=str, keep_default_na=False, usecols=["ANO_CMPT", "DT_INTER", "DT_SAIDA", "PROC_REA"])
                   for pasta in args.parciais for x in glob.glob(os.path.join(pasta, "RD*.csv"))], ignore_index=True)
    p["g"] = np.where(p.PROC_REA.isin(["0406010927", "0406010935", "32011016", "32039018", "48010073", "48010081"]),
                      "Com CEC", "Sem CEC")
    p = p[p.DT_INTER.str.match(r"^\d{8}$") & p.DT_SAIDA.str.match(r"^\d{8}$")]
    tabs = {nm_: p.assign(a=p[c].str[:4].astype(int)).query(f"{ini} <= a <= {fim}").groupby(["a", "g"]).size()
            for nm_, c in [("Internação", "DT_INTER"), ("Saída", "DT_SAIDA"), ("Competência", "ANO_CMPT")]}
    hd = ["Ano"] + [f"{g}\n{t}" for g in ["Com CEC", "Sem CEC", "Total"] for t in tabs]
    for j, x in enumerate(hd):
        ws.write(3, j, x, f["hdr"])
    ws.set_row(3, 30)
    for i, a in enumerate(anos):
        ws.write(4 + i, 0, a, f["rot"])
        j = 1
        for g in ["Com CEC", "Sem CEC", None]:
            for t, s in tabs.items():
                v = s.get((a, g), 0) if g else s.get((a, "Com CEC"), 0) + s.get((a, "Sem CEC"), 0)
                ws.write_number(4 + i, j, int(v), f["int_fix"])
                j += 1
    rt = 4 + len(anos)
    ws.write(rt, 0, "TOTAL", f["tot_rot"])
    tot_cols = [sum(tabs[t].get((a, g), 0) if g else tabs[t].get((a, "Com CEC"), 0) + tabs[t].get((a, "Sem CEC"), 0)
                    for a in anos) for g in ["Com CEC", "Sem CEC", None] for t in tabs]
    for j in range(1, len(hd)):
        c = COL(j)
        ws.write_formula(rt, j, f"=SUM({c}5:{c}{rt})", f["int_b"], int(tot_cols[j - 1]))
    ws.write(rt + 2, 0, "As diferenças vêm de AIHs processadas em competência posterior à internação (ou internações "
                        "iniciadas no fim do ano anterior). A consolidação anterior usava competência; esta usa internação.", f["sub"])
    ws.set_column(0, 0, 8)
    ws.set_column(1, len(hd), 12)
    ws.freeze_panes(4, 1)

    # ---------------- Base_cruzada ----------------
    ws = L.wb.add_worksheet("Base_cruzada")
    for j, c in enumerate(BASE_COLS):
        ws.write(0, j, c, f["hdr"])
    ws.set_row(0, 30)
    for i, row in enumerate(base.itertuples(index=False), start=1):
        for j, v in enumerate(row):
            if j < 7:
                ws.write(i, j, v, f["rot"])
            else:
                ws.write_number(i, j, float(v), f["rs2_fix" if j >= 13 else "int_fix"])
    ws.autofilter(0, 0, len(base), len(BASE_COLS) - 1)
    ws.freeze_panes(1, 0)
    ws.set_column(0, 6, 12)
    ws.set_column(7, len(BASE_COLS) - 1, 14)

    # ---------------- Série anual completa (opcional) ----------------
    if args.serie:
        se = pd.read_csv(args.serie)
        ws = L.wb.add_worksheet("Serie_anual")
        L.cabecalho(ws, "Série anual completa de revascularização miocárdica no SIH/SUS (todas as AIH extraídas)",
                    "Por ano de internação. 'Não separável' = AIH no código único de RVM (32011016) processadas antes de "
                    "mar/2002, quando ainda não existia código de RVM sem CEC. % sem CEC calculada entre as AIH separáveis.")
        for j, c in enumerate(se.columns):
            ws.write(3, j, c, f["hdr"])
        ws.set_row(3, 42)
        for i, row in enumerate(se.itertuples(index=False)):
            for j, v in enumerate(row):
                if isinstance(v, str):
                    ws.write_string(4 + i, j, v, f["rot"])
                elif pd.isna(v):
                    ws.write_blank(4 + i, j, None, f["rot"])
                else:
                    nm_ = se.columns[j]
                    ws.write_number(4 + i, j, float(v), f["dec_fix" if ("%" in nm_ or "médi" in nm_) else "int_fix"])
        ws.set_column(0, len(se.columns), 14)
        ws.freeze_panes(4, 1)

    # ---------------- Abas extras (tabelas fixas de outras análises) ----------------
    for spec in args.extra:
        nome_aba, titulo, arq = spec.split("|")
        ex = pd.read_csv(arq)
        ws = L.wb.add_worksheet(nome_aba[:31])
        L.cabecalho(ws, titulo, f"Tabela fixa gerada por outra análise ({os.path.relpath(arq)}). Ver Leia-me.")
        for j, c in enumerate(ex.columns):
            ws.write(3, j, c, f["hdr"])
        ws.set_row(3, 42)
        for i, row in enumerate(ex.itertuples(index=False)):
            for j, v in enumerate(row):
                if isinstance(v, str):
                    ws.write_string(4 + i, j, v, f["rot"])
                elif pd.isna(v):
                    ws.write_blank(4 + i, j, None, f["rot"])
                else:
                    ws.write_number(4 + i, j, float(v), f["dec_fix"] if float(v) != int(float(v)) else f["int_fix"])
        ws.set_column(0, len(ex.columns), 14)
        ws.freeze_panes(4, 1)

    # ---------------- Leia-me ----------------
    ws = L.wb.add_worksheet("Leia-me")
    ws.hide_gridlines(2)
    ws.set_column(0, 0, 130)
    ws.write(0, 0, "Leia-me", f["titulo"])
    txt = [
        ("O QUE É ESTE ARQUIVO", None),
        (f"Consolidação dos microdados do SIH/SUS (AIH reduzida, RD) de {ini}–{fim}: {N:,} AIH com revascularização miocárdica "
         "como procedimento realizado.".replace(",", "."), "t"),
        (args.codigos_txt, "t"),
        (args.fonte_txt, "t"),
    ] + [(t, "t") for t in args.notas_extra] + [
        ("COMO LER AS ABAS", None),
        ("Resumo: totais gerais por grupo e composição do custo.  Por_ano, Por_UF, Por_regiao, Ano_x_regiao, Por_sexo, Por_faixa_etaria, "
         "Por_carater: mesmas métricas, lado a lado para Com CEC, Sem CEC e Total.", "t"),
        ("Custo_componentes: custo médio por AIH de cada componente, por ano e grupo.  Por_hospital: cada estabelecimento (CNES) com nome, "
         "período de atividade e métricas.  Comparacao_anos: contagem por ano de internação, de saída e de competência.", "t"),
        ("Base_cruzada: base agregada (ano × região × UF × grupo × sexo × faixa × caráter) que alimenta as tabelas por fórmulas SUMIFS. "
         "Use filtro ou tabela dinâmica nela para qualquer outro cruzamento.", "t"),
        ("Linhas de 'Conferência': soma das linhas menos o total da base; mostram OK quando a diferença é 0.", "t"),
        ("DEFINIÇÕES", None),
        (f"Ano: {ref_txt}." + (" A consolidação anterior usava o ano de competência (processamento); recortada por competência, "
         "esta base reproduz exatamente a anterior (383.420 AIH, mesmos óbitos, dias, idades e custos)." if ini == 2008 else
         " Valores em R$ nominais de cada ano (antes de 2008, tabelas SIH antigas)."), "t"),
        ("Nº de AIH (procedimentos): número de registros. Não há N_AIH repetido nem AIH de longa permanência (IDENT=5) no recorte.", "t"),
        ("Estado/UF e Região: local de atendimento (2 primeiros dígitos do código IBGE do município do estabelecimento).", "t"),
        ("Sexo: 1 = masculino, 3 (ou 2) = feminino.  Idade: IDADE + COD_IDADE convertidos em anos.", "t"),
        ("Mortalidade: óbitos hospitalares (MORTE=1) ÷ nº de AIH. Bruta, sem ajuste de risco; não é mortalidade em 30 dias.", "t"),
        ("Permanência: DIAS_PERM.  UTI: UTI_MES_TO (dias de UTI na AIH); '% com UTI' = AIH com UTI_MES_TO > 0.", "t"),
        ("Custo: VAL_TOT, em R$ nominais. Componentes: VAL_SH (hospitalar: diárias, materiais, medicamentos, UTI), VAL_SP "
         "(profissionais: equipe), VAL_SADT, VAL_ORTP, VAL_SANGUE, Outros (VAL_RN, VAL_ACOMP, VAL_SADTSR, VAL_TRANSP, VAL_OBSANG, "
         "VAL_PED1AC), complemento federal (VAL_SH_FED + VAL_SP_FED) e complemento do gestor (VAL_SH_GES + VAL_SP_GES). "
         "VAL_UTI está contido em VAL_SH e não deve ser somado.", "t"),
        ("LIMITAÇÕES", None),
        ("Apenas o procedimento principal da AIH (arquivos RD). Valores nominais: deflacione para comparar anos. "
         "CID_MORTE vem vazio em ~59% dos óbitos.", "t"),
        ("Células em azul = valores fixos (medianas, hospitais e a base agregada); em preto = fórmulas.", "t"),
    ]
    for i, (t, k) in enumerate(txt, start=2):
        ws.write(i, 0, t, f["secao"] if k is None else f["texto"])

    L.wb.close()
    print("ok:", args.saida, "| AIH:", N, "| linhas Base_cruzada:", len(base), "| hospitais:", len(h))


if __name__ == "__main__":
    main()
