#!/usr/bin/env python3
"""Figuras 1–4 do manuscrito (2003–2025, sem caráter). Lê só os CSV de resultados; não recalcula nada.

Saída: artigo/manuscrito/figuras/Figure{1..4}.png (300 dpi) e .pdf
v2: python figuras_manuscrito.py --v2 → figuras_v2/ (Figura 1 com painel regional e marca do
complemento federal de mai/2023; Figura 4 com o estimando institucional em diferença de risco e a linha de referência
do efeito esperado se o OR individual fosse causal). Lê também artigo/resultados_adicionais/.
"""
import sys
import os

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import matplotlib.ticker  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

RAIZ = os.path.dirname(os.path.abspath(__file__))
RES = os.path.join(RAIZ, "..", "resultados_v2_2003_2025_sem_carater")
V2 = "--v2" in sys.argv
REV = os.path.join(RAIZ, "..", "resultados_adicionais")
OUT = os.path.join(RAIZ, "figuras_v2" if V2 else "figuras")
os.makedirs(OUT, exist_ok=True)

ON, OFF = "#2a78d6", "#eb6834"  # paleta categórica validada (slots 1 e 2)
INK, MUTED, GRID = "#1f1f1e", "#6b6a63", "#e4e3dd"
plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 9, "axes.edgecolor": MUTED, "axes.labelcolor": INK,
                     "xtick.color": MUTED, "ytick.color": MUTED, "axes.spines.top": False, "axes.spines.right": False,
                     "axes.grid": True, "grid.color": GRID, "grid.linewidth": 0.6, "axes.axisbelow": True})


def ler(n):
    return pd.read_csv(os.path.join(RES, n + ".csv"), dtype={"CNES": str})


def salvar(fig, nome):
    for ext in ("png", "pdf"):
        fig.savefig(os.path.join(OUT, f"{nome}.{ext}"), dpi=300, bbox_inches="tight")
    plt.close(fig)


def fig1():
    a = ler("A1_adocao_anual")
    fig, ax = plt.subplots(figsize=(6.8, 3.6))
    ax.fill_between(a.ano, a.lo, a.hi, color=OFF, alpha=0.18, lw=0)
    ax.plot(a.ano, a.pct_off, color=OFF, lw=2, marker="o", ms=4, label="Crude (95% CI)")
    ax.plot(a.ano, a.pct_off_ajustada_casemix, color=INK, lw=1.2, ls="--", label="Case-mix adjusted")
    for x, t in [(2004.5, "Table change\n(Jul 2004)"), (2008, "SIGTAP\n(Jan 2008)")]:
        ax.axvline(x, color=MUTED, lw=0.8, ls=":")
        ax.text(x + 0.15, 14.6, t, fontsize=7, color=MUTED, va="top")
    ax.set_ylim(0, 15)
    ax.set_xlim(2002.5, 2025.5)
    ax.set_xticks(range(2003, 2026, 2))
    ax.set_ylabel("Off-pump CABG, % of isolated CABG")
    ax.set_xlabel("Year of admission")
    ax.legend(frameon=False, loc="lower right")
    salvar(fig, "Figure1")


def fig1_v2():
    a = ler("A1_adocao_anual")
    r = pd.read_csv(os.path.join(REV, "R7a_serie_regiao_natureza.csv"))
    r = r[r.variavel == "Region"]
    fig, (ax, bx) = plt.subplots(1, 2, figsize=(7.4, 3.4), gridspec_kw={"width_ratios": [1.15, 1]})
    ax.fill_between(a.ano, a.lo, a.hi, color=OFF, alpha=0.18, lw=0)
    ax.plot(a.ano, a.pct_off, color=OFF, lw=2, marker="o", ms=3.5, label="Crude (95% CI)")
    ax.plot(a.ano, a.pct_off_ajustada_casemix, color=INK, lw=1.2, ls="--", label="Case-mix adjusted")
    for x, t in [(2004.5, "Table\nchange"), (2008, "SIGTAP"), (2023.33, "Federal\ncomplement")]:
        ax.axvline(x, color=MUTED, lw=0.8, ls=":")
        ax.text(x + 0.15, 14.6, t, fontsize=6.5, color=MUTED, va="top")
    ax.set_ylim(0, 15)
    ax.set_xlim(2002.5, 2025.5)
    ax.set_xticks(range(2003, 2026, 4))
    ax.set_ylabel("Off-pump CABG, % of isolated CABG")
    ax.set_xlabel("Year of admission")
    ax.set_title("A  Brazil", loc="left", fontsize=9, fontweight="bold")
    ax.legend(frameon=False, loc="lower right", fontsize=7)
    cores = {"Northeast": "#eb6834", "South": "#2a78d6", "Southeast": "#1f1f1e", "Center-West": "#8a5cc2", "North": "#3a9e6f"}
    for reg, c in cores.items():
        x = r[r.grupo == reg].sort_values("ano")
        sm = x.n_off.rolling(3, center=True, min_periods=2).sum() / x.n.rolling(3, center=True, min_periods=2).sum() * 100
        bx.plot(x.ano, sm, color=c, lw=1.6, label=reg.replace("Center", "Centre"))
    bx.set_xlim(2002.5, 2025.5)
    bx.set_xticks(range(2003, 2026, 4))
    bx.set_ylim(0, 40)
    bx.set_xlabel("Year of admission")
    bx.set_title("B  By region (3-year moving average)", loc="left", fontsize=9, fontweight="bold")
    bx.legend(frameon=False, fontsize=7, loc="upper left")
    fig.tight_layout()
    salvar(fig, "Figure1")


def fig4_v2():
    # 3ª rodada: especificação pré-especificada (2008–2025 com caráter)
    R08 = os.path.join(RAIZ, "..", "resultados_v2")
    V08 = os.path.join(RAIZ, "..", "resultados_adicionais_2008_2025_carater")
    i = pd.read_csv(os.path.join(R08, "M6_individual.csv")).set_index("modelo")
    rb = pd.read_csv(os.path.join(V08, "R1b_exposicao_predeterminada.csv"))
    ra = pd.read_csv(os.path.join(V08, "R3a_institucional_volume_maturidade.csv"))
    mu = pd.read_csv(os.path.join(V08, "F1_mundlak_within_between.csv"))
    mu = mu[mu.exposicao.str.startswith("Same-year") & (mu.amostra == "All hospitals")].set_index("componente")
    import json
    js = json.load(open(os.path.join(V08, "resultados_adicionais.json")))
    e = js["evento_Signed (per +10 pp change)"]
    s = pd.read_csv(os.path.join(R08, "M4_institucional.csv"))
    def rb_(mod, am, fe):
        x = rb[(rb.modelo == mod) & (rb.amostra == am) & (rb.efeitos_fixos_hospital == fe)].iloc[0]
        return x.dr_pp, x.dr_lo, x.dr_hi
    pri = s.iloc[0]
    fe = s.iloc[5]
    vs = ra[ra.modelo.str.contains("state")].iloc[0]
    linhas = [
        ("Between hospitals", None),
        ("  Primary (region, legal status)", (pri.dr_pp, pri.dr_lo, pri.dr_hi)),
        ("  + volume, programme age, state", (vs.dr_pp, vs.dr_lo, vs.dr_hi)),
        ("  Within–between model, between coefficient", tuple(mu.loc["Between (hospital mean), per +10 pp", ["dr_pp", "dr_lo", "dr_hi"]])),
        ("Within hospitals (hospital fixed effects)", None),
        ("  Same-year proportion", (fe.dr_pp, fe.dr_lo, fe.dr_hi)),
        ("  Prior-year proportion", rb_("Prior-year % (predetermined; prior year ≥10 CABG)", "Own sample", True)),
        ("  Trailing 3-year proportion", rb_("Trailing 3-year % (t−1 to t−3; predetermined)", "Own sample", True)),
        ("  Within–between model, within coefficient", tuple(mu.loc["Within (yearly deviation), per +10 pp", ["dr_pp", "dr_lo", "dr_hi"]])),
        ("  Event study, years 0–3 after change", (e["media_pos_0a3_pp"], e["lo"], e["hi"])),
    ]
    ind = i.loc["Adjusted: + region + legal status [PRIMARY individual]"]
    fig, ax = plt.subplots(figsize=(7.0, 3.9))
    y = len(linhas)
    ys, labels = [], []
    for nome, r in linhas:
        y -= 1
        ys.append(y)
        labels.append(nome)
        if r is None:
            continue
        ax.plot([r[1], r[2]], [y, y], color=INK, lw=1.4)
        ax.plot(r[0], y, "s", color=OFF, ms=7, mec="white", mew=0.8)
        ax.text(0.47, y, f"{r[0]:.2f} ({r[1]:.2f} to {r[2]:.2f})", va="center", fontsize=8, color=INK)
    esp = ind.dr_pp / 10
    ax.axvline(0, color=MUTED, lw=0.9, ls="--")
    ax.axvline(esp, color=ON, lw=1.2, ls="-.")
    ax.text(esp - 0.01, len(linhas) - 0.35, f"Expected if patient-level\nassociation were causal ({esp:.2f})", fontsize=7, color=ON, ha="right", va="bottom")
    ax.set_xlim(-0.5, 0.45)
    ax.set_yticks(ys)
    ax.set_yticklabels(labels, fontsize=8)
    for t in ax.get_yticklabels():
        if not t.get_text().startswith("  "):
            t.set_fontweight("bold")
    ax.grid(axis="y", visible=False)
    ax.set_ylim(-0.6, len(linhas) + 0.6)
    ax.set_xlabel("Risk difference for in-hospital death, percentage points per +10 pp hospital off-pump proportion")
    ax.text(0.47, len(linhas) - 0.4, "RD (95% CI)", fontsize=8, fontweight="bold", color=INK)
    salvar(fig, "Figure4")


def fig2():
    v = ler("V2_hospitais_pct_bruta_ajustada").sort_values("pct_off_ajustada").reset_index(drop=True)
    nac = 10.19
    fig, (ax, bx) = plt.subplots(1, 2, figsize=(7.2, 3.6), gridspec_kw={"width_ratios": [1.35, 1]})
    s = 6 + 60 * np.sqrt(v.n / v.n.max())
    ax.scatter(np.arange(1, len(v) + 1), v.pct_off_ajustada, s=s, color=OFF, alpha=0.75, edgecolor="white", lw=0.4)
    ax.axhline(nac, color=INK, lw=0.9, ls="--")
    ax.text(5, nac * 1.25, f"National {nac:.1f}%", fontsize=7, color=INK)
    ax.set_yscale("log")
    ax.set_ylim(0.01, 100)
    ax.set_yticks([0.01, 0.1, 1, 10, 100])
    ax.set_yticklabels(["0.01", "0.1", "1", "10", "100"])
    ax.set_xlabel("Hospitals ranked by adjusted off-pump %")
    ax.set_ylabel("Adjusted off-pump %, log scale")
    ax.set_title("A  Hospital-level use (n = 307)", loc="left", fontsize=9, color=INK)
    c = ler("A3_categorias_hospital").iloc[::-1].reset_index(drop=True)
    lab = [f"{r.cat}\n{r.hospitais} hospitals" for r in c.itertuples()]
    y = np.arange(len(c))
    bx.barh(y + 0.2, c.pct_das_aih, height=0.38, color="#b9b8b0", label="All CABG")
    bx.barh(y - 0.2, c.pct_das_off, height=0.38, color=OFF, label="Off-pump CABG")
    for yi, p1, p2 in zip(y, c.pct_das_aih, c.pct_das_off):
        bx.text(p1 + 1, yi + 0.2, f"{p1:.0f}%", va="center", fontsize=7, color=MUTED)
        bx.text(p2 + 1, yi - 0.2, f"{p2:.0f}%", va="center", fontsize=7, color=INK)
    bx.set_yticks(y)
    bx.set_yticklabels(lab, fontsize=7)
    bx.set_xlim(0, 80)
    bx.set_xlabel("Share of national procedures, %")
    bx.set_title("B  Hospital category (2003–2025 off-pump %)", loc="left", fontsize=9, color=INK)
    bx.legend(frameon=False, fontsize=7, loc="lower right")
    bx.grid(axis="y", visible=False)
    fig.tight_layout()
    salvar(fig, "Figure2")


def fig3():
    m = ler("M1_mortalidade_anual_por_tecnica")
    fig, ax = plt.subplots(figsize=(6.8, 3.6))
    for tec, cor in [("On-pump", ON), ("Off-pump", OFF)]:
        x = m[m.tecnica == tec]
        ax.fill_between(x.ano, x.lo, x.hi, color=cor, alpha=0.15, lw=0)
        ax.plot(x.ano, x.mort_pct, color=cor, lw=2, marker="o", ms=3.5, label=f"{tec} (crude, 95% CI)")
        ax.plot(x.ano, x.mort_ajustada_pct, color=cor, lw=1, ls="--", label=f"{tec} (adjusted)")
        ax.text(2025.4, x.mort_pct.iloc[-1], tec, color=INK, fontsize=8, va="center")
    ax.set_ylim(0, 8)
    ax.set_xlim(2002.5, 2027)
    ax.set_xticks(range(2003, 2026, 2))
    ax.set_ylabel("In-hospital mortality, %")
    ax.set_xlabel("Year of admission")
    ax.legend(frameon=False, ncol=2, fontsize=7, loc="lower left")
    salvar(fig, "Figure3")


def fig4():
    i = ler("M6_individual").set_index("modelo")
    s = ler("M4_institucional")
    linhas = [
        ("Patient level (off- vs on-pump)", None),
        ("  Crude", i.loc["Crude"]),
        ("  Adjusted (case mix, region, legal status)", i.loc["Adjusted: + region + legal status [PRIMARY individual]"]),
        ("  Adjusted + hospital fixed effects", i.loc["Adjusted + hospital FE"]),
        ("Hospital strategy (per +10 pp off-pump)", None),
        ("  Between hospitals [primary]", s.iloc[0]),
        ("  Within hospitals (hospital fixed effects)", s.iloc[5]),
    ]
    fig, ax = plt.subplots(figsize=(6.8, 3.2))
    y = len(linhas)
    ys, labels = [], []
    for nome, r in linhas:
        y -= 1
        ys.append(y)
        labels.append(nome)
        if r is None:
            continue
        cor = OFF
        ax.plot([r["lo"], r["hi"]], [y, y], color=INK, lw=1.4)
        ax.plot(r["or"], y, "s", color=cor, ms=7, mec="white", mew=0.8)
        d = 3 if "hospitals" in nome else 2
        ax.text(1.62, y, f"{r['or']:.{d}f} ({r['lo']:.{d}f}–{r['hi']:.{d}f})", va="center", fontsize=8, color=INK)
    ax.axvline(1, color=MUTED, lw=0.9, ls="--")
    ax.set_xscale("log")
    ax.set_xlim(0.4, 1.6)
    ax.set_xticks([0.5, 0.75, 1, 1.25])
    ax.set_xticklabels(["0.50", "0.75", "1.00", "1.25"])
    ax.xaxis.set_minor_formatter(matplotlib.ticker.NullFormatter())
    ax.set_yticks(ys)
    ax.set_yticklabels(labels, fontsize=8)
    for t in ax.get_yticklabels():
        if not t.get_text().startswith("  "):
            t.set_fontweight("bold")
    ax.grid(axis="y", visible=False)
    ax.set_xlabel("Odds ratio for in-hospital death (95% CI)")
    ax.text(1.62, len(linhas) - 0.4, "OR (95% CI)", fontsize=8, fontweight="bold", color=INK)
    salvar(fig, "Figure4")


if __name__ == "__main__":
    (fig1_v2 if V2 else fig1)()
    fig2()
    fig3()
    (fig4_v2 if V2 else fig4)()
    print(sorted(os.listdir(OUT)))
