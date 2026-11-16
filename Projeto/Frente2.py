import marimo

__generated_with = "0.25.2"
app = marimo.App(width="medium")


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    # Frente 2 — assimetria do joelho com wavelets e aprendizado de máquina

    Este notebook **só lê e mostra** os resultados gravados em `frente2/resultados`.
    Para (re)calcular tudo, rode no terminal, a partir da pasta `Projeto`:

    ```
    python -m frente2.executar --sensibilidade
    ```

    Etapas: (1) elegibilidade e integridade; (2) DWT db4 nível 4 e características
    por membro; (3) calibração com saudáveis (controle negativo esquerdo × direito)
    e primeira passada em K_R (membro afetado × contralateral).
    """)
    return


@app.cell
def _():
    import json

    import marimo as mo
    import matplotlib.pyplot as plt
    import numpy as np
    import pandas as pd

    from frente2 import config as cfg
    from frente2 import dwt

    R = cfg.RESULTADOS
    return R, dwt, json, mo, np, pd, plt


@app.cell
def _(R, dwt, json, mo):
    _info = json.loads((R / "execucao.json").read_text(encoding="utf-8"))
    mo.md(
        f"**Execução:** {_info['executado_em']} · DWT: {_info['dwt']} · "
        f"componentes GaitRec: {', '.join(_info['componentes_gaitrec'])} · "
        f"Gutenberg: {', '.join(_info['componentes_gutenberg'])}  \n"
        f"Implementação ativa neste ambiente: {dwt.implementacao()} · "
        f"filtros db4 válidos: {all(dwt.checar_filtros().values())}"
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Etapa 1 — Elegibilidade e integridade

    Mesmas regras para todas as tabelas: categoria exata, idade ≥ 18, sexo conhecido,
    SPEED = 2, sessão inicial (SESSION_TYPE = 1), pacientes unilaterais e uma sessão
    por participante. As sinalizações de integridade não excluem ninguém.
    """)
    return


@app.cell
def _(R, mo, pd):
    _fluxo = pd.read_csv(R / "etapa1_fluxo.csv").pivot_table(
        index=["base", "grupo"], columns="passo", values="participantes"
    )
    _resumo = pd.read_csv(R / "etapa1_resumo.csv")
    _sinal = pd.read_csv(R / "etapa1_sinalizacoes.csv")
    mo.vstack([
        mo.md("### Fluxo de seleção (participantes)"),
        mo.ui.table(_fluxo.reset_index(), selection=None, pagination=False),
        mo.md("### Corte elegível por base, grupo e sexo"),
        mo.ui.table(_resumo, selection=None, pagination=False),
        mo.md(f"### Participantes sinalizados ({len(_sinal)})"),
        mo.ui.table(_sinal, selection=None),
    ])
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Etapa 2 — DWT e características

    Por trial: remove o preenchimento NaN, interpola só lacunas internas, divide pelo
    peso corporal da sessão e aplica DWT db4, nível 4, borda simétrica (250 Hz).
    Por subbanda: energia (E), MSWC (média de c²) e energia relativa (Erel); além de
    máximo, mínimo, impulso e duração do apoio. Os trials são promediados por
    participante e membro.
    """)
    return


@app.cell
def _(R, mo, pd):
    membros = pd.read_csv(R / "etapa2_caracteristicas_membro.csv")
    _n = membros.groupby(["base", "lado"]).agg(
        participantes=("participante", "nunique"),
        trials_medio=("n_trials", "mean"),
        trials_min=("n_trials", "min"),
        apoio_medio_s=("dur_s", "mean"),
    ).round(3).reset_index()
    mo.vstack([mo.md("### Cobertura dos sinais"), mo.ui.table(_n, selection=None, pagination=False)])
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Etapa 3 — Calibração e primeira passada

    Cada participante gera dois exemplos espelhados (diferença A − B com rótulo 1 e
    B − A com rótulo 0). Nos saudáveis, A = esquerdo: o esperado é AUC ≈ 0,5.
    Nos pacientes, A = membro afetado. Validação cruzada 5 folds por participante,
    repetida (10× logística, 5× floresta); IC 95% por bootstrap de participantes.

    **Calibração:** subtrai a assimetria esquerdo − direito média dos saudáveis.
    Pacientes usam os saudáveis da GaitRec (mesmo laboratório). No controle
    negativo, cada base é calibrada pela *outra*: se a assimetria basal fosse
    comum às duas bases, a AUC calibrada cairia para perto de 0,5.
    """)
    return


@app.cell
def _(R, mo, pd):
    res = pd.read_csv(R / "etapa3_resultados.csv")
    try:
        _sens = pd.read_csv(R / "sens_ultima_resultados.csv")
        res = pd.concat([res, _sens], ignore_index=True)
    except FileNotFoundError:
        pass
    _cols = ["analise", "tipo", "calibracao", "atributos", "modelo", "n_participantes", "auc",
             "auc_ic95_inf", "auc_ic95_sup", "acerto", "acerto_ic95_inf",
             "acerto_ic95_sup", "p_permutacao"]
    mo.vstack([mo.md("### Resultados"), mo.ui.table(res[[c for c in _cols if c in res]], selection=None)])
    return (res,)


@app.cell
def _(np, plt, res):
    # AUC com IC 95% por análise (logística, B3); cor = sem ou com calibração
    _conj = "B3 clássico + wavelet"
    _d = res[(res["atributos"] == _conj) & (res["modelo"] == "logística")].copy()
    _d["serie"] = np.where(_d["calibracao"] == "bruto", "sem calibração", "calibrado")
    _cores = {"sem calibração": "#2a78d6", "calibrado": "#eb6834"}
    _analises = list(dict.fromkeys(_d["analise"]))
    _fig, _ax = plt.subplots(figsize=(9, 0.55 * len(_analises) + 1.2))
    for _k, (_serie, _cor) in enumerate(_cores.items()):
        _s = _d[_d["serie"] == _serie].set_index("analise").reindex(_analises)
        _y = np.arange(len(_analises)) + (_k - 0.5) * 0.25
        _ax.errorbar(_s["auc"], _y, xerr=[_s["auc"] - _s["auc_ic95_inf"], _s["auc_ic95_sup"] - _s["auc"]],
                     fmt="o", color=_cor, ms=6, lw=2, capsize=0, label=_serie)
    _ax.axvline(0.5, color="#888888", lw=1, ls="--")
    _ax.text(0.505, len(_analises) - 0.6, "acaso", color="#555555", fontsize=9)
    _ax.set_yticks(np.arange(len(_analises)), _analises, fontsize=9)
    _ax.invert_yaxis()
    _ax.set_xlim(0.3, 1.0)
    _ax.set_xlabel("AUC (IC 95%)")
    _ax.set_title(f"Controle negativo e primeira passada — logística, {_conj}", loc="left", fontsize=11)
    _ax.grid(axis="x", alpha=0.2)
    for _sp in ("top", "right"):
        _ax.spines[_sp].set_visible(False)
    _ax.legend(frameon=False, loc="upper left")
    _fig.tight_layout()
    _fig
    return


@app.cell
def _(R, mo, pd):
    _lado = pd.read_csv(R / "etapa3_acerto_por_lado.csv")
    _desc = pd.read_csv(R / "etapa3_assimetria_descritiva.csv")
    _top = (
        _desc.assign(abs_dz=_desc["dz"].abs())
        .sort_values(["analise", "abs_dz"], ascending=[True, False])
        .groupby("analise").head(8)
        .drop(columns="abs_dz")
    )
    mo.vstack([
        mo.md("### Acerto por lado afetado (logística, B3) — se o modelo tivesse aprendido o lado, os dois valores divergiriam"),
        mo.ui.table(_lado, selection=None, pagination=False),
        mo.md("### Atributos com maior assimetria padronizada (dz = média de Δ / DP de Δ)"),
        mo.ui.table(_top, selection=None),
    ])
    return


if __name__ == "__main__":
    app.run()
