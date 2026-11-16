import marimo

__generated_with = "0.25.2"
app = marimo.App(width="medium")


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
 
    """)
    return


@app.cell
def _():
    import marimo as mo

    return (mo,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    # Projeto — auditoria e pré-triagem das bases

    Auditoria dos sinais RAW da Gutenberg, visualização de um trial, caracterização
    demográfica e coorte elegível da GaitRec e da Gutenberg.

    As regras de elegibilidade e as checagens de integridade vêm do módulo
    `frente2.elegibilidade`, as mesmas usadas pela Frente 2 (notebook `Frente2.py`).
    A análise do artigo (DWT do componente mediolateral) está em
    `Analise_medio_lateral_v6_1.ipynb`.
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## GaitRec metadata definitions — documentation correction (v5.2)

    **This revision does not change the analytical pipeline or recalculate the results.** It corrects the interpretation/documentation of the GaitRec metadata used by the analysis.

    - `SPEED = 1`: slow walking
    - `SPEED = 2`: self-selected walking speed (**condition analyzed in this notebook**)
    - `SPEED = 3`: fast walking
    - `AFFECTED_SIDE = 0`: left
    - `AFFECTED_SIDE = 1`: right
    - `AFFECTED_SIDE = 2`: bilateral

    Therefore, every analysis performed with `SPEED == 2` in this notebook refers to **self-selected walking speed**, not fast walking. Bilateral knee-dysfunction cases are excluded from the side-specific primary comparisons.
    """)
    return


@app.cell
def _():
    import warnings

    import matplotlib.pyplot as plt
    import numpy as np
    import pandas as pd

    from frente2 import config as cfg

    warnings.filterwarnings("ignore", category=RuntimeWarning)
    pd.set_option("display.max_columns", 100)
    pd.set_option("display.width", 180)

    # Saídas desta auditoria ficam fora das pastas de dados brutos
    SAIDA_AUDITORIA = cfg.RAIZ_PROJETO / "resultados_auditoria"
    SAIDA_AUDITORIA.mkdir(exist_ok=True)
    return SAIDA_AUDITORIA, cfg, np, pd, plt


@app.cell
def _(SAIDA_AUDITORIA, cfg, np, pd):
    P = cfg.GUTENBERG; keys=['DATASET_ID','SUBJECT_ID','SESSION_ID','TRIAL_ID']
    files={}
    for axis in ['V','AP','ML']:
     for side in ['left','right']:
      files[(axis,side)]=next(P.glob(f'GRF_F_{axis}_RAW_{side}*.csv'))
    rows=[]; summary=[]; master=None
    for (axis,side),file in files.items():
     df=pd.read_csv(file); X=df.iloc[:,4:].to_numpy(dtype=float); finite=np.isfinite(X); length=finite.sum(axis=1); valid=length>0
     # trailing missing only: each row finite first length positions
     internal=(~np.take_along_axis(finite,np.minimum(np.arange(X.shape[1])[None,:],np.maximum(length[:,None]-1,0)),axis=1)) & (np.arange(X.shape[1])[None,:]<length[:,None])
     x_min=np.nanmin(X,axis=1);x_max=np.nanmax(X,axis=1);peak=np.nanmax(np.abs(X),axis=1)
     zeros=np.nanmax(np.abs(X),axis=1)<1e-9
     result=df[keys].copy(); result[f'n_{axis}_{side}']=length;result[f'peak_{axis}_{side}']=peak;result[f'range_{axis}_{side}']=x_max-x_min
     master=result if master is None else master.merge(result,on=keys,how='outer',validate='one_to_one')
     summary.append({'component':axis,'side':side,'trials':len(df),'min_n':length.min(),'median_n':np.median(length),'max_n':length.max(),'internal_missing_trials':int(internal.any(axis=1).sum()),'all_zero_trials':int(zeros.sum()),'nonfinite_values_inside':int(internal.sum()),'median_peak':np.median(peak),'p99_peak':np.percentile(peak,99),'max_peak':peak.max()})


    meta = pd.read_csv(
        next(P.glob('GRF_metadata*.csv'))
    )

    speed = pd.read_csv(
        next(P.glob('GRF_walking_speed.csv'))
    )


    master = master.merge(
        speed,
        on=keys,
        how='left',
        validate='one_to_one'
    )

    master.to_csv(
        SAIDA_AUDITORIA / 'gutenberg_auditoria_trials.csv',
        index=False
    )

    pd.DataFrame(summary).to_csv(
        SAIDA_AUDITORIA / 'gutenberg_resumo_componentes.csv',
        index=False
    )
    return files, keys, meta, speed, summary


@app.cell
def _(mo, pd, speed, summary):
    #GUTENBERG GAIT DATABASE

    # 1. Preparar os dados
    _resumo = pd.DataFrame(summary).copy()

    # 2. Tabela de integridade dos sinais
    _integridade = _resumo[
        [
            "component",
            "side",
            "trials",
            "min_n",
            "median_n",
            "max_n",
            "internal_missing_trials",
            "all_zero_trials",
            "nonfinite_values_inside"
        ]
    ].copy()

    _integridade.columns = [
        "Componente",
        "Membro",
        "Trials",
        "Mín. amostras",
        "Mediana amostras",
        "Máx. amostras",
        "Falhas internas",
        "Sinais zerados",
        "Valores inválidos"
    ]

    _integridade = _integridade.round(2)

    # 3. Tabela de amplitudes
    _amplitudes = _resumo[
        [
            "component",
            "side",
            "median_peak",
            "p99_peak",
            "max_peak"
        ]
    ].copy()

    _amplitudes.columns = [
        "Componente",
        "Membro",
        "Pico mediano (N)",
        "Percentil 99 (N)",
        "Pico máximo (N)"
    ]

    _amplitudes = _amplitudes.round(2)

    # 4. ESTATÍSTICAS DA VELOCIDADE DA MARCHA, utilizar diretamente o arquivo de velocidades
    _velocidades = speed["WALKING_SPEED"].dropna()

    _velocidade_resumo = pd.DataFrame({
        "Indicador": [
            "Trials totais",
            "Trials com velocidade",
            "Trials sem velocidade",
            "Velocidade média (m/s)",
            "Velocidade mediana (m/s)",
            "Velocidade mínima (m/s)",
            "Velocidade máxima (m/s)"
        ],
        "Resultado": [
            len(speed),
            len(_velocidades),
            speed["WALKING_SPEED"].isna().sum(),
            round(_velocidades.mean(), 3),
            round(_velocidades.median(), 3),
            round(_velocidades.min(), 3),
            round(_velocidades.max(), 3)
        ]
    })

    # 5. Duração dos registros
    _duracao = pd.DataFrame({
        "Componente": _resumo["component"],
        "Membro": _resumo["side"],
        "Duração mínima (s)": _resumo["min_n"] / 250,
        "Duração mediana (s)": _resumo["median_n"] / 250,
        "Duração máxima (s)": _resumo["max_n"] / 250
    }).round(3)


    # EXIBIÇÃO DO PAINEL

    mo.vstack([

        mo.md("# Gutenberg Gait Database"),
        mo.md("## Relatório de avaliação biomecânica"),
        mo.md("### 1. Integridade dos sinais"),
        mo.ui.table(
            _integridade,
            pagination=False,
            selection=None
        ),

        mo.md("### 2. Amplitudes das forças"),
        mo.ui.table(
            _amplitudes,
            pagination=False,
            selection=None
        ),

        mo.md("### 3. Velocidade da marcha"),
        mo.ui.table(
            _velocidade_resumo,
            pagination=False,
            selection=None
        ),

        mo.md("### 4. Duração dos registros"),
        mo.ui.table(
            _duracao,
            pagination=False,
            selection=None
        )
    ])
    return


@app.cell
def _(files, keys, np, pd, plt, speed):
    # ETAPA 3 - VISUALIZAÇÃO BIOMECÂNICA

    def visualizar_trial_gutenberg(arquivos, velocidades, identificadores):

    # 3.1. SELEÇÃO DE UM TRIAL, selecionar um trial com velocidade próxima da mediana

        velocidade_mediana = velocidades["WALKING_SPEED"].median()

        indice = (
            velocidades["WALKING_SPEED"]
            .sub(velocidade_mediana)
            .abs()
            .idxmin()
        )

        trial = velocidades.loc[indice]

    #3.2. CONFIGURAÇÃO DOS GRÁFICOS

        componentes = {
            "V": "Força vertical",
            "AP": "Força anteroposterior",
            "ML": "Força mediolateral"
        }

        membros = {
            "left": "Esquerdo",
            "right": "Direito"
        }

        fig, axes = plt.subplots(
            3,
            2,
            figsize=(15, 11),
            sharex=False
        )

    # Frequência de amostragem
        fs = 250

        #3.3. LEITURA E REPRESENTAÇÃO DOS SINAIS

        for i, (componente, nome) in enumerate(componentes.items()):

            for j, (membro, nome_membro) in enumerate(membros.items()):

                caminho = arquivos[(componente, membro)]

                dados = pd.read_csv(caminho)

                # Identificar o registro correspondente
                mascara = np.ones(len(dados), dtype=bool)

                for chave in identificadores:
                    mascara &= dados[chave].eq(trial[chave]).to_numpy()

                registro = dados.loc[mascara]

                if len(registro) != 1:
                    raise ValueError(
                        f"Registro não encontrado ou duplicado: "
                        f"{componente} - {membro}"
                    )

    # Extrair apenas os valores da força
                sinal = registro.iloc[0, 4:].to_numpy(dtype=float)

    # Remover preenchimento ausente ao final
                sinal = sinal[np.isfinite(sinal)]

    # Construir vetor temporal
                tempo = np.arange(len(sinal)) / fs

    # Gráfico
                ax = axes[i, j]

                ax.plot(
                    tempo,
                    sinal,
                    linewidth=1.8
                )

                ax.axhline(
                    y=0,
                    color="gray",
                    linestyle="--",
                    linewidth=0.8
                )

                ax.set_title(
                    f"{nome} - Membro {nome_membro}",
                    fontsize=11,
                    fontweight="bold"
                )

                ax.set_xlabel("Tempo (s)")
                ax.set_ylabel("Força (N)")

                ax.grid(
                    True,
                    linestyle=":",
                    alpha=0.5
                )

        # --------------------------------------------------------
        # 4. IDENTIFICAÇÃO DO PARTICIPANTE
        # --------------------------------------------------------

        fig.suptitle(
            "Gutenberg Gait Database\n"
            f"Participante: {trial['SUBJECT_ID']} | "
            f"Sessão: {trial['SESSION_ID']} | "
            f"Trial: {trial['TRIAL_ID']} | "
            f"Velocidade: {trial['WALKING_SPEED']:.2f} m/s",
            fontsize=14,
            fontweight="bold"
        )

        fig.tight_layout(rect=[0, 0, 1, 0.93])

        return fig


    # ============================================================
    # EXIBIÇÃO NO MARIMO
    # ============================================================

    painel_sinais_gutenberg = visualizar_trial_gutenberg(
        files,
        speed,
        keys
    )

    painel_sinais_gutenberg
    return


@app.cell
def _(meta):
    def _histogramas_gutenberg(_dados):
        import marimo as _mo
        import pandas as _pd
        import numpy as _np
        import matplotlib.pyplot as _plt
        from scipy.stats import norm as _norm
        from scipy.stats import shapiro as _shapiro

        # Verificação das colunas
        _colunas = {
            "DATASET_ID", "SUBJECT_ID", "SESSION_ID",
            "SEX", "AGE", "BODY_MASS"
        }

        _faltantes = _colunas - set(_dados.columns)

        if _faltantes:
            raise ValueError(
                f"Colunas ausentes: {_faltantes}"
            )

        # Copiar os dados sem modificar o DataFrame original
        _df = _dados.copy()

        # Converter colunas numéricas
        for _col in ["SEX", "AGE", "BODY_MASS"]:
            _df[_col] = _pd.to_numeric(
                _df[_col], errors="coerce"
            )

        # Considerar apenas uma sessão por participante
        _df = (
            _df.sort_values("SESSION_ID")
            .drop_duplicates(
                subset=["DATASET_ID", "SUBJECT_ID"]
            )
        )

        # Gutenberg: SEX = 1 corresponde aos homens
        # Selecionar somente adultos
        _homens = _df[
            (_df["SEX"] == 1) &
            (_df["AGE"] >= 18)
        ].copy()

        if _homens.empty:
            raise ValueError(
                "Nenhum homem adulto encontrado."
            )

        # Função para construir cada histograma
        def _criar_histograma(_coluna, _titulo, _unidade):

            _valores = (
                _homens[_coluna]
                .dropna()
                .to_numpy(dtype=float)
            )

            _n = len(_valores)
            _media = _np.mean(_valores)
            _dp = _np.std(_valores, ddof=1)

            if _n < 3 or _dp == 0:
                raise ValueError(
                    f"Dados insuficientes para {_titulo}"
                )

            # Teste de normalidade
            _estatistica, _p = _shapiro(_valores)

            # Criar figura
            _fig, _ax = _plt.subplots(figsize=(9, 5))

            # Histograma
            _ax.hist(
                _valores,
                bins="auto",
                density=True,
                alpha=0.7,
                edgecolor="white",
                label="Distribuição observada"
            )

            # Distribuição normal teórica ajustada
            _x = _np.linspace(
                min(_valores.min(), _media - 3 * _dp),
                max(_valores.max(), _media + 3 * _dp),
                400
            )

            _y = _norm.pdf(
                _x,
                loc=_media,
                scale=_dp
            )

            _ax.plot(
                _x,
                _y,
                linewidth=2.5,
                label="Distribuição normal ajustada"
            )

            _ax.set_title(
                f"Gutenberg — Distribuição de {_titulo}"
            )

            _ax.set_xlabel(f"{_titulo} ({_unidade})")
            _ax.set_ylabel("Densidade")

            _ax.grid(axis="y", alpha=0.2)
            _ax.legend()

            # Estatísticas no gráfico
            _texto = (
                f"N = {_n}\n"
                f"Média = {_media:.2f}\n"
                f"DP = {_dp:.2f}\n"
                f"Shapiro-Wilk p = {_p:.4g}"
            )

            _ax.text(
                0.98,
                0.97,
                _texto,
                transform=_ax.transAxes,
                ha="right",
                va="top",
                bbox=dict(
                    facecolor="white",
                    alpha=0.85
                )
            )

            _fig.tight_layout()

            return _fig

        # Histograma de idade
        _fig_idade = _criar_histograma(
            "AGE", "Idade", "anos"
        )

        # Histograma de massa corporal
        _fig_massa = _criar_histograma(
            "BODY_MASS", "Massa corporal", "kg"
        )

        # Exibir no marimo
        return _mo.vstack([
            _mo.md(
                f"## Gutenberg — Homens adultos (N = {len(_homens)})"
            ),
            _fig_idade,
            _fig_massa
        ])


    # Substituir pelo nome do DataFrame existente na aplicação
    _histogramas_gutenberg(meta)
    return


@app.cell
def _(mo, np, plt):
    # GAITREC E GUTENBERG — coorte elegível com as regras únicas da Frente 2
    from frente2 import elegibilidade

    selecao = elegibilidade.executar()
    _el = selecao["elegiveis"]

    _fluxo = (
        selecao["fluxo"]
        .pivot_table(index=["base", "grupo"], columns="passo", values="participantes")
        .reset_index()
    )

    # Histogramas de idade das subclasses de joelho (unilaterais, sessão inicial)
    _pac = _el[(_el["base"] == "GaitRec") & (_el["grupo"] != "HC")]
    _bins = np.arange(15, int(np.ceil(_pac["AGE"].max() / 5) * 5) + 6, 5)
    _fig, _eixos = plt.subplots(1, 2, figsize=(12, 4.5), sharey=True)
    for _eixo, _sexo, _nome in zip(_eixos, ["M", "F"], ["Homens", "Mulheres"]):
        for _classe in ["K_P", "K_F", "K_R"]:
            _idades = _pac.loc[(_pac["sexo"] == _sexo) & (_pac["grupo"] == _classe), "AGE"].dropna()
            if not _idades.empty:
                _eixo.hist(_idades, bins=_bins, density=True, histtype="step", linewidth=2,
                           label=f"{_classe} (n={len(_idades)})")
        _eixo.set_title(_nome)
        _eixo.set_xlabel("Idade (anos)")
        _eixo.grid(alpha=0.2)
        _eixo.legend()
    _eixos[0].set_ylabel("Densidade de probabilidade")
    _fig.suptitle("GaitRec — distribuição etária por subclasse (unilaterais, sessão inicial)")
    _fig.tight_layout()

    mo.vstack([
        mo.md("## Coorte elegível — GaitRec e Gutenberg"),
        mo.md(
            "Regras do módulo `frente2.elegibilidade`: categoria exata, idade ≥ 18, "
            "sexo conhecido, SPEED = 2, sessão inicial (SESSION_TYPE = 1), pacientes "
            "unilaterais e uma sessão por participante."
        ),
        mo.md("### Fluxo de seleção (participantes)"),
        mo.ui.table(_fluxo, selection=None, pagination=False),
        mo.md("### Coorte por base, grupo e sexo"),
        mo.ui.table(selecao["resumo"], selection=None, pagination=False),
        mo.as_html(_fig),
    ])
    return elegibilidade, selecao


@app.cell
def _(elegibilidade, mo, pd, selecao):
    # AUDITORIA DE INTEGRIDADE — todas as sessões; nada é excluído por estas sinalizações
    _flags = selecao["flags"]
    _sinal = _flags[_flags["alguma_sinalizacao"]]

    _gr, _gu = elegibilidade.carregar_metadados()
    _meta = pd.concat([_gr, _gu], ignore_index=True)
    _colunas = [
        "base", "participante", "SESSION_ID", "SESSION_DATE", "SEX", "AGE", "HEIGHT",
        "BODY_MASS", "CLASS_LABEL_DETAILED", "AFFECTED_SIDE", "SPEED", "SESSION_TYPE",
    ]
    _detalhes = (
        _meta[_meta["participante"].isin(_sinal["participante"])]
        .reindex(columns=_colunas)
        .sort_values(["participante", "SESSION_ID"])
    )

    mo.vstack([
        mo.md("## Auditoria de integridade dos identificadores"),
        mo.md(
            "Sinaliza sexo, classe ou lado afetado que mudam entre sessões, idade "
            "incompatível com a data da sessão e altura que varia mais de 5 cm. "
            "Nenhum registro é corrigido ou excluído."
        ),
        mo.ui.table(selecao["info"], selection=None, pagination=False),
        mo.md(f"### Participantes sinalizados ({len(_sinal)})"),
        mo.ui.table(_sinal, selection=None),
        mo.md("### Todas as sessões dos participantes sinalizados"),
        mo.ui.table(_detalhes, selection=None),
    ])
    return


if __name__ == "__main__":
    app.run()
