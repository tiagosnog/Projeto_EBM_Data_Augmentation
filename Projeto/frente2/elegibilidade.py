"""Etapa 1: elegibilidade e integridade dos metadados.

Uma única função define quem entra na análise, com as mesmas regras para
todas as tabelas:

* GaitRec: categoria EXATA de CLASS_LABEL_DETAILED (HC, K_P, K_F ou K_R),
  idade >= 18, SEX conhecido, SPEED = 2 (autosselecionada),
  SESSION_TYPE = 1 (sessão inicial); pacientes só unilaterais
  (AFFECTED_SIDE 0 ou 1); uma sessão por participante (a mais antiga).
* Gutenberg: idade >= 18, SEX conhecido, SESSION_TYPE = 1 (primeira sessão);
  participante = DATASET_ID + SUBJECT_ID.

As checagens de integridade usam TODAS as sessões de cada participante e só
sinalizam: nenhum registro é excluído por causa delas.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from . import config as cfg

GRUPOS = ["HC"] + cfg.SUBCLASSES_K


def _datas(serie: pd.Series) -> pd.Series:
    # GaitRec: "29.01.2015"; Gutenberg: "12.05.2014 10:13:04"
    return pd.to_datetime(serie.astype(str).str.slice(0, 10), format="%d.%m.%Y", errors="coerce")


def carregar_metadados() -> tuple[pd.DataFrame, pd.DataFrame]:
    gr = pd.read_csv(cfg.GAITREC / "GRF_metadata.csv")
    gu = pd.read_csv(cfg.GUTENBERG / "GRF_metadata.csv")
    for df in (gr, gu):
        df["DATA"] = _datas(df["SESSION_DATE"])
        df["CLASS_LABEL_DETAILED"] = df["CLASS_LABEL_DETAILED"].astype(str).str.strip().str.upper()
    gr["base"] = "GaitRec"
    gr["participante"] = "GR-" + gr["SUBJECT_ID"].astype(str)
    gu["base"] = "Gutenberg"
    gu["participante"] = "GU-" + gu["DATASET_ID"].astype(str) + "-" + gu["SUBJECT_ID"].astype(str)
    return gr, gu


def checar_integridade(meta: pd.DataFrame) -> pd.DataFrame:
    """Uma linha por participante, com as sinalizações de integridade."""
    linhas = []
    for pid, g in meta.sort_values(["DATA", "SESSION_ID"]).groupby("participante", sort=False):
        anos = (g["DATA"] - g["DATA"].iloc[0]).dt.days / 365.25
        dif_idade = (g["AGE"] - g["AGE"].iloc[0]) - anos
        altura = g["HEIGHT"].dropna()
        lados = g["AFFECTED_SIDE"].dropna().unique()
        linhas.append(
            {
                "participante": pid,
                "n_sessoes": len(g),
                "sexo_inconsistente": g["SEX"].nunique(dropna=True) > 1,
                "classe_inconsistente": g["CLASS_LABEL_DETAILED"].nunique(dropna=True) > 1,
                "lado_inconsistente": len(lados) > 1,
                "idade_incoerente_com_data": bool(
                    (dif_idade.abs() > cfg.TOL_IDADE_ANOS).any() or (g["AGE"].diff() < 0).any()
                ),
                "altura_instavel": bool(len(altura) > 1 and (altura.max() - altura.min()) > cfg.TOL_ALTURA_CM),
            }
        )
    flags = pd.DataFrame(linhas)
    if meta["base"].iloc[0] == "Gutenberg":
        # Lote de 2016 do DATASET_ID 3: SEX parece preenchido com 0 por padrão.
        primeira = meta.sort_values(["DATA", "SESSION_ID"]).groupby("participante").first()
        lote = (primeira["DATASET_ID"] == 3) & (primeira["DATA"].dt.year == 2016)
        flags["sexo_nao_verificavel_ds3_2016"] = flags["participante"].map(lote).fillna(False)
    colunas = [c for c in flags.columns if c not in ("participante", "n_sessoes")]
    flags["alguma_sinalizacao"] = flags[colunas].any(axis=1)
    return flags


def _uma_sessao(df: pd.DataFrame) -> pd.DataFrame:
    return df.sort_values(["DATA", "SESSION_ID"]).drop_duplicates("participante", keep="first")


def selecionar(gr: pd.DataFrame, gu: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Retorna (participantes elegíveis, fluxo de seleção passo a passo)."""
    fluxo = []

    def registrar(base, grupo, passo, df):
        fluxo.append({"base": base, "grupo": grupo, "passo": passo, "participantes": df["participante"].nunique()})

    # --- GaitRec ---
    elegiveis = []
    for grupo in GRUPOS:
        d = gr[gr["CLASS_LABEL_DETAILED"] == grupo]
        registrar("GaitRec", grupo, "1. categoria exata", d)
        d = d[(d["AGE"] >= cfg.IDADE_MINIMA) & d["SEX"].isin([0, 1])]
        registrar("GaitRec", grupo, "2. adulto, sexo conhecido", d)
        d = d[(d["SPEED"] == cfg.SPEED_AUTOSELECIONADA) & (d["SESSION_TYPE"] == cfg.SESSION_TYPE_INICIAL)]
        registrar("GaitRec", grupo, "3. SPEED=2 e sessão inicial", d)
        if grupo != "HC":
            d = d[d["AFFECTED_SIDE"].isin([0, 1])]
            registrar("GaitRec", grupo, "4. unilateral", d)
        d = _uma_sessao(d)
        registrar("GaitRec", grupo, "5. uma sessão por participante", d)
        elegiveis.append(d.assign(grupo=grupo))

    # --- Gutenberg (todos saudáveis) ---
    d = gu.copy()
    registrar("Gutenberg", "HC", "1. categoria exata", d)
    d = d[(d["AGE"] >= cfg.IDADE_MINIMA) & d["SEX"].isin([0, 1])]
    registrar("Gutenberg", "HC", "2. adulto, sexo conhecido", d)
    d = d[d["SESSION_TYPE"] == cfg.SESSION_TYPE_INICIAL]
    registrar("Gutenberg", "HC", "3. SPEED=2 e sessão inicial", d)
    d = _uma_sessao(d)
    registrar("Gutenberg", "HC", "5. uma sessão por participante", d)
    elegiveis.append(d.assign(grupo="HC"))

    colunas = [
        "base", "grupo", "participante", "DATASET_ID", "SUBJECT_ID", "SESSION_ID", "DATA",
        "SEX", "AGE", "HEIGHT", "BODY_MASS", "BODY_WEIGHT", "AFFECTED_SIDE",
        "SHOD_CONDITION", "READMISSION",
    ]
    tab = pd.concat([e.reindex(columns=colunas) for e in elegiveis], ignore_index=True)
    tab["lado_afetado"] = tab["AFFECTED_SIDE"].map(cfg.LADO_AFETADO)
    tab["sexo"] = tab["SEX"].map({0: "F", 1: "M"})
    return tab, pd.DataFrame(fluxo)


def selecionar_ultima_sessao(gr: pd.DataFrame, coorte_inicial: pd.DataFrame) -> pd.DataFrame:
    """Sensibilidade: última sessão de acompanhamento dos pacientes da coorte principal.

    Mesmas regras (adulto, SPEED=2, unilateral, mesma categoria), mas a sessão
    escolhida é a mais recente e precisa ser posterior à sessão inicial.
    """
    pacientes = coorte_inicial[(coorte_inicial["base"] == "GaitRec") & (coorte_inicial["grupo"] != "HC")]
    inicial = pacientes.set_index("participante")
    d = gr[
        gr["participante"].isin(inicial.index)
        & (gr["SPEED"] == cfg.SPEED_AUTOSELECIONADA)
        & gr["AFFECTED_SIDE"].isin([0, 1])
    ].copy()
    d["grupo"] = d["participante"].map(inicial["grupo"])
    d = d[d["CLASS_LABEL_DETAILED"] == d["grupo"]]
    d = d.sort_values(["DATA", "SESSION_ID"]).drop_duplicates("participante", keep="last")
    d = d[d["SESSION_ID"] != d["participante"].map(inicial["SESSION_ID"])]
    d["dias_desde_inicial"] = (d["DATA"] - pd.to_datetime(d["participante"].map(inicial["DATA"]))).dt.days
    d["lado_afetado"] = d["AFFECTED_SIDE"].map(cfg.LADO_AFETADO)
    d["sexo"] = d["SEX"].map({0: "F", 1: "M"})
    colunas = [
        "base", "grupo", "participante", "DATASET_ID", "SUBJECT_ID", "SESSION_ID", "DATA",
        "SEX", "AGE", "HEIGHT", "BODY_MASS", "BODY_WEIGHT", "AFFECTED_SIDE", "SESSION_TYPE",
        "dias_desde_inicial", "lado_afetado", "sexo",
    ]
    return d.reindex(columns=colunas).reset_index(drop=True)


def checar_primeira_sessao_gutenberg(gu: pd.DataFrame) -> int:
    """Participantes em que SESSION_TYPE=1 não é a sessão mais antiga."""
    s = gu.sort_values(["DATA", "SESSION_ID"])
    mais_antiga = s.groupby("participante")["SESSION_ID"].first()
    tipo1 = s[s["SESSION_TYPE"] == 1].groupby("participante")["SESSION_ID"].first()
    comum = mais_antiga.index.intersection(tipo1.index)
    return int((mais_antiga[comum] != tipo1[comum]).sum())


def executar() -> dict[str, pd.DataFrame]:
    gr, gu = carregar_metadados()
    flags = pd.concat([checar_integridade(gr), checar_integridade(gu)], ignore_index=True)
    elegiveis, fluxo = selecionar(gr, gu)
    elegiveis = elegiveis.merge(flags, on="participante", how="left", validate="one_to_one")

    resumo = (
        elegiveis.groupby(["base", "grupo", "sexo"])
        .agg(
            n=("participante", "nunique"),
            idade_media=("AGE", "mean"),
            idade_dp=("AGE", "std"),
            massa_media=("BODY_MASS", "mean"),
            lado_esq=("lado_afetado", lambda s: int((s == "left").sum())),
            lado_dir=("lado_afetado", lambda s: int((s == "right").sum())),
            sinalizados=("alguma_sinalizacao", "sum"),
        )
        .round(1)
        .reset_index()
    )
    info = pd.DataFrame(
        [
            {"item": "Gutenberg: SESSION_TYPE=1 não é a sessão mais antiga", "valor": checar_primeira_sessao_gutenberg(gu)},
            {"item": "GaitRec: participantes com alguma sinalização (todas as classes)",
             "valor": int(flags[flags.participante.str.startswith("GR")]["alguma_sinalizacao"].sum())},
            {"item": "Gutenberg: participantes com alguma sinalização",
             "valor": int(flags[flags.participante.str.startswith("GU")]["alguma_sinalizacao"].sum())},
        ]
    )
    return {"elegiveis": elegiveis, "fluxo": fluxo, "resumo": resumo, "flags": flags, "info": info}
