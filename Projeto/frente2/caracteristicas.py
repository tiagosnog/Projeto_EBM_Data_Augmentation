"""Etapa 2: pipeline DWT e tabela de características.

Regras herdadas da análise v6 do artigo:

* sinal RAW, sem filtragem adicional;
* remove o preenchimento NaN do início e do fim, interpola só lacunas internas;
* DWT db4, nível 4, 250 Hz, borda simétrica, calculada trial a trial;
* MSWC = média dos coeficientes ao quadrado em cada subbanda;
* valores dos trials são promediados por participante e membro.

Novidades da Frente 2: sinal dividido pelo peso corporal da sessão
(BODY_WEIGHT, em N), energia e energia relativa por subbanda e atributos
clássicos (máximo, mínimo, impulso e duração do apoio).
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from . import config as cfg
from .dwt import wavedec

CHAVES = {"GaitRec": ["SUBJECT_ID", "SESSION_ID", "TRIAL_ID"],
          "Gutenberg": ["DATASET_ID", "SUBJECT_ID", "SESSION_ID", "TRIAL_ID"]}


def preparar_sinal(valores: np.ndarray) -> tuple[np.ndarray | None, int]:
    """Remove NaN das pontas e interpola lacunas internas. Retorna (sinal, n_lacunas)."""
    finitos = np.isfinite(valores)
    if finitos.sum() < 2:
        return None, 0
    pos = np.flatnonzero(finitos)
    x = np.asarray(valores[pos[0] : pos[-1] + 1], dtype=float)
    lacunas = ~np.isfinite(x)
    n_lacunas = int(lacunas.sum())
    if n_lacunas:
        ok = np.flatnonzero(~lacunas)
        x[lacunas] = np.interp(np.flatnonzero(lacunas), ok, x[ok])
    return x, n_lacunas


def atributos_sinal(x: np.ndarray, comp: str) -> dict[str, float]:
    coefs = wavedec(x, cfg.WAVELET, cfg.NIVEL, cfg.MODO_BORDA)
    energias = np.array([float(np.sum(c**2)) for c in coefs])
    total = energias.sum()
    f: dict[str, float] = {}
    for banda, c, e in zip(cfg.SUBBANDAS, coefs, energias):
        f[f"{comp}_{banda}_E"] = e
        f[f"{comp}_{banda}_MSWC"] = float(np.mean(c**2))
        f[f"{comp}_{banda}_Erel"] = e / total if total > 0 else np.nan
    f[f"{comp}_max"] = float(x.max())
    f[f"{comp}_min"] = float(x.min())
    f[f"{comp}_impulso"] = float(x.sum() / cfg.FS)
    f[f"{comp}_media"] = float(x.mean())
    f[f"{comp}_n"] = len(x)
    return f


def _ler_sessoes(arquivo, sessoes: set, tamanho_bloco: int = 4000):
    """Lê só as linhas das sessões selecionadas, em blocos (economiza memória)."""
    for bloco in pd.read_csv(arquivo, chunksize=tamanho_bloco, low_memory=False):
        sel = bloco[bloco["SESSION_ID"].isin(sessoes)]
        if len(sel):
            yield sel


def extrair_trials(base: str, elegiveis: pd.DataFrame, componentes: list[str] | None = None) -> pd.DataFrame:
    """Uma linha por participante x sessão x trial x lado, com os atributos de cada componente."""
    el = elegiveis[elegiveis["base"] == base]
    peso = el.set_index("SESSION_ID")["BODY_WEIGHT"].to_dict()
    participante = el.set_index("SESSION_ID")["participante"].to_dict()
    sessoes = set(peso)
    chaves = CHAVES[base]
    componentes = componentes or cfg.componentes_disponiveis(base)

    partes = []
    for lado in cfg.LADOS:
        por_comp = []
        for comp in componentes:
            linhas = []
            for bloco in _ler_sessoes(cfg.arquivo_sinal(base, comp, lado), sessoes):
                ids = bloco[chaves].to_numpy()
                valores = bloco.drop(columns=chaves).to_numpy(dtype=float)
                for chave, v in zip(ids, valores):
                    sess = int(chave[chaves.index("SESSION_ID")])
                    x, n_lacunas = preparar_sinal(v)
                    if x is None or len(x) < cfg.N_MINIMO_DWT:
                        continue
                    f = atributos_sinal(x / peso[sess], comp)
                    f[f"{comp}_lacunas"] = n_lacunas
                    f.update({"SESSION_ID": sess, "TRIAL_ID": int(chave[-1])})
                    linhas.append(f)
            por_comp.append(pd.DataFrame(linhas).set_index(["SESSION_ID", "TRIAL_ID"]))
        # mantém só trials com todas as componentes
        tab = pd.concat(por_comp, axis=1, join="inner").reset_index()
        tab["lado"] = lado
        partes.append(tab)

    trials = pd.concat(partes, ignore_index=True)
    trials.insert(0, "participante", trials["SESSION_ID"].map(participante))
    trials.insert(0, "base", base)
    trials["dur_s"] = trials["V_n"] / cfg.FS
    return trials


def agregar_por_membro(trials: pd.DataFrame) -> pd.DataFrame:
    """Média dos trials por participante e membro (como no artigo)."""
    numericas = [
        c for c in trials.columns
        if c not in ("base", "participante", "SESSION_ID", "TRIAL_ID", "lado") and not c.endswith("_lacunas")
    ]
    agg = trials.groupby(["base", "participante", "SESSION_ID", "lado"])[numericas].mean()
    agg["n_trials"] = trials.groupby(["base", "participante", "SESSION_ID", "lado"]).size()
    return agg.reset_index()
