"""Etapa 3: calibração (controle negativo) e primeira passada em K_R.

Formulação (roteiro, seção de características):

* cada participante gera dois exemplos espelhados: a diferença
  (membro A menos membro B) com rótulo 1 e a diferença oposta com rótulo 0;
* nos pacientes, A = membro afetado; nos saudáveis (controle negativo),
  A = membro esquerdo; o esperado nos saudáveis é AUC próximo de 0,5;
* energias, MSWC e energia relativa entram em log, então a diferença é um
  log da razão entre os membros; idade, massa e velocidade se cancelam;
* validação cruzada por participante (os dois exemplos ficam juntos),
  estratificada pelo lado afetado (pacientes) ou pelo sexo (saudáveis).

Métricas: AUC dos exemplos espelhados e acurácia de identificação
(proporção de participantes em que o modelo aponta o membro A).
Intervalos por bootstrap de participantes; p-valor por permutação
(troca do rótulo dentro de cada par).
"""

from __future__ import annotations

import numpy as np
import pandas as pd
try:
    from sklearn.base import clone
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.linear_model import LogisticRegression
    from sklearn.metrics import roc_auc_score
    from sklearn.model_selection import StratifiedKFold
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler
except ImportError as erro:  # pragma: no cover
    raise ImportError(
        "A etapa 3 precisa do scikit-learn. Instale com: python -m pip install scikit-learn"
    ) from erro

from . import config as cfg

METRICAS_WAVELET = ("E", "MSWC", "Erel")
# Atributos clássicos (B1, versão 2): ver o docstring de caracteristicas.py
CLASSICOS = {
    "V": ("pico1", "vale", "pico2", "impulso"),
    "AP": ("pico_freio", "pico_prop", "imp_freio", "imp_prop"),
    "ML": ("max", "min", "impulso"),
}


def conjuntos_atributos(componentes: list[str]) -> dict[str, list[str]]:
    b1 = [f"{c}_{s}" for c in componentes for s in CLASSICOS[c]] + ["dur_s"]
    b2 = [f"{c}_{b}_{m}" for c in componentes for b in cfg.SUBBANDAS for m in METRICAS_WAVELET]
    return {"B1 clássico": b1, "B2 wavelet": b2, "B3 clássico + wavelet": b1 + b2}


def transformar(df: pd.DataFrame, colunas: list[str]) -> pd.DataFrame:
    """Log nas medidas de energia (a diferença vira log da razão)."""
    out = df[colunas].astype(float).copy()
    for c in colunas:
        if c.endswith(METRICAS_WAVELET):
            out[c] = np.log(out[c].clip(lower=1e-12))
    return out


def diferencas(membros: pd.DataFrame, lado_a: pd.Series, colunas: list[str]) -> pd.DataFrame:
    """Δ = atributo(membro A) − atributo(membro B), uma linha por participante."""
    m = membros.set_index(["participante", "lado"])
    t = transformar(m, colunas)
    linhas = {}
    for p, a in lado_a.items():
        b = "right" if a == "left" else "left"
        if (p, a) in t.index and (p, b) in t.index:
            linhas[p] = t.loc[(p, a)] - t.loc[(p, b)]
    return pd.DataFrame(linhas).T.dropna()


def modelos_disponiveis(semente: int) -> dict:
    return {
        "logística": make_pipeline(StandardScaler(), LogisticRegression(C=1.0, max_iter=5000)),
        "floresta": RandomForestClassifier(
            n_estimators=300, min_samples_leaf=5, random_state=semente, n_jobs=-1
        ),
    }


def _validacao(
    D: np.ndarray, estrato: np.ndarray, modelo, repeticoes: int, semente: int, centralizar: bool = False
):
    """Probabilidades fora da amostra para +Δ e −Δ.

    ``centralizar``: subtrai a média de Δ do treino. Atenção: na formulação
    espelhada, a regressão logística só aprende a direção média de Δ; com Δ
    centralizado ela fica exatamente no acaso. Por isso o controle negativo
    calibrado usa a assimetria basal da outra base, e não esta opção.
    """
    n = len(D)
    p_pos = np.zeros((repeticoes, n))
    p_neg = np.zeros((repeticoes, n))
    for r in range(repeticoes):
        folds = StratifiedKFold(n_splits=5, shuffle=True, random_state=semente + r)
        for tr, te in folds.split(D, estrato):
            mu = D[tr].mean(axis=0) if centralizar else 0.0
            d_tr, d_te = D[tr] - mu, D[te] - mu
            X = np.vstack([d_tr, -d_tr])
            y = np.r_[np.ones(len(tr)), np.zeros(len(tr))]
            m = clone(modelo).fit(X, y)
            p_pos[r, te] = m.predict_proba(d_te)[:, 1]
            p_neg[r, te] = m.predict_proba(-d_te)[:, 1]
    return p_pos, p_neg


def _auc(pp: np.ndarray, pn: np.ndarray) -> float:
    return roc_auc_score(np.r_[np.ones(len(pp)), np.zeros(len(pn))], np.r_[pp, pn])


def _acerto(pp: np.ndarray, pn: np.ndarray) -> float:
    return float(np.mean((pp > pn) + 0.5 * (pp == pn)))


def avaliar(
    D: np.ndarray,
    estrato: np.ndarray,
    modelo,
    repeticoes: int = 10,
    semente: int = cfg.SEMENTE,
    n_boot: int = 2000,
    n_perm: int = 0,
    centralizar: bool = False,
) -> dict:
    p_pos, p_neg = _validacao(D, estrato, modelo, repeticoes, semente, centralizar)
    aucs = [_auc(p_pos[r], p_neg[r]) for r in range(repeticoes)]
    pp, pn = p_pos.mean(axis=0), p_neg.mean(axis=0)

    rng = np.random.default_rng(semente)
    boot_auc, boot_acc = [], []
    for _ in range(n_boot):
        i = rng.integers(0, len(D), len(D))
        boot_auc.append(_auc(pp[i], pn[i]))
        boot_acc.append(_acerto(pp[i], pn[i]))

    res = {
        "n_participantes": len(D),
        "auc": float(np.mean(aucs)),
        "auc_dp_repeticoes": float(np.std(aucs)),
        "auc_ic95_inf": float(np.percentile(boot_auc, 2.5)),
        "auc_ic95_sup": float(np.percentile(boot_auc, 97.5)),
        "acerto": _acerto(pp, pn),
        "acerto_ic95_inf": float(np.percentile(boot_acc, 2.5)),
        "acerto_ic95_sup": float(np.percentile(boot_acc, 97.5)),
        "_pp": pp,
        "_pn": pn,
    }
    if n_perm:
        # mesmo protocolo com 1 repetição, para observado e permutações
        o_pos, o_neg = _validacao(D, estrato, modelo, 1, semente, centralizar)
        auc_obs = _auc(o_pos[0], o_neg[0])
        nulos = []
        for k in range(n_perm):
            sinais = rng.choice([-1.0, 1.0], size=len(D))[:, None]
            q_pos, q_neg = _validacao(D * sinais, estrato, modelo, 1, semente + 1000 + k, centralizar)
            nulos.append(_auc(q_pos[0], q_neg[0]))
        res["p_permutacao"] = float((1 + np.sum(np.array(nulos) >= auc_obs)) / (1 + n_perm))
        res["n_permutacoes"] = n_perm
    return res


def _analises(membros: pd.DataFrame, elegiveis: pd.DataFrame) -> list[dict]:
    el = elegiveis.drop_duplicates("participante").set_index("participante")
    tem = set(membros["participante"])
    lista = []
    comps_gr = [c for c in cfg.COMPONENTES if c in cfg.componentes_disponiveis("GaitRec")]
    controles = [("GaitRec", comps_gr), ("Gutenberg", comps_gr)]
    if comps_gr != cfg.COMPONENTES:
        controles.append(("Gutenberg", cfg.COMPONENTES))
    # controle negativo: saudáveis, A = esquerdo
    for base, comps in controles:
        sel = el[(el["base"] == base) & (el["grupo"] == "HC") & el.index.isin(tem)]
        lista.append({
            "analise": f"Controle negativo, {base} HC ({'+'.join(comps)})", "tipo": "controle negativo",
            "grupo": "HC", "base": base, "componentes": comps,
            "lado_a": pd.Series("left", index=sel.index), "estrato": sel["sexo"],
        })
    # primeira passada: pacientes, A = membro afetado
    for grupo in [cfg.SUBCLASSE_PRINCIPAL] + [g for g in cfg.SUBCLASSES_K if g != cfg.SUBCLASSE_PRINCIPAL]:
        sel = el[(el["base"] == "GaitRec") & (el["grupo"] == grupo) & el.index.isin(tem)]
        tipo = "principal" if grupo == cfg.SUBCLASSE_PRINCIPAL else "exploratória"
        lista.append({
            "analise": f"Primeira passada, {grupo} ({'+'.join(comps_gr)})", "tipo": tipo,
            "grupo": grupo, "base": "GaitRec", "componentes": comps_gr,
            "lado_a": sel["lado_afetado"], "estrato": sel["lado_afetado"],
        })
    return lista


def assimetria_basal(referencia: pd.DataFrame, colunas: list[str]) -> pd.Series:
    """Média de Δ = esquerdo − direito nos saudáveis de referência (GaitRec HC)."""
    lado_a = pd.Series("left", index=referencia["participante"].unique())
    return diferencas(referencia, lado_a, colunas).mean()


def calibrar(Df: pd.DataFrame, lado_a: pd.Series, basal: pd.Series) -> pd.DataFrame:
    """Remove a assimetria esquerdo − direito típica dos saudáveis.

    Afetado esquerdo: Δ = (E − D) − b; afetado direito: Δ = (D − E) + b.
    """
    sinal = np.where(lado_a.reindex(Df.index).to_numpy() == "left", 1.0, -1.0)[:, None]
    return Df - sinal * basal[Df.columns].to_numpy()[None, :]


def executar(
    membros: pd.DataFrame,
    elegiveis: pd.DataFrame,
    n_perm: int = 200,
    prefixo: str = "",
    minimo: int = 20,
    referencia_hc: pd.DataFrame | None = None,
) -> dict[str, pd.DataFrame]:
    """Roda controle negativo e primeira passada, sem e com calibração.

    ``referencia_hc``: características por membro dos saudáveis da GaitRec,
    usadas para estimar a assimetria basal dos pacientes. Se ausente, sai de
    ``membros`` e ``elegiveis``.
    """
    def _hc(base):
        ids = elegiveis[(elegiveis["base"] == base) & (elegiveis["grupo"] == "HC")]["participante"]
        return membros[membros["participante"].isin(ids)]

    if referencia_hc is None:
        referencia_hc = _hc("GaitRec")
    # Para o controle negativo calibrado, a assimetria basal vem da OUTRA base:
    # se ela for comum às duas bases, a AUC calibrada cai para perto de 0,5.
    referencia_cruzada = {"GaitRec": _hc("Gutenberg"), "Gutenberg": referencia_hc}

    resultados, previsoes, descritiva, por_lado = [], [], [], []
    for a in _analises(membros, elegiveis):
        if len(a["lado_a"]) < minimo:
            continue
        a["analise"] = prefixo + a["analise"]
        conjuntos = conjuntos_atributos(a["componentes"])
        for nome_conj, colunas in conjuntos.items():
            bruto = diferencas(membros, a["lado_a"], colunas)
            estrato = a["estrato"].reindex(bruto.index).astype(str).to_numpy()
            for calibracao in ("bruto", "calibrado"):
                centralizar = False
                if calibracao == "bruto":
                    Df = bruto
                else:
                    ref = referencia_cruzada[a["base"]] if a["grupo"] == "HC" else referencia_hc
                    if ref.empty:
                        continue
                    Df = calibrar(bruto, a["lado_a"], assimetria_basal(ref, colunas))
                rotulo = calibracao if calibracao == "bruto" else (
                    "calibrado pela outra base" if a["grupo"] == "HC" else "calibrado pelos HC da GaitRec"
                )
                for nome_mod, modelo in modelos_disponiveis(cfg.SEMENTE).items():
                    b3 = nome_conj.startswith("B3")
                    if nome_mod == "floresta" and not b3:
                        continue  # floresta só no conjunto completo (custo)
                    logistica_b3 = nome_mod == "logística" and b3
                    r = avaliar(
                        Df.to_numpy(), estrato, modelo,
                        repeticoes=10 if nome_mod == "logística" else 5,
                        n_perm=n_perm if logistica_b3 else 0,
                        centralizar=centralizar,
                    )
                    pp, pn = r.pop("_pp"), r.pop("_pn")
                    resultados.append({"analise": a["analise"], "tipo": a["tipo"], "calibracao": rotulo,
                                       "atributos": nome_conj, "n_atributos": len(colunas),
                                       "modelo": nome_mod, **r})
                    if logistica_b3:
                        previsoes.append(pd.DataFrame({
                            "analise": a["analise"], "calibracao": rotulo, "participante": Df.index,
                            "lado_a": a["lado_a"].reindex(Df.index).to_numpy(),
                            "prob_membro_a": pp, "prob_membro_b": pn,
                        }))
                        if a["grupo"] != "HC":
                            for lado in ("left", "right"):
                                i = estrato == lado
                                por_lado.append({"analise": a["analise"], "calibracao": rotulo,
                                                 "lado_afetado": lado, "n": int(i.sum()),
                                                 "acerto": _acerto(pp[i], pn[i])})
        # assimetria descritiva (B3, sem calibração), com dz = média/DP de Δ
        Df = diferencas(membros, a["lado_a"], conjuntos["B3 clássico + wavelet"])
        d = pd.DataFrame({"analise": a["analise"], "atributo": Df.columns,
                          "media_delta": Df.mean().to_numpy(), "dp_delta": Df.std().to_numpy()})
        d["dz"] = d["media_delta"] / d["dp_delta"]
        descritiva.append(d)

    res = pd.DataFrame(resultados)
    num = res.select_dtypes("number").columns.drop(["n_participantes", "n_atributos", "n_permutacoes"], errors="ignore")
    res[num] = res[num].round(4)
    return {
        "resultados": res,
        "acerto_por_lado": pd.DataFrame(por_lado).round(4),
        "assimetria_descritiva": pd.concat(descritiva, ignore_index=True).round(4),
        "previsoes": pd.concat(previsoes, ignore_index=True).round(4),
    }
