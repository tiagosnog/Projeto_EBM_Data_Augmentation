"""Executa as etapas da Frente 2 e grava tudo em ``frente2/resultados``.

Uso, a partir da pasta ``Projeto``:

    python -m frente2.executar            # etapas 1, 2 e 3
    python -m frente2.executar --etapas 1 2
    python -m frente2.executar --etapas 3   # reaproveita as características já gravadas
"""

from __future__ import annotations

import argparse
import json
import time
from datetime import datetime

import pandas as pd

from . import config as cfg
from . import dwt


def etapa1() -> pd.DataFrame:
    from . import elegibilidade

    r = elegibilidade.executar()
    cfg.RESULTADOS.mkdir(parents=True, exist_ok=True)
    r["elegiveis"].to_csv(cfg.RESULTADOS / "etapa1_elegiveis.csv", index=False)
    r["fluxo"].to_csv(cfg.RESULTADOS / "etapa1_fluxo.csv", index=False)
    r["resumo"].to_csv(cfg.RESULTADOS / "etapa1_resumo.csv", index=False)
    r["flags"][r["flags"]["alguma_sinalizacao"]].to_csv(
        cfg.RESULTADOS / "etapa1_sinalizacoes.csv", index=False
    )
    r["info"].to_csv(cfg.RESULTADOS / "etapa1_info.csv", index=False)
    return r["elegiveis"]


def etapa2(elegiveis: pd.DataFrame | None = None) -> pd.DataFrame:
    from . import caracteristicas

    if elegiveis is None:
        elegiveis = pd.read_csv(cfg.RESULTADOS / "etapa1_elegiveis.csv")
    trials, membros = [], []
    for base in ("Gutenberg", "GaitRec"):
        t0 = time.time()
        t = caracteristicas.extrair_trials(base, elegiveis)
        trials.append(t)
        membros.append(caracteristicas.agregar_por_membro(t))
        print(f"  {base}: {len(t)} linhas de trial em {time.time() - t0:.0f} s")
    trials = pd.concat(trials, ignore_index=True)
    membros = pd.concat(membros, ignore_index=True)
    trials.to_csv(cfg.RESULTADOS / "etapa2_trials.csv.gz", index=False, compression="gzip")
    membros.to_csv(cfg.RESULTADOS / "etapa2_caracteristicas_membro.csv", index=False)
    return membros


def etapa3(membros: pd.DataFrame | None = None, elegiveis: pd.DataFrame | None = None) -> None:
    from . import modelos

    if membros is None:
        membros = pd.read_csv(cfg.RESULTADOS / "etapa2_caracteristicas_membro.csv")
    if elegiveis is None:
        elegiveis = pd.read_csv(cfg.RESULTADOS / "etapa1_elegiveis.csv")
    r = modelos.executar(membros, elegiveis)
    for nome, tab in r.items():
        tab.to_csv(cfg.RESULTADOS / f"etapa3_{nome}.csv", index=False)


def sensibilidade_ultima_sessao() -> None:
    """Repete a primeira passada com a última sessão de acompanhamento dos pacientes."""
    from . import caracteristicas, elegibilidade, modelos

    gr, _ = elegibilidade.carregar_metadados()
    inicial = pd.read_csv(cfg.RESULTADOS / "etapa1_elegiveis.csv", parse_dates=["DATA"])
    ultima = elegibilidade.selecionar_ultima_sessao(gr, inicial)
    ultima.to_csv(cfg.RESULTADOS / "sens_ultima_elegiveis.csv", index=False)
    t = caracteristicas.extrair_trials("GaitRec", ultima)
    membros = caracteristicas.agregar_por_membro(t)
    membros.to_csv(cfg.RESULTADOS / "sens_ultima_caracteristicas_membro.csv", index=False)
    # assimetria basal estimada nos saudáveis da GaitRec da coorte principal
    inicial_m = pd.read_csv(cfg.RESULTADOS / "etapa2_caracteristicas_membro.csv")
    hc = inicial[(inicial["base"] == "GaitRec") & (inicial["grupo"] == "HC")]["participante"]
    r = modelos.executar(membros, ultima, prefixo="[Última sessão] ",
                         referencia_hc=inicial_m[inicial_m["participante"].isin(hc)])
    for nome, tab in r.items():
        tab.to_csv(cfg.RESULTADOS / f"sens_ultima_{nome}.csv", index=False)


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--etapas", nargs="+", type=int, default=[1, 2, 3])
    p.add_argument("--sensibilidade", action="store_true", help="também roda a última sessão dos pacientes")
    args = p.parse_args()

    cfg.RESULTADOS.mkdir(parents=True, exist_ok=True)
    print("DWT:", dwt.implementacao())
    el = membros = None
    if 1 in args.etapas:
        print("Etapa 1: elegibilidade e integridade")
        el = etapa1()
    if 2 in args.etapas:
        print("Etapa 2: DWT e características")
        membros = etapa2(el)
    if 3 in args.etapas:
        print("Etapa 3: calibração e primeira passada")
        etapa3(membros, el)
    if args.sensibilidade:
        print("Sensibilidade: última sessão dos pacientes")
        sensibilidade_ultima_sessao()

    info = {
        "executado_em": datetime.now().isoformat(timespec="seconds"),
        "etapas": args.etapas,
        "dwt": dwt.implementacao(),
        "componentes_gaitrec": cfg.componentes_disponiveis("GaitRec"),
        "componentes_gutenberg": cfg.componentes_disponiveis("Gutenberg"),
        "semente": cfg.SEMENTE,
    }
    (cfg.RESULTADOS / "execucao.json").write_text(json.dumps(info, indent=2, ensure_ascii=False))
    print("Resultados em", cfg.RESULTADOS)


if __name__ == "__main__":
    main()
