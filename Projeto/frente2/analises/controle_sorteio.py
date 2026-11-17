"""Controle negativo com o membro A sorteado (73/139 esquerdo, como no K_R), 100 sorteios, em saudáveis de cada base.
Uso, a partir da pasta Projeto: python frente2/analises/controle_sorteio.py"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))  # pasta Projeto
import numpy as np, pandas as pd
from frente2 import modelos as m, config as cfg
R = str(cfg.RESULTADOS) + "/"
mem = pd.read_csv(R + "etapa2_caracteristicas_membro.csv")
el = pd.read_csv(R + "etapa1_elegiveis.csv").drop_duplicates("participante").set_index("participante")
p_left = 73/139
modelo = m.modelos_disponiveis(cfg.SEMENTE)["logística"]
comps = ["V","AP","ML"]; conj = m.conjuntos_atributos(comps)
out = []
for base in ("GaitRec","Gutenberg"):
    ids = el[(el.base==base)&(el.grupo=="HC")].index
    ids = [i for i in ids if i in set(mem.participante)]
    for nome, cols in conj.items():
        # lado A fixo (esquerdo), como no controle negativo original
        fix = pd.Series("left", index=ids)
        D = m.diferencas(mem, fix, cols)
        est = el.loc[D.index, "sexo"].astype(str).to_numpy()
        pp, pn = m._validacao(D.to_numpy(), est, modelo, 1, cfg.SEMENTE, False)
        out.append((base, nome, "A = esquerdo (fixo)", m._auc(pp[0], pn[0]), np.nan, np.nan, np.nan))
        aucs = []
        for k in range(100):
            rng = np.random.default_rng(1000+k)
            la = pd.Series(np.where(rng.random(len(ids)) < p_left, "left", "right"), index=ids)
            D = m.diferencas(mem, la, cols)
            est = la.reindex(D.index).to_numpy()
            pp, pn = m._validacao(D.to_numpy(), est, modelo, 1, cfg.SEMENTE+k, False)
            aucs.append(m._auc(pp[0], pn[0]))
        a = np.array(aucs)
        out.append((base, nome, "A aleatório (73/139 esquerdo)", a.mean(), np.percentile(a,2.5), np.percentile(a,97.5), a.max()))
        print(base, nome, "ok", flush=True)
df = pd.DataFrame(out, columns=["base","atributos","atribuicao_de_A","auc","p2.5","p97.5","max"]).round(3)
df.to_csv(str(cfg.RESULTADOS) + "/controle_negativo_A_aleatorio.csv", index=False)
print(df.to_string(index=False))
