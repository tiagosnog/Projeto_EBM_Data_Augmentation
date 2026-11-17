"""H2: AUC por conjunto e diferença pareada (B2−B1, B3−B1, B3−B2) por bootstrap de participantes, em K_R, K_F e K_P.
Uso, a partir da pasta Projeto: python frente2/analises/h2_pareado.py
Precisa de etapa1_elegiveis.csv e etapa2_caracteristicas_membro.csv em frente2/resultados."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))  # pasta Projeto
import time
import numpy as np, pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from frente2 import modelos as m, config as cfg
R = str(cfg.RESULTADOS) + "/"
mem = pd.read_csv(R + "etapa2_caracteristicas_membro.csv"); el = pd.read_csv(R + "etapa1_elegiveis.csv")
def lr(C): return make_pipeline(StandardScaler(), LogisticRegression(C=C, max_iter=5000))
MODELOS = {"logística C=1": (lr(1.0), 10), "logística C=0,1": (lr(0.1), 10), "logística C=0,01": (lr(0.01), 10),
           "floresta": (RandomForestClassifier(n_estimators=300, min_samples_leaf=5, random_state=cfg.SEMENTE, n_jobs=-1), 5)}
SETS = ["B1 clássico", "B2 wavelet", "B3 clássico + wavelet"]
N_BOOT = 2000
lin, dif = [], []
t0 = time.time()
for a in m._analises(mem, el):
    if a["tipo"] == "controle negativo": continue
    g = a["grupo"]; conj = m.conjuntos_atributos(a["componentes"])
    Ds = {s: m.diferencas(mem, a["lado_a"], conj[s]) for s in SETS}
    idx = Ds[SETS[0]].index
    for s in SETS: idx = idx.intersection(Ds[s].index)
    est = a["estrato"].reindex(idx).astype(str).to_numpy()
    rng = np.random.default_rng(cfg.SEMENTE)
    boots = [rng.integers(0, len(idx), len(idx)) for _ in range(N_BOOT)]
    for nome_m, (mod, reps) in MODELOS.items():
        pr = {}
        for s in SETS:
            pp, pn = m._validacao(Ds[s].loc[idx].to_numpy(), est, mod, reps, cfg.SEMENTE, False)
            pr[s] = (pp.mean(0), pn.mean(0))
        auc_obs = {s: m._auc(*pr[s]) for s in SETS}
        bs = {s: np.array([m._auc(pr[s][0][i], pr[s][1][i]) for i in boots]) for s in SETS}
        for s in SETS:
            lin.append(dict(grupo=g, modelo=nome_m, atributos=s, n=len(idx), auc=auc_obs[s],
                            ic_inf=np.percentile(bs[s], 2.5), ic_sup=np.percentile(bs[s], 97.5)))
        for x, y in (("B2 wavelet", "B1 clássico"), ("B3 clássico + wavelet", "B1 clássico"), ("B3 clássico + wavelet", "B2 wavelet")):
            d = bs[x] - bs[y]
            dif.append(dict(grupo=g, modelo=nome_m, comparacao=f"{x[:2]} − {y[:2]}", n=len(idx),
                            dif_auc=auc_obs[x] - auc_obs[y], ic_inf=np.percentile(d, 2.5), ic_sup=np.percentile(d, 97.5),
                            p_bootstrap=min(1.0, 2 * min((d <= 0).mean(), (d >= 0).mean()))))
        print(g, nome_m, "ok %.0f s" % (time.time() - t0), flush=True)
pd.DataFrame(lin).round(4).to_csv(str(cfg.RESULTADOS) + "/h2_auc_por_conjunto.csv", index=False)
pd.DataFrame(dif).round(4).to_csv(str(cfg.RESULTADOS) + "/h2_diferencas_pareadas.csv", index=False)
print("fim")
