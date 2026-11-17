"""Efeito de lado nos saudáveis por família de atributos e transferência entre bases.
Uso, a partir da pasta Projeto: python frente2/analises/efeito_de_lado.py"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))  # pasta Projeto
import numpy as np, pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from frente2 import modelos as m, config as cfg
R = str(cfg.RESULTADOS) + "/"
mem = pd.read_csv(R + "etapa2_caracteristicas_membro.csv"); el = pd.read_csv(R + "etapa1_elegiveis.csv").drop_duplicates("participante").set_index("participante")
allc = m.conjuntos_atributos(["V","AP","ML"])["B3 clássico + wavelet"]
mod = make_pipeline(StandardScaler(), LogisticRegression(C=1.0, max_iter=5000))
H = {b: [p for p in el[(el.base==b)&(el.grupo=="HC")].index if p in set(mem.participante)] for b in ("GaitRec","Gutenberg")}
def auc_cv(b, cols, reps=5):
    la = pd.Series("left", index=H[b]); D = m.diferencas(mem, la, cols)
    est = el.loc[D.index, "sexo"].astype(str).to_numpy()
    pp, pn = m._validacao(D.to_numpy(), est, mod, reps, cfg.SEMENTE, False)
    return m._auc(pp.mean(0), pn.mean(0))
def transfer(b_tr, b_te, cols):
    Dtr = m.diferencas(mem, pd.Series("left", index=H[b_tr]), cols).to_numpy(); Dte = m.diferencas(mem, pd.Series("left", index=H[b_te]), cols).to_numpy()
    f = mod.fit(np.vstack([Dtr, -Dtr]), np.r_[np.ones(len(Dtr)), np.zeros(len(Dtr))])
    return m._auc(f.predict_proba(Dte)[:,1], f.predict_proba(-Dte)[:,1])
fam = {"B3 (todos, 57)": allc,
       "só V": [c for c in allc if c.startswith("V_")], "só AP": [c for c in allc if c.startswith("AP_")], "só ML": [c for c in allc if c.startswith("ML_")],
       "só B1 clássico (12)": m.conjuntos_atributos(["V","AP","ML"])["B1 clássico"], "só B2 wavelet (45)": m.conjuntos_atributos(["V","AP","ML"])["B2 wavelet"]}
for sb in cfg.SUBBANDAS: fam[f"subbanda {sb} (9)"] = [c for c in allc if f"_{sb}_" in c]
rows = []
for nome, cols in fam.items():
    rows.append(dict(familia=nome, n_atributos=len(cols), auc_GaitRec=auc_cv("GaitRec", cols), auc_Gutenberg=auc_cv("Gutenberg", cols),
                     transf_GR_para_GU=transfer("GaitRec","Gutenberg",cols), transf_GU_para_GR=transfer("Gutenberg","GaitRec",cols)))
df = pd.DataFrame(rows).round(3); df.to_csv(str(cfg.RESULTADOS) + "/efeito_de_lado_por_familia.csv", index=False)
print(df.to_string(index=False))
