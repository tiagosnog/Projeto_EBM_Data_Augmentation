"""Ranking de dz do efeito de lado nos saudáveis (A = esquerdo) e variabilidade entre trials em K_R.
Uso, a partir da pasta Projeto: python frente2/analises/dz_e_variabilidade.py
Precisa de etapa1_elegiveis.csv, etapa2_trials.csv.gz e etapa3_assimetria_descritiva.csv em frente2/resultados.
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))  # pasta Projeto
import numpy as np
import pandas as pd

from frente2 import config as cfg

R = cfg.RESULTADOS

# 1) dz por atributo nos saudáveis das duas bases
d = pd.read_csv(R / "etapa3_assimetria_descritiva.csv")
c = d[d.analise.str.startswith("Controle") & d.analise.str.contains(r"\(V\+AP\+ML\)")]
w = c.assign(base=np.where(c.analise.str.contains("GaitRec"), "GR", "GU")).pivot(index="atributo", columns="base", values="dz")
w["mesmo_sinal"] = np.sign(w.GR) == np.sign(w.GU)
w["max_abs"] = w[["GR", "GU"]].abs().max(axis=1)
w = w.sort_values("max_abs", ascending=False)
w.round(3).to_csv(R / "efeito_de_lado_dz_por_atributo.csv")
print("mesmo sinal nas duas bases: %d de %d; correlação dos dz: %.2f" % (w.mesmo_sinal.sum(), len(w), w.GR.corr(w.GU)))

# 2) variabilidade entre trials do mesmo membro, comparada ao desvio de Δ entre participantes (K_R)
t = pd.read_csv(R / "etapa2_trials.csv.gz")
el = pd.read_csv(R / "etapa1_elegiveis.csv").drop_duplicates("participante").set_index("participante")
t["dur_s"] = t["V_n"] / cfg.FS
kr = el[(el.base == "GaitRec") & (el.grupo == "K_R")]
k = t[t.participante.isin(kr.index)].copy()
n = k.groupby(["participante", "lado"]).size()
print("trials por membro em K_R: mediana %d, mínimo %d, máximo %d" % (n.median(), n.min(), n.max()))
feats = ["V_pico1", "V_vale", "V_pico2", "AP_pico_freio", "AP_pico_prop", "ML_min", "dur_s",
         "V_ylp4_MSWC", "V_yhp1_MSWC", "AP_yhp3_MSWC"]
rows = []
for f in feats:
    em_log = f.endswith("MSWC")
    k["_v"] = np.log(k[f].clip(lower=1e-12)) if em_log else k[f]
    dp_trial = k.groupby(["participante", "lado"])["_v"].std().median()
    m = k.groupby(["participante", "lado"])["_v"].mean().unstack()
    afet = m.index.map(kr.lado_afetado)
    delta = np.where(afet == "left", m["left"] - m["right"], m["right"] - m["left"])
    dp_delta = np.std(delta, ddof=1)
    rows.append(dict(atributo=f + (" (log)" if em_log else ""), sd_entre_trials_mesmo_membro=dp_trial,
                     media_delta=np.mean(delta), dp_delta_participantes=dp_delta,
                     razao_ruido_trial_sobre_dp_delta=dp_trial / dp_delta))
pd.DataFrame(rows).round(3).to_csv(R / "variabilidade_trials_K_R.csv", index=False)
