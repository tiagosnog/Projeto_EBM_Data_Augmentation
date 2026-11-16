"""Configuração da Frente 2 (aprendizado de máquina com wavelets).

Todos os caminhos são relativos à pasta ``Projeto``; nada depende do disco C:.
Para usar outra pasta de dados ou de resultados, defina as variáveis de
ambiente ``FRENTE2_DADOS`` e ``FRENTE2_RESULTADOS``.
"""

from __future__ import annotations

import os
from pathlib import Path

# Pasta "Projeto" (um nível acima desta pasta "frente2")
RAIZ_PROJETO = Path(__file__).resolve().parents[1]

DADOS = Path(os.environ.get("FRENTE2_DADOS", RAIZ_PROJETO / "marimo-env"))
GAITREC = DADOS / "GaitRec Database"
GUTENBERG = DADOS / "Gutenberg Gait Database"

RESULTADOS = Path(
    os.environ.get("FRENTE2_RESULTADOS", RAIZ_PROJETO / "frente2" / "resultados")
)

# Sinal e wavelet: as mesmas escolhas do artigo (análise v6)
FS = 250  # Hz, as duas bases (reamostradas de 1000 e 2000 Hz pelos autores)
WAVELET = "db4"
NIVEL = 4
MODO_BORDA = "symmetric"
# Nomes do artigo, na ordem de pywt.wavedec:
# ylp4 0-7,81 Hz | yhp4 7,81-15,63 | yhp3 15,63-31,25 | yhp2 31,25-62,5 | yhp1 62,5-125
SUBBANDAS = ["ylp4", "yhp4", "yhp3", "yhp2", "yhp1"]
N_MINIMO_DWT = 7 * 2**4  # pywt.dwt_max_level(n, 8) >= 4, mesmo critério do v6.1

COMPONENTES = ["V", "AP", "ML"]
LADOS = ["left", "right"]

# Grupos analisados (categorias EXATAS de CLASS_LABEL_DETAILED)
SUBCLASSES_K = ["K_P", "K_F", "K_R"]
SUBCLASSE_PRINCIPAL = "K_R"  # pré-especificada no roteiro

# Regras de elegibilidade (iguais para todas as tabelas)
IDADE_MINIMA = 18
SPEED_AUTOSELECIONADA = 2
SESSION_TYPE_INICIAL = 1

# Tolerâncias das checagens de integridade
TOL_IDADE_ANOS = 1.1  # diferença entre variação de idade e tempo decorrido
TOL_ALTURA_CM = 5.0

SEMENTE = 20260819  # mesma semente do notebook do artigo

# AFFECTED_SIDE (documentação v5.2 do notebook): 0 esquerdo, 1 direito, 2 bilateral
LADO_AFETADO = {0: "left", 1: "right"}


def arquivo_sinal(base: str, componente: str, lado: str) -> Path:
    pasta = GAITREC if base == "GaitRec" else GUTENBERG
    return pasta / f"GRF_F_{componente}_RAW_{lado}.csv"


def componentes_disponiveis(base: str) -> list[str]:
    """Componentes com os dois arquivos (esquerdo e direito) presentes."""
    return [
        c
        for c in COMPONENTES
        if all(arquivo_sinal(base, c, lado).exists() for lado in LADOS)
    ]
