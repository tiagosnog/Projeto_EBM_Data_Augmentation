"""Decomposição wavelet discreta (db4, nível 4, borda simétrica).

Usa o PyWavelets quando ele está instalado. Sem ele, usa uma implementação em
NumPy que segue a mesma convenção do ``pywt.wavedec(..., mode="symmetric")``.
Antes de usar resultados calculados sem o PyWavelets, rode
``python -m frente2.dwt`` num ambiente com ``pywt``: a função
``validar_contra_pywt`` compara as duas implementações.
"""

from __future__ import annotations

import numpy as np

try:  # pragma: no cover - depende do ambiente
    import pywt as _pywt
except ImportError:  # pragma: no cover
    _pywt = None

# Filtros de decomposição db4 (mesma ordem do pywt.Wavelet("db4").dec_lo)
DB4_DEC_LO = np.array(
    [
        -0.010597401784997278,
        0.032883011666982945,
        0.030841381835986965,
        -0.18703481171888114,
        -0.02798376941698385,
        0.6308807679295904,
        0.7148465705525415,
        0.23037781330885523,
    ]
)
# Filtro passa-alta por espelhamento em quadratura
DB4_DEC_HI = np.array(
    [(-1) ** (k + 1) * DB4_DEC_LO[len(DB4_DEC_LO) - 1 - k] for k in range(len(DB4_DEC_LO))]
)


def _indices_simetricos(n_sinal: int, inicio: int, fim: int) -> np.ndarray:
    """Índices da extensão simétrica de meia amostra (modo 'symmetric')."""
    idx = np.arange(inicio, fim)
    periodo = 2 * n_sinal
    m = np.mod(idx, periodo)
    return np.where(m >= n_sinal, periodo - 1 - m, m)


def _dwt_numpy(x: np.ndarray, lo: np.ndarray, hi: np.ndarray):
    f = len(lo)
    n = len(x)
    xp = x[_indices_simetricos(n, -(f - 1), n + f - 1)]
    n_saida = (n + f - 1) // 2
    ca = np.convolve(xp, lo)[f::2][:n_saida]
    cd = np.convolve(xp, hi)[f::2][:n_saida]
    return ca, cd


def wavedec_numpy(x: np.ndarray, nivel: int = 4) -> list[np.ndarray]:
    """Equivalente a ``pywt.wavedec(x, "db4", mode="symmetric", level=nivel)``."""
    a = np.asarray(x, dtype=float)
    detalhes = []
    for _ in range(nivel):
        a, d = _dwt_numpy(a, DB4_DEC_LO, DB4_DEC_HI)
        detalhes.append(d)
    return [a] + detalhes[::-1]


def wavedec(x: np.ndarray, wavelet: str = "db4", nivel: int = 4, modo: str = "symmetric"):
    """Retorna [cA_n, cD_n, ..., cD1], como o ``pywt.wavedec``."""
    if _pywt is not None:
        return _pywt.wavedec(np.asarray(x, dtype=float), wavelet, mode=modo, level=nivel)
    if wavelet != "db4" or modo != "symmetric":
        raise RuntimeError("Sem PyWavelets, só db4 com borda simétrica está implementada.")
    return wavedec_numpy(x, nivel)


def implementacao() -> str:
    return f"pywt {_pywt.__version__}" if _pywt is not None else "numpy (sem PyWavelets)"


def checar_filtros() -> dict:
    """Propriedades que um filtro db4 ortonormal precisa satisfazer."""
    lo, hi = DB4_DEC_LO, DB4_DEC_HI
    k = np.arange(len(hi))
    return {
        "soma_lo_igual_raiz2": bool(np.isclose(lo.sum(), np.sqrt(2))),
        "norma_unitaria": bool(np.isclose((lo**2).sum(), 1.0)),
        "ortogonal_deslocamentos_pares": bool(
            all(np.isclose((lo[: -2 * s] * lo[2 * s :]).sum(), 0.0, atol=1e-12) for s in (1, 2, 3))
        ),
        "momentos_nulos_0_a_3": bool(
            all(np.isclose((k**m * hi).sum(), 0.0, atol=1e-8) for m in range(4))
        ),
    }


def validar_contra_pywt(n_sinais: int = 200, semente: int = 0) -> float:
    """Maior diferença absoluta entre NumPy e PyWavelets em sinais aleatórios."""
    if _pywt is None:
        raise RuntimeError("PyWavelets não está instalado neste ambiente.")
    rng = np.random.default_rng(semente)
    pior = 0.0
    for _ in range(n_sinais):
        x = rng.normal(size=int(rng.integers(100, 420)))
        a = _pywt.wavedec(x, "db4", mode="symmetric", level=4)
        b = wavedec_numpy(x, 4)
        for ca, cb in zip(a, b):
            if ca.shape != cb.shape:
                raise AssertionError(f"Tamanhos diferentes: {ca.shape} x {cb.shape}")
            pior = max(pior, float(np.max(np.abs(ca - cb))))
    return pior


if __name__ == "__main__":  # python -m frente2.dwt
    print("Filtros:", checar_filtros())
    print("Implementação ativa:", implementacao())
    if _pywt is not None:
        dif = validar_contra_pywt()
        print(f"Maior diferença NumPy x PyWavelets: {dif:.2e}")
        print("OK" if dif < 1e-9 else "ATENÇÃO: implementações divergem")
