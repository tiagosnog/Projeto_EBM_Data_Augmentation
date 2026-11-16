# Projeto de pesquisa EBM 01 — forças de reação do solo, wavelets e assimetria do joelho

Código da pesquisa de mestrado de Tiago da Silva Nogueira (Engenharia Biomédica, UFABC),
orientada por Renato Naville Watanabe. O trabalho analisa a força de reação do solo (GRF)
na marcha com a transformada wavelet discreta (DWT), comparando o membro afetado e o
contralateral em pacientes com lesão unilateral de joelho.

**Estado:** em andamento. Os resultados em `Projeto/frente2/resultados` são preliminares
(rodada de 02/11/2026) e ainda não foram revisados por pares.

## Conteúdo

| Caminho | O que é |
| --- | --- |
| `Projeto/Analise_medio_lateral_v6_1.ipynb` | Análise do artigo: DWT da componente mediolateral, joelho × saudáveis |
| `Projeto/Projeto.py` | Notebook Marimo: auditoria dos sinais, coorte elegível e integridade dos metadados |
| `Projeto/Frente2.py` | Notebook Marimo: exibe os resultados da Frente 2 |
| `Projeto/frente2/` | Pipeline: elegibilidade, DWT e características, modelos e validação |
| `Projeto/frente2/resultados/` | Tabelas agregadas e `RESUMO_RESULTADOS.md` |

## Dados

As bases não fazem parte do repositório. Baixe da figshare e coloque os arquivos
`GRF_metadata.csv` e `GRF_F_{V,AP,ML}_RAW_{left,right}.csv` nas pastas indicadas
(a Gutenberg também precisa do `GRF_walking_speed.csv`):

| Base | DOI | Pasta |
| --- | --- | --- |
| GaitRec (Horst et al., 2020) | [10.6084/m9.figshare.c.4788012.v1](https://doi.org/10.6084/m9.figshare.c.4788012.v1) | `Projeto/marimo-env/GaitRec Database/` |
| Gutenberg Gait Database (Horst et al., 2021) | [10.6084/m9.figshare.c.5311538.v1](https://doi.org/10.6084/m9.figshare.c.5311538.v1) | `Projeto/marimo-env/Gutenberg Gait Database/` |

Para usar outra pasta, defina a variável de ambiente `FRENTE2_DADOS`.

## Como rodar

Python 3.13, a partir da pasta `Projeto`:

```
python -m pip install -r frente2/requirements.txt
python -m frente2.dwt                       # confere a DWT em NumPy contra o PyWavelets
python -m frente2.executar --sensibilidade  # etapas 1, 2, 3 e última sessão (20 a 25 min)
marimo edit Frente2.py                      # mostra os resultados
marimo edit Projeto.py                      # auditoria das bases
```

Detalhes do pipeline em [`Projeto/frente2/README.md`](Projeto/frente2/README.md).

## Método, em resumo

- Coorte: adultos, velocidade autosselecionada, sessão inicial, lesão unilateral, uma sessão por participante.
- Sinal RAW a 250 Hz, sem filtro adicional; DWT db4, nível 4, borda simétrica, trial a trial.
- Atributos por subbanda (energia, MSWC, energia relativa) e atributos clássicos (picos, impulso, duração do apoio).
- Diferença entre membros da mesma pessoa; validação cruzada agrupada por participante;
  controle negativo esquerdo × direito em saudáveis.

## Referências das bases

- Horst, F. et al. (2020). GaitRec, a large-scale ground reaction force dataset of healthy and impaired gait. *Scientific Data*, 7, 143.
- Horst, F. et al. (2021). Gutenberg Gait Database, a ground reaction force database of level overground walking in healthy individuals. *Scientific Data*, 8, 232.
