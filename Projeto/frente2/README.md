# Frente 2 — wavelets e aprendizado de máquina

Código das três primeiras etapas do roteiro: elegibilidade e integridade, pipeline
DWT e características, calibração (controle negativo) e primeira passada em K_R.

## Como rodar

Na pasta `Projeto`, com o mesmo ambiente do notebook (precisa de `numpy`, `pandas`,
`scikit-learn`, `matplotlib`, `marimo`; o `PyWavelets` é usado se estiver instalado):

```
pip install -r frente2/requirements.txt
python -m frente2.dwt                      # confere a DWT NumPy contra o PyWavelets
python -m frente2.executar --sensibilidade # etapas 1, 2, 3 e a última sessão
marimo edit Frente2.py                     # mostra os resultados
```

Os dados são lidos de `marimo-env/GaitRec Database` e `marimo-env/Gutenberg Gait Database`.
Para usar outra pasta, defina `FRENTE2_DADOS`; para gravar em outro lugar, `FRENTE2_RESULTADOS`.

## Arquivos

| Arquivo | O que faz |
| --- | --- |
| `config.py` | Caminhos relativos, parâmetros da DWT, regras de elegibilidade, semente |
| `dwt.py` | `wavedec` db4 nível 4 simétrico: usa `pywt` ou a versão NumPy equivalente |
| `elegibilidade.py` | Etapa 1: regras únicas de seleção, fluxo, checagens de integridade |
| `caracteristicas.py` | Etapa 2: leitura em blocos, preparo do sinal, DWT, atributos por trial e por membro |
| `modelos.py` | Etapa 3: diferenças espelhadas, validação por participante, bootstrap, permutação |
| `executar.py` | Roda as etapas e grava tudo em `resultados/` |

## Decisões fixadas no código

- Mesmo preparo do v6.1: sem filtro, NaN das pontas removido, lacunas internas interpoladas,
  db4, nível 4, 250 Hz, borda simétrica, trial com menos de 112 amostras descartado.
- A DWT em NumPy reproduz os MSWC dos 104 controles do v6.1 (diferença relativa ≤ 2×10⁻⁶).
- Sinais divididos pelo peso corporal da sessão (BODY_WEIGHT, N). Na diferença entre membros
  em log, essa escala se cancela.
- Subbandas com os nomes do artigo: ylp4, yhp4, yhp3, yhp2, yhp1.
- Sinalizações de integridade (sexo, classe, lado, idade × data, altura) não excluem ninguém.
