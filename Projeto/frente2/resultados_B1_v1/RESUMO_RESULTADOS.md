# Frente 2 — resumo da primeira rodada (02/11/2026)

DWT calculada com a implementação NumPy, que reproduz os MSWC do notebook v6.1
(104 controles homens, ML, cinco subbandas, dois lados; diferença relativa ≤ 2×10⁻⁶).
Componentes: V, AP e ML nas duas bases.

## Etapa 1 — Elegibilidade e integridade

| Base | Grupo | n | Mulheres | Homens | Afetado esq. / dir. |
| --- | --- | --- | --- | --- | --- |
| GaitRec | HC | 193 | 99 | 94 | — |
| GaitRec | K_R | 139 | 43 | 96 | 73 / 66 |
| GaitRec | K_F | 232 | 74 | 158 | 130 / 102 |
| GaitRec | K_P | 60 | 20 | 40 | 34 / 26 |
| Gutenberg | HC | 290 | 119 | 171 | — |

- Regras únicas: categoria exata, idade ≥ 18, sexo conhecido, SPEED = 2, sessão inicial
  (SESSION_TYPE = 1), pacientes unilaterais, uma sessão por participante.
- K_R: 153 → 150 (adultos) → 141 (SPEED 2 e sessão inicial) → 139 (unilaterais).
- GaitRec, 3 identificadores sinalizados: 1137 (está na coorte, em K_P), 2115 e 261 (fora da coorte).
- Gutenberg, 26 participantes com SEX trocado no lote de 2016 do DATASET_ID 3 (24 na coorte).
  A altura deles (média 181,6 cm) é igual à dos homens do mesmo conjunto (181,3 cm) e bem
  acima da das mulheres (170,0 cm): o sexo da primeira sessão (masculino) é o correto.

## Etapa 2 — DWT e características

- Gutenberg: 3.748 trials (7.496 membros); GaitRec: 5.371 trials (10.742 membros). Nenhuma lacuna interna.
- Observação: na GaitRec, o pico vertical nunca fica abaixo de 1,0 peso corporal (mínimo 0,9999),
  o que sugere um critério de inclusão da própria base.

## Etapa 3 — Calibração e primeira passada (logística, IC 95% por bootstrap)

| Análise | Atributos | AUC [IC 95%] | Acerto |
| --- | --- | --- | --- |
| Controle negativo, GaitRec HC (esq. × dir.) | B3 | 0,767 [0,709; 0,838], p = 0,005 | 0,70 |
| Controle negativo, Gutenberg HC (esq. × dir.) | B3 | 0,831 [0,791; 0,878], p = 0,005 | 0,76 |
| Idem, calibrado pela outra base | B3 | 0,924 (GaitRec) e 0,933 (Gutenberg) | — |
| K_R, sessão inicial | B1 clássico | 0,899 [0,847; 0,952] | 0,83 |
| K_R, sessão inicial | B2 wavelet | 0,884 [0,837; 0,947] | 0,81 |
| K_R, sessão inicial | B3 clássico + wavelet | 0,869 [0,820; 0,936], p = 0,005 | 0,81 |
| K_R, sessão inicial, calibrado pelos HC da GaitRec | B3 | 0,867 [0,816; 0,934] | 0,81 |
| K_R, última sessão (n = 124, mediana 21 dias depois) | B1 / B2 / B3 | 0,888 / 0,882 / 0,868 | 0,80–0,83 |

Floresta aleatória (B3): K_R 0,921 [0,865; 0,970], acerto 0,91.
Acerto por lado afetado em K_R (logística, B3): esquerdo 0,77 (n = 73), direito 0,85 (n = 66).

## Leitura

1. O controle negativo falhou: nos saudáveis, esquerdo e direito se distinguem (AUC 0,77 e 0,83).
   Calibrar uma base com a assimetria da outra piora o resultado, então essa assimetria é própria
   de cada base (placa, protocolo ou ordem dos passos), não uma característica comum da marcha.
2. Na K_R, a direção da patologia quase não se mistura com esse efeito de lado: calibrar pelos HC da
   GaitRec muda a AUC de 0,869 para 0,867.
3. O membro afetado é identificado com AUC de 0,87 a 0,92, mas as wavelets não acrescentam nada
   aos atributos clássicos (B1 ≥ B2 ≥ B3). A assimetria da duração do apoio sozinha já é forte.
4. O padrão se mantém na última sessão de acompanhamento.
