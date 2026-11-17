# Frente 2 — resumo (rodada 2, 16/11/2026: atributos clássicos revisados)

A rodada 1 (02/11/2026, B1 com V_min e AP_impulso) está arquivada em `../resultados_B1_v1/`.
Só o conjunto B1 mudou; B2 (wavelets) reproduz a rodada 1 até a quarta casa decimal.

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

- B1 revisado (12 atributos): V = primeiro pico, vale, segundo pico, impulso; AP = pico e impulso de
  frenagem e de propulsão; ML = máximo, mínimo, impulso; duração do apoio. B3 passa a ter 57 atributos.
- Sinal do AP conferido nos dados: frenagem negativa no início do apoio, propulsão positiva no fim
  (GaitRec −71 N e +95 N; Gutenberg −116 N e +133 N, médias do 1º e do último quarto do apoio).
- Dois picos distintos em 99,8% dos membros (profundidade do vale < 0,02 peso corporal em 0,2%).
- Medianas (peso corporal): GaitRec pico1 1,06, vale 0,81, pico2 1,06; Gutenberg pico1 1,17, vale 0,71, pico2 1,18.
  A curva da GaitRec é mais achatada, possivelmente por um critério de inclusão da própria base.
- Gutenberg: 3.748 trials (7.496 membros); GaitRec: 5.371 trials (10.742 membros). Nenhuma lacuna interna.
- Observação: na GaitRec, o pico vertical nunca fica abaixo de 1,0 peso corporal (mínimo 0,9999),
  o que sugere um critério de inclusão da própria base.

## Etapa 3 — Calibração e primeira passada (logística, IC 95% por bootstrap)

| Análise | Atributos | AUC [IC 95%] | Acerto |
| --- | --- | --- | --- |
| Controle negativo, GaitRec HC (esq. × dir.) | B3 | 0,765 [0,705; 0,836], p = 0,005 | 0,71 |
| Controle negativo, Gutenberg HC (esq. × dir.) | B3 | 0,847 [0,811; 0,893], p = 0,005 | 0,77 |
| Idem, calibrado pela outra base | B3 | 0,958 (GaitRec) e 0,932 (Gutenberg) | — |
| K_R, sessão inicial | B1 clássico (v2) | 0,909 [0,863; 0,963] | 0,90 |
| K_R, sessão inicial | B2 wavelet | 0,884 [0,837; 0,947] | 0,81 |
| K_R, sessão inicial | B3 clássico + wavelet | 0,882 [0,835; 0,949], p = 0,005 | 0,83 |
| K_R, sessão inicial, calibrado pelos HC da GaitRec | B1 / B3 | 0,909 / 0,880 | 0,88 / 0,83 |
| K_R, última sessão (n = 124, mediana 21 dias depois) | B1 / B2 / B3 | 0,900 / 0,882 / 0,879 | 0,81–0,85 |

Comparação com a rodada 1 (K_R, logística): B1 0,899 → 0,909; B3 0,869 → 0,882; B2 inalterado (0,884).
Outras subclasses (B1 v1 → v2): K_F 0,961 → 0,964; K_P 0,931 → 0,927.
Floresta aleatória (B3): K_R 0,923 [0,868; 0,970], acerto 0,90.
Acerto por lado afetado em K_R (logística, B3): esquerdo 0,79 (n = 73), direito 0,88 (n = 66).

Assimetria (afetado − contralateral) em K_R, tamanho de efeito dz: pico 2 de V −0,80; pico de frenagem −0,70;
impulso de frenagem −0,67; duração do apoio −0,69; pico 1 de V −0,42; vale de V +0,35; pico de propulsão −0,14.

## Leitura

1. O controle negativo continua falhando (AUC 0,76 e 0,85): esquerdo e direito se distinguem nos
   saudáveis, e calibrar com a outra base não resolve. O efeito de lado é próprio de cada base.
2. Na K_R, calibrar pelos HC da GaitRec quase não altera a AUC (0,882 → 0,880).
3. Com B1 corrigido, os atributos clássicos ficam um pouco à frente (0,909 contra 0,884 das wavelets);
   os intervalos se sobrepõem, então não há diferença demonstrada, mas as wavelets também não superam
   os clássicos. As diferenças mais fortes são pico 2 de V, frenagem e duração do apoio.
4. O padrão se mantém na última sessão de acompanhamento.
5. Cuidado metodológico: a troca de atributos foi feita depois de ver a rodada 1 (por razão
   biomecânica, não por ajuste da AUC). Ambas as rodadas estão arquivadas e devem ser relatadas.
