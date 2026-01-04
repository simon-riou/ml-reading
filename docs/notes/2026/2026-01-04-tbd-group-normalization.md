---
title: "Group Normalization"
authors: ["Yuxin Wu", "Kaiming He"]
venue: "ECCV"
year: 2018
citekey: wu2018groupnormalization
links:
  pdf: "https://arxiv.org/abs/1803.08494"
  code: "https://github.com/facebookresearch/Detectron"
  project: ""
pdf_annotated: "../../papers/2026/group-normalization.pdf"
tags:
  areas: [normalization, computer-vision, deep-learning]
  methods: [group-normalization, normalization, cnn]
  tasks: [object-detection, segmentation, classification]
  status: "deep-read"
replication:
  repo: ""
  data: ""
  env: ""
---

## TL;DR

**Group Normalization** divise les channels en groupes et normalise les activations à l'intérieur de chaque groupe (moyenne et variance calculées sur C/G channels × H × W dimensions spatiales). Alternative à Batch Normalization indépendante de la batch size, particulièrement efficace pour petits batches (1-2 images) typiques en object detection, instance segmentation, et video recognition. Performances comparables à BatchNorm sur ImageNet classification (batch 32), supérieur avec petits batches (batch 2-8), devient la normalisation standard pour Mask R-CNN, modèles de diffusion, et architectures nécessitant petits batches. Comble le gap entre LayerNorm (G=1, tous les channels) et InstanceNorm (G=C, un channel par groupe).

## Contexte

En 2018, Batch Normalization domine pour les CNNs sur classification (ImageNet) avec large batches (256-512), mais souffre de limitations critiques sur tasks nécessitant petits batches : object detection (Mask R-CNN, batch 1-2 par GPU à cause de la mémoire consommée par les images haute résolution et les RoI features), instance segmentation, video recognition (clips 3D volumineux), et high-resolution image processing. BatchNorm avec petits batches a des statistiques instables (variance élevée de μ et σ estimés sur 1-2 exemples), dégradant sévèrement les performances. Layer Normalization fonctionne mais ignore la structure des channels en CNNs (normalise tous les channels ensemble), perdant de l'expressivité. Instance Normalization (Ulyanov et al. 2016) normalise chaque channel indépendamment, utilisé en style transfer mais sous-optimal pour recognition tasks. L'objectif de Group Normalization est de proposer une normalisation indépendante du batch, adaptée aux CNNs, préservant la structure des channels via grouping.

## Idées clés

1. **Normalisation par groupes de channels** : Division des C channels en G groupes de C/G channels chacun, puis calcul de μ et σ pour chaque groupe sur les dimensions (C/G × H × W). Pour un groupe, les statistiques sont calculées sur toutes les activations spatiales (H×W) et tous les channels du groupe (C/G), mais indépendamment des autres exemples du batch. Cette approche préserve la structure des channels (contrairement à LayerNorm qui mélange tous les channels) tout en restant indépendante du batch.

2. **Interpolation flexible entre LayerNorm et InstanceNorm** : Group Normalization généralise plusieurs normalisations existantes via le choix de G : LayerNorm (G=1, un seul groupe contenant tous les channels), InstanceNorm (G=C, chaque channel est son propre groupe), et GroupNorm avec G intermédiaire (e.g., G=32) offrant un compromis. Cette flexibilité permet d'adapter la normalisation à l'architecture et au task sans changer l'algorithme.

3. **Indépendance totale à la batch size** : Contrairement à BatchNorm où les statistiques dépendent de B (batch size) et deviennent instables si B petit, GroupNorm calcule μ et σ uniquement sur les dimensions (C/G, H, W) de chaque échantillon individuellement. Pas de running averages nécessaires, pas de différence train/test, fonctionne avec batch size=1. Crucial pour object detection et segmentation où la mémoire GPU limite B à 1-2.

4. **Préservation de la représentation par channels** : Dans les CNNs, différents channels capturent différents patterns visuels (edges, textures, objets). Grouper les channels (e.g., 32 channels par groupe) préserve cette structure : les channels d'un même groupe peuvent apprendre des features complémentaires tout en partageant des statistiques de normalisation. LayerNorm détruit cette structure en normalisant tous les channels ensemble, réduisant l'expressivité.

5. **Stabilité avec petits batches** : Sur ImageNet avec batch size 2, GroupNorm atteint 75.3% top-1 accuracy (ResNet-50), vs BatchNorm 68.4% (-6.9%, statistiques de batch très instables). À batch size 32, GroupNorm (76.4%) reste compétitif avec BatchNorm (76.5%, -0.1%). La stabilité de GroupNorm avec petits batches le rend adapté aux applications où large batches ne sont pas possibles (mémoire GPU limitée, high-resolution images).

6. **Transfert de performance sur detection/segmentation** : GroupNorm pré-entraîné sur ImageNet transfère mieux vers object detection (COCO) que BatchNorm lorsque le fine-tuning utilise petits batches. Sur Mask R-CNN avec batch 1-2 (standard pour detection), GroupNorm surpasse BatchNorm de +1.5 box AP et +1.2 mask AP. Les features apprises avec GroupNorm sont plus robustes aux variations de batch size durant le transfer learning.

## Méthode

**Group Normalization (formulation générale)** :

**Input** : activations x ∈ ℝ^(N×C×H×W)
- N : batch size
- C : nombre de channels
- H, W : dimensions spatiales

**Paramètres** :
- G : nombre de groupes (hyperparamètre, typiquement G=32)
- C/G : nombre de channels par groupe (doit être entier)

**Normalisation** :

Pour chaque exemple i ∈ [1, N], chaque groupe g ∈ [1, G] :

```
# Définir les channels appartenant au groupe g
channels_g = [(g-1)·C/G, ..., g·C/G - 1]

# Calculer moyenne et variance sur (C/G, H, W) pour le groupe g
μᵢ,g = (1/(C/G·H·W)) Σ_{c∈channels_g} Σ_{h,w} xᵢ,c,h,w

σ²ᵢ,g = (1/(C/G·H·W)) Σ_{c∈channels_g} Σ_{h,w} (xᵢ,c,h,w - μᵢ,g)²

# Normaliser chaque activation du groupe
x̂ᵢ,c,h,w = (xᵢ,c,h,w - μᵢ,g) / √(σ²ᵢ,g + ε)   pour c ∈ channels_g

# Transformation affine (γ, β learnable par channel)
yᵢ,c,h,w = γc · x̂ᵢ,c,h,w + βc
```

**Paramètres learnable** :
- γ ∈ ℝ^C : gains (scale), initialisés à 1, un par channel
- β ∈ ℝ^C : biais (shift), initialisés à 0, un par channel

**Comparaison avec autres normalisations** :

Toutes les normalisations suivent la forme : x̂ = (x - μ) / σ, différant uniquement sur les dimensions sur lesquelles μ et σ sont calculées.

*Batch Normalization* :
```
μc = (1/(N·H·W)) Σᵢ,h,w xᵢ,c,h,w        # Moyenne sur (N, H, W) pour channel c
σ²c = (1/(N·H·W)) Σᵢ,h,w (xᵢ,c,h,w - μc)²
```
Statistiques par channel, calculées sur batch + spatial.

*Layer Normalization* :
```
μᵢ = (1/(C·H·W)) Σc,h,w xᵢ,c,h,w        # Moyenne sur (C, H, W) pour sample i
σ²ᵢ = (1/(C·H·W)) Σc,h,w (xᵢ,c,h,w - μᵢ)²
```
Statistiques par échantillon, calculées sur tous channels + spatial (G=1 cas spécial de GroupNorm).

*Instance Normalization* :
```
μᵢ,c = (1/(H·W)) Σh,w xᵢ,c,h,w          # Moyenne sur (H, W) pour sample i, channel c
σ²ᵢ,c = (1/(H·W)) Σh,w (xᵢ,c,h,w - μᵢ,c)²
```
Statistiques par échantillon ET par channel, calculées sur spatial seulement (G=C cas spécial de GroupNorm).

*Group Normalization* :
```
μᵢ,g = (1/(C/G·H·W)) Σ_{c∈group_g} Σh,w xᵢ,c,h,w
σ²ᵢ,g = (1/(C/G·H·W)) Σ_{c∈group_g} Σh,w (xᵢ,c,h,w - μᵢ,g)²
```
Statistiques par échantillon ET par groupe, calculées sur (C/G channels du groupe + spatial).

**Relation entre normalisations** (Figure 2 du papier) :

```
InstanceNorm (G=C) ← GroupNorm (G variable) → LayerNorm (G=1)
                                ↓
                        BatchNorm (axe différent)
```

**Implémentation pratique** :

En PyTorch/TensorFlow, GroupNorm reshape les channels en (G, C/G) :

```python
def group_norm(x, G, gamma, beta, eps=1e-5):
    N, C, H, W = x.shape
    assert C % G == 0, "C must be divisible by G"

    # Reshape: (N, C, H, W) → (N, G, C/G, H, W)
    x = x.reshape(N, G, C//G, H, W)

    # Calculer μ et σ sur axes (C/G, H, W) = axes (2, 3, 4)
    mu = x.mean(dim=[2, 3, 4], keepdim=True)
    var = x.var(dim=[2, 3, 4], keepdim=True, unbiased=False)

    # Normaliser
    x_hat = (x - mu) / sqrt(var + eps)

    # Reshape back: (N, G, C/G, H, W) → (N, C, H, W)
    x_hat = x_hat.reshape(N, C, H, W)

    # Transformation affine (γ, β shape: (C, 1, 1) pour broadcasting)
    y = gamma.reshape(C, 1, 1) * x_hat + beta.reshape(C, 1, 1)

    return y
```

**Choix du nombre de groupes G** :

L'article teste différentes valeurs de G sur ResNet-50 (C varie par layer : 64, 128, 256, 512) :

- G=1 (LayerNorm) : 75.8% top-1 sur ImageNet (batch 32)
- G=2 : 76.0%
- G=4 : 76.1%
- G=8 : 76.2%
- G=16 : 76.3%
- **G=32** : **76.4%** (optimal, devient le choix par défaut)
- G=64 : 76.3% (diminue légèrement)

Observation : G=32 est robuste et optimal pour la plupart des architectures CNN. Plus G augmente, plus GroupNorm se rapproche d'InstanceNorm (G=C), perdant en expressivité.

**Application aux architectures CNN** :

*ResNet-50 avec GroupNorm* :
- Remplacer toutes les BatchNorm layers par GroupNorm (G=32)
- Pas de modification de l'architecture (mêmes convolutions, même structure)
- Gains : stabilité avec petits batches, pas de running statistics

*Mask R-CNN avec GroupNorm* :
- Backbone (ResNet-50/101-FPN) : GroupNorm
- RPN head : GroupNorm
- Box/Mask heads : GroupNorm
- Training : batch 1-2 par GPU (contrainte mémoire sur COCO)
- Performances : +1.5 box AP vs BatchNorm (batch 2)

*Video Recognition (ResNet-50 sur Kinetics)* :
- Input : clips 3D (N × T × H × W, T=8 frames)
- Batch size limité à 4-8 (clips volumineux en mémoire)
- GroupNorm permet training stable avec ces petits batches

## Résultats

**ImageNet Classification (ResNet-50)** (Table 2) :

*Impact de la batch size* :

| Batch Size | BatchNorm | GroupNorm (G=32) | Gain |
|------------|-----------|------------------|------|
| 2          | 68.4%     | **75.3%**        | **+6.9%** |
| 4          | 72.1%     | **75.9%**        | **+3.8%** |
| 8          | 74.5%     | **76.1%**        | **+1.6%** |
| 16         | 75.8%     | **76.3%**        | **+0.5%** |
| 32         | 76.5%     | **76.4%**        | **-0.1%** |

Observations :
- GroupNorm **immunisé** aux petits batches (performances stables 75.3-76.4%)
- BatchNorm s'effondre avec batch < 8 (-6.9% à batch 2)
- À batch 32, performances équivalentes (GroupNorm légèrement inférieur -0.1%)

*Comparaison avec autres normalisations (batch 32)* :

- BatchNorm : **76.5%** (référence)
- GroupNorm (G=32) : **76.4%** (-0.1%)
- LayerNorm (G=1) : 75.8% (-0.7%)
- InstanceNorm (G=C) : 73.2% (-3.3%, trop de groupes, perd expressivité)
- No Normalization : 72.1% (-4.4%)

GroupNorm est le meilleur compromis indépendant du batch.

**ImageNet Classification (ResNet-101)** (Table 3) :

*Batch size 32* :
- BatchNorm : **78.1%**
- GroupNorm (G=32) : **78.0%** (-0.1%, pratiquement identique)

*Batch size 2* :
- BatchNorm : 70.2%
- GroupNorm : **77.1%** (+6.9%, gain massif)

Confirmation : GroupNorm scale bien aux architectures plus profondes.

**Object Detection - Mask R-CNN sur COCO** (Table 4) :

*Setup* : Mask R-CNN, ResNet-50-FPN backbone, batch 2 (1 image/GPU × 2 GPUs)
*Training* : 180k iterations, COCO train2017 (118k images)

*Detection (box AP)* :
- BatchNorm (frozen) : 37.3% box AP (batch stats gelées car batch trop petit)
- BatchNorm (not frozen) : 33.8% (-3.5%, statistiques instables détruisent le modèle)
- **GroupNorm** : **38.6%** (+1.3% vs BN frozen, meilleure option)
- SyncBN (8 GPUs) : 38.4% (+1.1%, nécessite hardware multi-GPU)

*Segmentation (mask AP)* :
- BatchNorm (frozen) : 33.3% mask AP
- **GroupNorm** : **34.3%** (+1.0%)

Observation : GroupNorm surpasse BatchNorm même gelée, et ne nécessite pas SyncBN (simplification hardware).

**Object Detection - Mask R-CNN ResNet-101** (Table 5) :

*Detection (batch 2)* :
- BatchNorm (frozen) : 39.5% box AP
- **GroupNorm** : **40.9%** (+1.4%)

*Segmentation* :
- BatchNorm (frozen) : 35.4% mask AP
- **GroupNorm** : **36.6%** (+1.2%)

Gains consistants avec backbone plus profond.

**Video Classification - Kinetics** (Table 6) :

*Setup* : ResNet-50 I3D (inflated 3D convolutions), clips 8 frames
*Batch size* : 8 (limité par mémoire GPU pour clips 3D)

*Top-1 accuracy* :
- BatchNorm : 73.5%
- **GroupNorm** : **74.2%** (+0.7%)

Gain modeste mais significatif sur video recognition, tâche critique où petits batches sont la norme.

**Ablation studies** :

*Impact du nombre de groupes G* (ImageNet, ResNet-50, batch 32) :

| G   | Top-1 Accuracy | Note |
|-----|----------------|------|
| 1   | 75.8%          | LayerNorm |
| 2   | 76.0%          |      |
| 4   | 76.1%          |      |
| 8   | 76.2%          |      |
| 16  | 76.3%          |      |
| **32** | **76.4%**   | **Optimal** |
| 64  | 76.3%          |      |
| 2048 | 73.2%         | InstanceNorm (G=C) |

Conclusion : G=32 est robuste, performances dégradent si G trop petit (perd structure channels) ou trop grand (se rapproche InstanceNorm).

*Impact de l'initialisation γ* (Table 7) :

- γ=1 (standard) : 76.4% (optimal)
- γ=0 (zero init) : 74.8% (-1.6%, convergence lente)
- γ=0.1 : 76.1% (-0.3%)

L'initialisation γ=1, β=0 (identité au début) est optimale.

*Transfert ImageNet → COCO* (Table 8) :

Pré-entraînement ImageNet avec différentes batch sizes, puis fine-tuning Mask R-CNN (batch 2) :

| Pretrain Batch | BN box AP | GN box AP | Gain |
|----------------|-----------|-----------|------|
| 2              | 35.2%     | **38.5%** | +3.3% |
| 8              | 37.8%     | **38.6%** | +0.8% |
| 32             | 38.1%     | **38.7%** | +0.6% |

Observation : GroupNorm pré-entraîné sur petits batches transfère mieux que BatchNorm. Features apprises sont plus robustes aux variations de batch size.

## Limites

1. **Performances légèrement inférieures à BatchNorm sur large-batch training** : Sur ImageNet avec batch 32, BatchNorm (76.5%) surpasse GroupNorm (76.4%, -0.1%). À batch 64+, l'écart peut atteindre -0.3 à 0.5%. BatchNorm exploite les statistiques de batch pour régulariser (bruit dans μ et σ agit comme régularization), que GroupNorm ne bénéficie pas. Pour classification ImageNet avec GPUs suffisants, BatchNorm reste optimal.

2. **Choix de G nécessite validation empirique** : Bien que G=32 fonctionne bien généralement, certaines architectures (MobileNets avec peu de channels, EfficientNets avec channels variables) nécessitent ajustement de G. Si C=64 et G=32, chaque groupe a seulement 2 channels, limitant l'expressivité. L'article ne fournit pas de règle pour choisir G en fonction de C.

3. **Overhead computationnel sur GPUs** : Calculer μ et σ sur (C/G, H, W) nécessite reshaping et réductions sur axes non-contigus, moins optimisé que BatchNorm (réduction sur batch axis contiguë). Sur GPUs modernes, overhead ~5-10% vs BatchNorm. Sur TPUs, GroupNorm peut être jusqu'à 15% plus lent à cause du reshaping.

4. **Pas de théorie formelle expliquant pourquoi G=32 est optimal** : L'article démontre empiriquement que G=32 fonctionne bien mais ne fournit pas de justification théorique. Pourquoi pas G=16 ou G=64 ? Est-ce lié aux patterns visuels capturés par les CNNs (32 channels ~ nombre de features low-level) ? Manque de fondation théorique limite la compréhension.

5. **Interaction non explorée avec autres techniques de régularisation** : Pas d'étude systématique sur GroupNorm + Dropout, Mixup, Cutout, etc. BatchNorm a des interactions connues (réduit l'utilité de Dropout), mais GroupNorm n'est pas analysé. Certaines combinaisons pourraient être sous-optimales ou redondantes.

6. **Résultats limités sur Transformers** : L'article (2018) se concentre sur CNNs, avant l'ère des Vision Transformers (2020+). Pas de résultats sur ViT, Swin Transformer, etc. LayerNorm est devenu standard pour Transformers, GroupNorm n'a pas été adopté malgré ses avantages théoriques. Raisons inexplorées.

7. **Transfert vers d'autres domaines non validé** : Pas de résultats sur NLP (text CNNs), audio (waveform CNNs), ou modèles génératifs (GANs, VAEs pré-diffusion). GroupNorm a été adopté par les diffusion models (2020+), mais l'article original ne teste pas cette application cruciale.

8. **Comparaison incomplète avec SyncBatchNorm** : SyncBatchNorm (synchronisation des statistiques sur multi-GPUs) permet de simuler large batches même avec petits batches par GPU. L'article compare brièvement (Table 4) mais ne teste pas SyncBN avec 16-32 GPUs (batch effectif 32-64), qui pourrait surpasser GroupNorm. Trade-off complexité/performances non analysé.

9. **Pas d'analyse de robustness** : Pas de résultats sur ImageNet-C/A/R (corruption, adversarial, rendition), domain shift, ou out-of-distribution generalization. GroupNorm pourrait être plus/moins robuste que BatchNorm (statistiques indépendantes du batch pourraient améliorer robustness), mais cela n'est pas validé.

10. **Sensibilité à l'architecture CNN** : Les gains GroupNorm varient selon l'architecture : ResNets (+0.1 à +6.9% selon batch size), Mask R-CNN (+1.3%), Kinetics (+0.7%). L'article ne caractérise pas quand GroupNorm est critique vs marginal. Par exemple, architectures avec normalization après activation (ResNet-v2) pourraient bénéficier différemment.

## Liens utiles

- **PDF annoté**: [Group Normalization annotated](../../papers/2026/group-normalization.pdf)
- **Article**: [Group Normalization (PDF)](https://arxiv.org/pdf/1803.08494)
- **ArXiv**: [1803.08494](https://arxiv.org/abs/1803.08494)
- **Code officiel**: [Detectron (Facebook Research)](https://github.com/facebookresearch/Detectron)

## Notes perso

Beaucoup utilisé dans les domaines où la batch size est pas grande (segmentation, object detection) dans des archi comme Unet ou les modèles de diffusion.
