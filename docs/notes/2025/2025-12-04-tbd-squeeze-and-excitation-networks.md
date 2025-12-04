---
title: "Squeeze-and-Excitation Networks"
authors: ["Jie Hu", "Li Shen", "Samuel Albanie", "Gang Sun", "Enhua Wu"]
venue: "CVPR"
year: 2019
citekey: hu2019squeezeandexcitationnetworks
links:
  pdf: "https://arxiv.org/abs/1709.01507"
  code: "https://github.com/hujie-frank/SENet"
  project: ""
pdf_annotated: "../../papers/2025/SENet.pdf"
tags:
  areas: ["Computer Vision", "Deep Learning"]
  methods: ["CNN", "Attention Mechanism", "Channel-wise Attention", "Squeeze-and-Excitation", "Feature Recalibration"]
  tasks: ["Image Classification", "Object Detection", "Scene Classification"]
  status: "replicated"
replication:
  repo: "https://github.com/simon-riou/ml-replicating"
  data: "ImageNet, CIFAR-10, CIFAR-100, Places365, COCO"
  env: ""
---

## TL;DR

SENet introduit un mécanisme d'attention sur les canaux appelé "Squeeze-and-Excitation" (SE) qui recalibre adaptivement les réponses des features en modélisant explicitement les interdépendances entre canaux. Le bloc SE effectue une opération de **squeeze** (global average pooling) pour capturer l'information globale, suivie d'une **excitation** (deux FC layers avec bottleneck) pour générer des poids de modulation par canal. SENet a remporté la 1ère place à ILSVRC 2017 avec 2.251% top-5 error (~25% d'amélioration relative). Les blocs SE sont légers computationnellement et peuvent être intégrés facilement dans toute architecture existante (ResNet, Inception, ResNeXt, MobileNet, etc.) avec des gains de performance significatifs (+0.86% top-5 sur ResNet-50 pour seulement 0.26% de FLOPs supplémentaires).

## Contexte

En 2017-2019, les CNNs fusionnent l'information spatiale et channel-wise via des convolutions locales. La plupart des recherches se concentrent sur la composante spatiale (multi-scale processing dans Inception, spatial attention, spatial dependencies). Les relations entre canaux sont implicitement encodées dans les filtres convolutionnels mais restent entangled avec les corrélations spatiales locales. Les architectures précédentes (VGGNet, ResNet, Inception, ResNeXt, DenseNet) améliorent les performances principalement par la profondeur, la largeur ou la cardinalité, mais sans mécanisme explicite de recalibration des canaux basé sur l'information globale.

## Idées clés

1. **Mécanisme d'attention sur les canaux** : Au lieu de se concentrer uniquement sur l'aspect spatial (comme les travaux précédents), SENet se focalise sur la relation entre les canaux et introduit un mécanisme de recalibration adaptative des features.

2. **Architecture Squeeze-and-Excitation** : Le bloc SE opère en deux étapes séquentielles :
   - **Squeeze** : Agrégation de l'information spatiale globale via global average pooling (H×W → 1×1) pour créer un descripteur par canal avec un champ réceptif global
   - **Excitation** : Self-gating mechanism avec deux FC layers (bottleneck avec ratio r=16) et activation sigmoid pour produire des poids de modulation par canal

3. **Feature recalibration** : Les poids générés par l'excitation sont appliqués aux feature maps via multiplication channel-wise, permettant au réseau d'emphasizer les features informatives et de supprimer les moins utiles de manière input-specific.

4. **Flexibilité d'intégration** : Les blocs SE sont génériques et peuvent être intégrés comme drop-in replacement dans n'importe quelle architecture (après la transformation, avant l'agrégation avec le shortcut pour les réseaux résiduels).

5. **Comportement adaptatif selon la profondeur** : Les blocs SE ont des rôles différents selon la profondeur :
   - Couches précoces : excitent les features de manière class-agnostic (low-level features partagées)
   - Couches tardives : deviennent highly class-specific et répondent différemment selon les inputs
   - Dernières couches : tendent vers un état saturé (activations proches de 1), équivalent à l'identité

6. **Importance de l'information globale** : Les expériences d'ablation (NoSqueeze variant) démontrent que l'utilisation d'embeddings globaux (via squeeze) est cruciale pour la performance, et plus efficace computationnellement que des 1×1 convolutions locales.

## Méthode

**Bloc Squeeze-and-Excitation** (Figure 1, Équations 2-4) :

Pour une transformation F_tr : X ∈ R^(H'×W'×C') → U ∈ R^(H×W×C), le bloc SE opère en 3 étapes :

1. **Squeeze: Global Information Embedding** (Équation 2)
   - Global average pooling sur les dimensions spatiales H×W
   - z_c = (1/(H×W)) Σ_i Σ_j u_c(i,j)
   - Génère un vecteur z ∈ R^C avec champ réceptif global

2. **Excitation: Adaptive Recalibration** (Équation 3)
   - Deux FC layers avec bottleneck (ratio de réduction r=16)
   - s = σ(W_2 δ(W_1 z)) où W_1 ∈ R^(C/r×C), W_2 ∈ R^(C×C/r)
   - δ = ReLU, σ = Sigmoid
   - Génère des poids de modulation s ∈ R^C (non mutually-exclusive)

3. **Scale** (Équation 4)
   - Multiplication channel-wise : x̃_c = s_c · u_c
   - Applique les poids aux feature maps U

**Intégration dans les architectures** (Figures 2-3, Table 1) :

*SE-Inception* :
- F_tr = module Inception entier
- SE block après chaque Inception module (avant scaling final)

*SE-ResNet* :
- F_tr = branche non-identité du residual module
- SE block avant la sommation avec le shortcut
- Exemple SE-ResNet-50 : 3+4+6+3 blocs dans 4 stages
- FC layers dans SE blocks : [16,256], [32,512], [64,1024], [128,2048]

*SE-ResNeXt-50 (32×4d)* :
- Intégration similaire à SE-ResNet
- F_tr = transformations agrégées avec grouped convolutions (C=32)
- Bottleneck width = 4d par groupe

**Détails d'implémentation (Section 5.1)** :

*Entraînement ImageNet* :
- Système distribué ROCS : synchronous SGD, momentum 0.9, batch size 1024
- Learning rate initial : 0.6, divisé par 10 tous les 30 epochs, 100 epochs total
- Data augmentation : random crop (scale + aspect ratio) à 224×224, random horizontal flip
- Mean RGB-channel subtraction, weight initialization [He et al.]
- Reduction ratio r=16 par défaut

*Entraînement CIFAR* :
- Data augmentation standard : random horizontal flip, zero-padding 4 pixels puis random crop 32×32
- Mean/std normalization

*Entraînement Places365* :
- Training from scratch
- Même protocole que ImageNet

*COCO Object Detection* :
- Faster R-CNN framework
- Weights initialisés par le modèle pré-entraîné ImageNet
- '2x' learning schedule, end-to-end training

**Complexité (Section 4, Équation 5)** :

Paramètres additionnels : (2/r) Σ_s N_s · C_s²

Exemple SE-ResNet-50 (r=16) :
- GFLOPs : 3.87 (vs 3.86 pour ResNet-50) → +0.26% overhead
- Paramètres : 28.1M (vs 25.6M) → +10% params, réductible à +4% en retirant le dernier stage
- Runtime : 209ms forward+backward vs 190ms pour ResNet-50 (batch 256, 8 Titan X)
- CPU inference : 167ms vs 164ms (224×224 input)

## Résultats

**ImageNet-1K Classification** (Tables 2-3) :

*Profondeur du réseau* (ResNets) :
- SE-ResNet-50 : 22.28% / 6.03% (top-1/top-5) vs ResNet-50 : 23.30% / 6.55% → **gains de +1.02% / +0.52%**
- SE-ResNet-50 (6.62% top-5) approche ResNet-101 (6.52%) avec moitié moins de compute (3.87 vs 7.58 GFLOPs)
- SE-ResNet-101 : 22.38% / 6.07% → **dépasse ResNet-152** (22.42% / 6.34%) avec moins de paramètres
- SE-ResNet-152 : 21.57% / 5.73%

*Architectures modernes* :
- SE-Inception-ResNet-v2 : 19.80% / 4.79% vs 20.37% / 5.21% → **+0.57% / +0.42%**
- SE-ResNeXt-50 (32×4d) : 21.10% / 5.49% vs ResNeXt-50 : 22.11% / 5.90% → **+1.01% / +0.41%**
- SE-ResNeXt-101 : 20.70% / 5.01% → dépasse ResNeXt-101 (21.18% / 5.57%)
- SE-VGG-16 (avec BN) : 25.22% / 7.70% vs VGG-16 : 27.02% / 8.81% → **+1.80% / +1.11%**
- SE-BN-Inception : 24.23% / 7.14% vs 25.38% / 7.89% → **+1.15% / +0.75%**

*Architectures mobiles* (Table 3) :
- SE-MobileNet : 25.3% / 7.7% vs MobileNet : 28.4% / 9.4% → **+3.1% / +1.7%** (572 vs 569 MFLOPs)
- SE-ShuffleNet : 31.0% / 11.1% vs ShuffleNet : 32.6% / 12.5% → **+1.6% / +1.4%** (142 vs 140 MFLOPs)

*Gains cohérents durant l'optimisation* (Figure 4) :
- Les courbes d'entraînement montrent que SE blocks améliorent l'optimisation de manière steady et consistent pour ResNet-50/101/152, BN-Inception

**CIFAR-10/100** (Tables 4-5) :

*CIFAR-10* :
- SE-ResNet-110 : 5.21% vs ResNet-110 : 6.37% → **-1.16%**
- SE-ResNet-164 : 4.39% vs ResNet-164 : 5.46% → **-1.07%**
- SE-WRN-16-8 : 3.88% vs WRN-16-8 : 4.27% → **-0.39%**
- SE-Shake-Shake 26 2×96d + Cutout : **2.12%** vs 2.56% → state-of-the-art

*CIFAR-100* :
- SE-ResNet-110 : 23.85% vs 26.88% → **-3.03%**
- SE-ResNet-164 : 21.31% vs 24.33% → **-3.02%**
- SE-WRN-16-8 : 19.14% vs 20.43% → **-1.29%**
- SE-Shake-Even 29 2×4×64d + Cutout : **15.41%** vs 15.85%

**Places365 Scene Classification** (Table 6) :
- SE-ResNet-152 : 40.37% / 11.01% (top-1/top-5)
- ResNet-152 : 41.15% / 11.61% → **+0.78% / +0.60%**
- Dépasse Places-365-CNN (11.48% top-5) → nouveau state-of-the-art

**COCO Object Detection** (Table 7) :
- Faster R-CNN avec SE-ResNet-50 : 61.0% / 40.4% (AP@0.5 / AP)
- Faster R-CNN avec ResNet-50 : 57.9% / 38.0% → **+3.1% / +2.4%** (relative 6.3% sur AP)
- Faster R-CNN avec SE-ResNet-101 : 62.7% / 41.9%
- Faster R-CNN avec ResNet-101 : 60.1% / 39.9% → **+2.6% / +2.0%** (relative 5.0% sur AP)

**ILSVRC 2017 Competition** (Tables 8-9, Section 5.4) :

*SENet-154 (single model)* :
- 224×224 crop : 18.68% / 4.47% (top-1/top-5)
- 320×320 crop : **17.28% / 3.79%** → meilleur résultat single-model à l'époque

*Ensemble SENet (winning submission)* :
- Multi-scale + multi-crop fusion : **2.251% top-5 error** sur test set
- ~25% d'amélioration relative vs winner 2016 (2.991%)

*Comparaison state-of-the-art* (320×320 crops, ImageNet seulement) :
- SENet-154 : 17.28% / 3.79%
- DPN-131 : 18.55% / 4.16%
- PyramidNet-200 : 19.2% / 4.7%
- Inception-ResNet-v2 : 19.9% / 4.9%

## Limites

1. **Overhead computationnel** : Bien que modeste (0.26% FLOPs pour SE-ResNet-50), l'implémentation pratique montre ~10% d'augmentation du temps d'entraînement (209ms vs 190ms). Des optimisations GPU spécifiques pourraient réduire cet overhead.

2. **Augmentation des paramètres** : +10% de paramètres pour SE-ResNet-50 (réductible à ~4% en retirant les SE blocks du dernier stage avec <0.1% de perte de performance). Peut être problématique pour les applications avec contraintes mémoire strictes.

3. **Choix du ratio de réduction** : r=16 est un compromis empirique. Un tuning plus fin par stage pourrait améliorer les performances, mais les auteurs n'ont pas exploré cette direction de manière exhaustive.

4. **Saturation en fin de réseau** : Les derniers SE blocks (SE_5_2, SE_5_3) tendent vers un état saturé (activations ≈ 1), équivalent à l'identité. Cela suggère une redondance potentielle, confirmée par l'ablation montrant qu'ils peuvent être retirés avec peu d'impact.

5. **Analyse théorique limitée** : L'approche est principalement empirique. Une compréhension théorique plus profonde des mécanismes d'attention sur les canaux et de leur interaction avec la profondeur du réseau serait bénéfique.

6. **Dépendance au global pooling** : Le choix du global average pooling est simple mais d'autres stratégies d'agrégation plus sophistiquées (second-order statistics, etc.) pourraient offrir des gains supplémentaires.

7. **Hyperparamètres non-optimisés** : Les auteurs utilisent les mêmes hyperparamètres d'entraînement que les architectures de base (ResNet, etc.). Un tuning spécifique à SENet pourrait améliorer les résultats.

8. **Position du SE block** : Bien que les auteurs aient exploré plusieurs variantes (SE-PRE, SE-POST, SE-Identity), d'autres positions ou configurations (par exemple, au niveau des couches 3×3 spécifiques) pourraient être plus optimales pour certaines architectures.

## Liens utiles

- **Implementation**: [SENet](https://github.com/simon-riou/ml-replicating/blob/master/models/SENet.py)
- **PDF annoté**: [SENet annotated](../../papers/2025/SENet.pdf)
- **Article**: [Squeeze-and-Excitation Networks (PDF)](https://arxiv.org/abs/1709.01507)
- **ArXiv**: [1709.01507](https://arxiv.org/abs/1709.01507)
- **Code officiel**: [SENet GitHub](https://github.com/hujie-frank/SENet)

## Notes perso
