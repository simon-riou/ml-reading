---
title: "EfficientNet: Rethinking Model Scaling for Convolutional Neural Networks"
authors: ["Mingxing Tan", "Quoc V. Le"]
venue: "ICML"
year: 2020
citekey: tan2020efficientnetrethinkingmodelscaling
links:
  pdf: "https://arxiv.org/abs/1905.11946"
  code: "https://github.com/tensorflow/tpu/tree/master/models/official/efficientnet"
  project: ""
pdf_annotated: "../../papers/2025/EfficientNet.pdf"
tags:
  areas: ["Computer Vision", "Model Scaling", "Neural Architecture"]
  methods: ["CNN", "Compound Scaling", "Neural Architecture Search", "MBConv", "Squeeze-and-Excitation"]
  tasks: ["Image Classification", "Transfer Learning"]
  status: "replicated"
replication:
  repo: ""
  data: "ImageNet"
  env: ""
---

## TL;DR

EfficientNet propose une méthode de **compound scaling** qui scale uniformément la profondeur, largeur et résolution des CNNs avec un coefficient composé. En équilibrant ces trois dimensions avec un ratio fixe (α·β²·γ² ≈ 2), EfficientNet-B7 atteint 84.3% top-1 accuracy sur ImageNet avec 8.4× moins de paramètres et 6.1× plus rapide que GPipe, tout en se transférant exceptionnellement bien sur d'autres datasets.

## Contexte

En 2019, scaler les CNNs pour améliorer l'accuracy se fait principalement en augmentant une seule dimension : profondeur (ResNet-18 → ResNet-200), largeur (WideResNet, MobileNets), ou résolution (224×224 → 480×480). Cependant, cette approche unidimensionnelle nécessite un tuning manuel fastidieux et produit des résultats sous-optimaux. Le problème fondamental est qu'augmenter une dimension seule montre des rendements décroissants rapides au-delà d'un certain point.

Les auteurs identifient que ces dimensions sont interdépendantes : pour des images haute résolution, il faut plus de profondeur (champ réceptif) et plus de largeur (patterns fins). Aucune méthode systématique n'existait pour balancer ces trois dimensions de manière principiée.

## Idées clés

1. **Compound Scaling Method** : Scale uniformément depth (d = α^φ), width (w = β^φ), et resolution (r = γ^φ) avec un coefficient composé φ et la contrainte α·β²·γ² ≈ 2, garantissant que doubler les ressources (φ+1) double approximativement les FLOPS

2. **Balance des dimensions** : Les trois dimensions ne sont pas indépendantes. Pour des images haute résolution, il faut augmenter depth (champs réceptifs plus larges) et width (capturer patterns plus fins), validé empiriquement par les observations 1 et 2

3. **Rendements décroissants** : Scaler une seule dimension améliore l'accuracy mais sature rapidement (observation 1). Le compound scaling atteint +2.5% accuracy vs single-dimension scaling à FLOPS égaux

4. **EfficientNet baseline** : Développé via Neural Architecture Search multi-objectif (optimisant ACC×[FLOPS/T]^w), utilisant MBConv (inverted bottleneck) + Squeeze-and-Excitation comme building blocks

5. **Two-step scaling** : (1) Grid search sur petit modèle (φ=1) pour trouver α, β, γ optimaux, (2) Scale avec ces coefficients fixes pour tous les modèles plus grands → évite le coût prohibitif de re-search sur grands modèles

6. **Meilleure utilisation des ressources** : Les Class Activation Maps montrent que compound scaling capture plus de détails d'objets et se concentre sur des régions plus pertinentes que les autres méthodes

## Méthode

**Formulation du problème** :
- ConvNet : N = ⊙_{i=1..s} F_i^{d·L̂_i}(X_{<r·Ĥ_i, r·Ŵ_i, w·Ĉ_i>})
- Optimisation : max_{d,w,r} Accuracy(N(d,w,r)) sous contraintes Memory et FLOPS
- Contrainte clé : α·β²·γ² ≈ 2 (car FLOPS ∝ d·w²·r²)

**EfficientNet-B0 baseline** (Table 1) :
- Développé via NAS avec search space similaire à MnasNet
- Target : 400M FLOPS (vs ~300M pour MnasNet)
- 9 stages, résolution 224×224 → 7×7, channels 32 → 1280
- Building block : MBConv6 (inverted bottleneck avec expansion ratio 6) + SE
- Kernels : mix de 3×3 et 5×5 selon le stage

**Compound Scaling (2 étapes)** :

STEP 1 - Grid search sur B0 (φ=1, 2× ressources) :
- Trouve α=1.2, β=1.1, γ=1.15 sous contrainte α·β²·γ² ≈ 2
- Ces coefficients restent fixes pour tous les modèles

STEP 2 - Scale avec différents φ :
- B1 à B7 : augmente φ progressivement avec les mêmes α, β, γ
- B7 : φ≈3.7 → depth×2.7, width×2.2, resolution×2.3 (vs B0)

**Training settings** :
- Optimizer : RMSProp (decay 0.9, momentum 0.9)
- Batch norm momentum : 0.99
- Weight decay : 1e-5, learning rate : 0.256 (decay 0.97 every 2.4 epochs)
- Activation : SiLU (Swish-1)
- Regularization : AutoAugment, stochastic depth (survival 0.8), dropout linéaire 0.2→0.5 (B0→B7)
- Early stopping sur 25K minival set

**Validation sur modèles existants** :
- MobileNetV1/V2 et ResNet-50 scalés avec compound method
- Coefficients trouvés via grid search : d=1.4, w=1.2, r=1.3
- Amélioration systématique vs single-dimension scaling

## Résultats

**ImageNet Performance (Table 2)** :
- **EfficientNet-B7** : 84.3% top-1 / 97.0% top-5 avec 66M params, 37B FLOPS
- vs GPipe : même accuracy, 8.4× moins de params, 6.1× plus rapide (CPU inference)
- vs ResNet-152 : +6.5% top-1 avec 7.6× moins de params (B1)
- vs SENet : +1.6% top-1 avec 7.7× moins de params (B4)
- Réduction systématique : jusqu'à 16× moins de FLOPS à accuracy équivalente

**Latency réelle (Table 4, Intel Xeon CPU)** :
- EfficientNet-B1 (78.8% acc) : 5.7× plus rapide que ResNet-152 (77.8% acc)
- EfficientNet-B7 (84.4% acc) : 6.1× plus rapide que GPipe (84.3% acc)

**Transfer Learning (Table 5, 8 datasets)** :
- State-of-the-art sur 5/8 datasets (CIFAR-10, CIFAR-100, Birdsnap, Stanford Cars, Flowers)
- **9.6× moins de paramètres** en moyenne vs meilleurs résultats publics
- CIFAR-100 : 91.7% (B7) vs 87.5% NASNet-A avec 21× moins de params (B0)
- Flowers : 98.8% (B7), nouveau SOTA
- Transfert systématiquement meilleur avec moins de ressources

**Scaling existing models (Table 3)** :
- MobileNetV2 compound : 77.4% (+0.6% vs depth scaling, +1.0% vs width)
- ResNet-50 compound : 78.8% (+0.7% vs depth scaling, +1.1% vs width)
- Démontre l'applicabilité générale de la méthode

**Class Activation Maps (Figure 7)** :
- Compound scaling capture plus de détails d'objets
- Focus sur régions plus pertinentes vs autres méthodes
- Modèles depth/width/resolution seuls manquent de détails ou d'objets complets

**Ablation (Figure 8)** :
- Compound scaling : +2.5% accuracy vs meilleures méthodes single-dimension
- À 1.8B FLOPS : compound 81.1% vs depth 79.0%, width 78.9%, resolution 79.1%

## Limites

1. **Dépendance au baseline** : L'efficacité du scaling dépend fortement de la qualité du réseau de base (d'où l'utilisation de NAS pour EfficientNet-B0)

2. **Coût du NAS** : Le baseline EfficientNet-B0 nécessite Neural Architecture Search, coûteux en ressources computationnelles

3. **Grid search sur petit modèle** : Les coefficients α, β, γ sont optimisés sur B0 puis réutilisés. Une re-optimisation sur modèles plus grands pourrait donner de meilleurs résultats mais est prohibitivement coûteuse

4. **Contrainte d'uniformité** : Le scaling uniforme (même ratio pour tous les layers) pourrait être sous-optimal. Certains layers/stages pourraient bénéficier de ratios différents

5. **Focus ImageNet** : Bien que le transfert learning soit excellent, l'optimisation est centrée sur ImageNet qui pourrait ne pas généraliser à tous les domaines

6. **Augmentation mémoire** : Le compound scaling augmente rapidement la mémoire nécessaire, notamment pour les hautes résolutions (r²)

7. **Regularization manuelle** : Le dropout doit être ajusté manuellement de manière linéaire (0.2→0.5), suggérant que tous les aspects ne sont pas complètement automatisés

8. **Limitations théoriques** : Bien que validé empiriquement, le choix de la contrainte α·β²·γ² ≈ 2 est basé sur l'analyse FLOPS mais d'autres contraintes (latency, mémoire) pourraient être plus pertinentes selon les cas

## Liens utiles

- **Implementation**: [EfficientNet](https://github.com/simon-riou/ml-replicating/blob/master/models/EfficientNet.py)
- **PDF annoté**: [EfficientNet - PDF annoté](../../papers/2025/EfficientNet.pdf)
- **Article**: [EfficientNet: Rethinking Model Scaling for Convolutional Neural Networks (PDF)](https://arxiv.org/pdf/1905.11946)
- **ArXiv**: [https://arxiv.org/abs/1905.11946](https://arxiv.org/abs/1905.11946)
- **Code**: [TensorFlow TPU - EfficientNet](https://github.com/tensorflow/tpu/tree/master/models/official/efficientnet)

## Notes perso
