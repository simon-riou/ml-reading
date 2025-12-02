---
title: "Aggregated Residual Transformations for Deep Neural Networks"
authors: ["Saining Xie", "Ross Girshick", "Piotr Dollár", "Zhuowen Tu", "Kaiming He"]
venue: "CVPR"
year: 2017
citekey: xie2017aggregatedresidualtransformationsdeep
links:
  pdf: "https://arxiv.org/abs/1611.05431"
  code: "https://github.com/facebookresearch/ResNeXt"
  project: ""
pdf_annotated: "../../papers/2025/ResNeXt.pdf"
tags:
  areas: ["Computer Vision", "Deep Learning"]
  methods: ["CNN", "Residual Connections", "Grouped Convolutions", "Cardinality", "Split-Transform-Merge"]
  tasks: ["Image Classification", "Object Detection"]
  status: "replicated"
replication:
  repo: "https://github.com/simon-riou/ml-replicating"
  data: "ImageNet-1K, ImageNet-5K, CIFAR-10, CIFAR-100, COCO"
  env: ""
---

## TL;DR

ResNeXt introduit une nouvelle dimension architecturale appelée "cardinalité" (le nombre de transformations agrégées) en plus de la profondeur et de la largeur. En répétant un building block modulaire qui agrège un ensemble de transformations avec la même topologie, ResNeXt améliore la précision tout en maintenant la complexité. Sur ImageNet-1K, ResNeXt-101 (32×4d) surpasse ResNet-200 avec seulement 50% de complexité. Augmenter la cardinalité est plus efficace que d'augmenter la profondeur ou la largeur.

## Contexte

En 2017, le design d'architectures devient de plus en plus complexe avec le nombre croissant d'hyperparamètres (largeur, tailles de filtres, strides, etc.). VGGNet et ResNet ont démontré qu'une stratégie simple (empiler des blocs de même forme) réduit les choix d'hyperparamètres et améliore la robustesse. Les modèles Inception utilisent une stratégie "split-transform-merge" efficace mais nécessitent des designs personnalisés complexes pour chaque module, rendant leur adaptation à de nouveaux datasets difficile.

## Idées clés

1. **Nouvelle dimension : Cardinalité** : Au-delà de la profondeur et de la largeur, la cardinalité C (taille de l'ensemble de transformations) est une dimension essentielle qui contrôle le nombre de transformations complexes agrégées.

2. **Architecture homogène modulaire** : Toutes les transformations partagent la même topologie (contrairement à Inception), permettant d'étendre facilement à un grand nombre de transformations sans designs spécialisés.

3. **Split-Transform-Merge simplifié** : Inspiré d'Inception mais avec une topologie uniforme pour tous les chemins, réalisable via des convolutions groupées (grouped convolutions).

4. **Cardinalité vs Profondeur/Largeur** : Augmenter la cardinalité améliore la précision plus efficacement que d'augmenter la profondeur ou la largeur, surtout quand ces dernières donnent des rendements décroissants.

5. **Équivalence de formulations** : Le module ResNeXt a trois formulations équivalentes : (a) transformations agrégées multiples, (b) concaténation précoce, (c) convolutions groupées (implémentation la plus succincte).

6. **Analogie avec les neurones** : Un neurone simple fait du produit scalaire Σ(w_i·x_i). ResNeXt généralise cela en remplaçant w_i·x_i par des fonctions T_i(x) plus complexes, puis les agrège : F(x) = Σ T_i(x).

## Méthode

**Template modulaire** :
- Suit les deux règles de VGG/ResNet :
  - (i) Blocs produisant des cartes spatiales de même taille partagent les mêmes hyperparamètres
  - (ii) Quand la carte spatiale est sous-échantillonnée (÷2), la largeur est multipliée (×2)
- Notation : ResNeXt-50 (32×4d) signifie cardinalité=32, largeur bottleneck=4d

**Bloc ResNeXt** (Figure 1 droite & Figure 3) :
- Formulation (a) - Transformations agrégées : y = x + Σ T_i(x) où chaque T_i est un bottleneck (1×1 → 3×3 → 1×1)
- Formulation (b) - Concaténation : Similar à Inception-ResNet mais tous les chemins ont la même topologie
- Formulation (c) - Convolutions groupées : La plus succincte, utilisée en pratique
  - Structure : 1×1, 128-d → 3×3, 128-d, groups=32 → 1×1, 256-d
  - Les 128 channels sont divisés en 32 groupes de 4 channels chacun

**Capacité du modèle** :
- Pour préserver la complexité en variant C, ajuster la largeur du bottleneck d
- Formule : C · (256·d + 3·3·d·d + d·256) paramètres par bloc
- Table 2 montre les relations : C=32, d=4 ≈ 70k params (équivalent à ResNet bottleneck)

**Architecture ResNeXt-50/101** (Table 1) :
- ResNeXt-50 : 25.0M params, 4.2×10⁹ FLOPs (similaire à ResNet-50)
- Structure typique : conv1 (7×7) → conv2-5 (stages avec blocs répétés) → GAP → FC
- Exemple conv2 pour ResNeXt-50 (32×4d) : [1×1,128; 3×3,128,C=32; 1×1,256] ×3

**Détails d'implémentation** :
- SGD, mini-batch 256 sur 8 GPUs
- Learning rate 0.1, divisé par 10 trois fois selon schedule [11]
- Weight decay 0.0001, momentum 0.9
- Augmentation : scale and aspect ratio [38], crop 224×224
- Shortcuts : identité sauf pour changements de dimensions (projections type B)
- Downsampling : stride-2 convolutions dans la couche 3×3 du premier bloc de chaque stage
- BN après chaque convolution, ReLU après BN (sauf sortie de bloc : ReLU après addition au shortcut)

## Résultats

**ImageNet-1K - Cardinalité vs Largeur** (Table 3) :

*Complexité préservée (~4.1B FLOPs, ResNet-50)* :
- ResNet-50 (1×64d) : 23.9% top-1 error
- ResNeXt-50 (32×4d) : **22.2%** (-1.7%)
- Tendance claire : augmenter C de 1→32 réduit continûment l'erreur

*Complexité préservée (~7.8B FLOPs, ResNet-101)* :
- ResNet-101 (1×64d) : 22.0%
- ResNeXt-101 (32×4d) : **21.2%** (-0.8%)

**Augmenter Cardinalité vs Profondeur/Largeur** (Table 4, ~15B FLOPs) :
- ResNet-101 baseline : 22.0%
- ResNet-200 (plus profond) : 21.7% (-0.3%)
- ResNet-101 wider (1×100d) : 21.3% (-0.7%)
- ResNeXt-101 (2×64d, doubler C) : 20.7% (-1.3%)
- ResNeXt-101 (64×4d, quadrupler C) : **20.4%** (-1.6%)

**Conclusion claire : augmenter la cardinalité est plus efficace que d'augmenter la profondeur ou la largeur.**

**Connexions résiduelles** :
- Retirer shortcuts de ResNeXt-50 : 22.2% → 26.1% (+3.9%)
- Retirer shortcuts de ResNet-50 : 23.9% → 31.2% (+7.3%)
- Les transformations agrégées sont des représentations plus fortes même sans shortcuts

**Comparaison state-of-the-art** (Table 5) :
- ResNeXt-101 (64×4d) : 20.4% top-1 / 5.3% top-5 (224×224)
- ResNeXt-101 (64×4d) : **19.1% top-1 / 4.4% top-5** (320×320)
- Comparable ou meilleur que Inception-v3/v4, Inception-ResNet-v2
- Architecture beaucoup plus simple que tous les modèles Inception
- 2ème place ILSVRC 2016 classification

**ImageNet-5K** (Table 6, Figure 6) :
- Dataset : 6.8M images, 5K catégories (1K originales + 4K supplémentaires)
- ResNeXt-50 vs ResNet-50 : -3.2% en 5K-way top-1, -2.7% en 1K-way top-1
- ResNeXt-101 vs ResNet-101 : -2.3% en 5K-way top-1, -2.0% en 1K-way top-1
- **Les gains s'amplifient avec plus de données**, confirmant la puissance représentationnelle supérieure

**CIFAR-10/100** (Table 7, Figure 7) :
- ResNeXt-29 (8×64d) : **3.65%** CIFAR-10, **17.77%** CIFAR-100
- ResNeXt-29 (16×64d) : **3.58%** CIFAR-10, **17.31%** CIFAR-100
- Meilleur que Wide ResNet [43] : 4.17% / 20.50%
- **State-of-the-art** sur CIFAR avec data augmentation standard
- Figure 7 montre clairement : augmenter cardinalité > augmenter largeur

**COCO Object Detection** (Table 8) :
- Faster R-CNN avec ResNeXt-50 vs ResNet-50 : +2.1% AP@0.5, +1.0% AP
- Faster R-CNN avec ResNeXt-101 vs ResNet-101 : +0.8% AP@0.5, +0.2% AP
- Gains plus petits sur 101-layer, probablement plus visibles avec plus de données
- ResNeXt adopté dans Mask R-CNN [12] pour state-of-the-art COCO segmentation/détection

## Limites

1. **Overhead computationnel** : L'implémentation brute de grouped convolutions dans Torch n'est pas optimisée pour la parallélisation. ResNeXt-101 (32×4d) prend 0.95s/mini-batch vs 0.70s pour ResNet-101. Une implémentation CUDA optimisée réduirait cet overhead.

2. **Saturation de la cardinalité** : Avec complexité préservée, augmenter C en réduisant la largeur du bottleneck montre une saturation quand d devient très petit (< 4d). Il n'est pas rentable de réduire davantage la largeur.

3. **Gains diminuent sur datasets saturés** : Sur ImageNet-1K, les gains sont plus modestes car le dataset commence à saturer. Les gains sont plus importants sur ImageNet-5K.

4. **Implémentation groupée convolutions requise** : Nécessite des bibliothèques supportant les grouped convolutions (Caffe, Torch, PyTorch). Originalement développé comme compromis engineering pour AlexNet (distribution sur 2 GPUs).

5. **Profondeur minimale requise** : Les reformulations ne produisent des topologies non-triviales que pour des blocs de profondeur ≥3. Avec profondeur=2, cela mène simplement à un module plus large et dense.

6. **Hyperparamètres hérités de ResNet** : Les hyperparamètres d'entraînement suivent ResNet sans tuning spécifique à ResNeXt. Un tuning dédié pourrait améliorer les résultats.

## Liens utiles

- **Implementation**: [ResNeXt](https://github.com/simon-riou/ml-replicating/blob/master/models/ResNeXt.py)
- **PDF annoté**: [ResNeXt annotated](../../papers/2025/ResNeXt.pdf)
- **Article**: [Aggregated Residual Transformations for Deep Neural Networks (PDF)](https://arxiv.org/abs/1611.05431)
- **ArXiv**: [1611.05431](https://arxiv.org/abs/1611.05431)
- **Code officiel**: [ResNeXt GitHub](https://github.com/facebookresearch/ResNeXt)

## Notes perso
