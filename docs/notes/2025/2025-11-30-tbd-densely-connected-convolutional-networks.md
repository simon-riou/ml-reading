---
title: "Densely Connected Convolutional Networks"
authors: ["Gao Huang", "Zhuang Liu", "Laurens van der Maaten", "Kilian Q. Weinberger"]
venue: "CVPR"
year: 2016
citekey: huang2018denselyconnectedconvolutionalnetworks
links:
  pdf: "https://arxiv.org/abs/1608.06993"
  code: "https://github.com/liuzhuang13/DenseNet"
  project: ""
pdf_annotated: "../../papers/2025/densenet.pdf"
tags:
  areas: ["Computer Vision", "Deep Learning"]
  methods: ["CNN", "Dense Connections", "Feature Reuse", "Bottleneck Layers", "Compression"]
  tasks: ["Image Classification"]
  status: "replicated"
replication:
  repo: ""
  data: "ImageNet, CIFAR-10, CIFAR-100, SVHN"
  env: ""
---

## TL;DR

DenseNet connecte chaque couche à toutes les couches suivantes via concaténation (au lieu de sommation comme ResNet), créant L(L+1)/2 connexions dans un réseau à L couches. Cette connectivité dense améliore le flux de gradients, encourage la réutilisation de features, et réduit drastiquement le nombre de paramètres. Atteint des performances state-of-the-art sur CIFAR-10/100, SVHN et ImageNet avec ~3× moins de paramètres que ResNet pour une précision comparable.

## Contexte

En 2016-2018, les réseaux profonds souffrent du problème de vanishing gradient : l'information et les gradients se "diluent" en traversant de nombreuses couches. Les solutions récentes (ResNets, Highway Networks, Stochastic Depth, FractalNets) partagent une caractéristique commune : créer des chemins courts entre couches précoces et tardives. ResNets utilisent des skip connections avec sommation (x_l = H_l(x_{l-1}) + x_{l-1}), mais cette sommation peut impéder le flux d'information.

## Idées clés

1. **Connexions denses** : Chaque couche reçoit en input les feature-maps de TOUTES les couches précédentes via concaténation : x_l = H_l([x_0, x_1, ..., x_{l-1}]). Cela crée L(L+1)/2 connexions au lieu de L.

2. **Concaténation vs sommation** : Contrairement à ResNets qui somment les features, DenseNet les concatène, préservant mieux l'information et évitant que l'identité et H_l(·) s'annulent.

3. **Réutilisation de features** : Les couches ont accès à la "connaissance collective" du réseau, éliminant le besoin de réapprendre des features redondantes. Chaque couche ajoute seulement k feature-maps (growth rate).

4. **Efficacité paramétrique** : DenseNets nécessitent beaucoup moins de paramètres car pas de duplication de features. Les couches sont très étroites (k=12 typiquement).

5. **Supervision implicite profonde** : Chaque couche reçoit une supervision directe de la loss via au plus 2-3 transition layers, facilitant l'entraînement.

6. **Effet régularisant** : Les connexions denses réduisent l'overfitting, particulièrement visible sur datasets sans data augmentation.

## Méthode

**Architecture DenseNet** (Figure 2) :
- Divisé en plusieurs **dense blocks** où toutes les couches sont connectées densément
- **Transition layers** entre blocks : BN → 1×1 Conv → 2×2 Average Pooling (réduction de résolution)
- **Composite function** H_l(·) : BN → ReLU → 3×3 Conv

**Growth rate k** :
- Hyperparamètre crucial définissant le nombre de feature-maps ajoutées par couche
- La l-ième couche a k_0 + k×(l-1) inputs (k_0 = channels initiaux)
- k=12 suffit pour des performances state-of-the-art
- Interprétation : "état global" du réseau accessible partout, chaque couche y contribue k nouvelles features

**DenseNet-B (Bottleneck)** :
- Ajout de 1×1 Conv avant chaque 3×3 Conv pour réduire le nombre d'inputs
- Structure : BN-ReLU-Conv(1×1)-BN-ReLU-Conv(3×3)
- Chaque 1×1 Conv produit 4k feature-maps
- Améliore significativement l'efficacité computationnelle

**DenseNet-C (Compression)** :
- Compression factor θ ∈ (0,1] appliqué aux transition layers
- Si un dense block contient m feature-maps, transition layer génère ⌊θm⌋ outputs
- θ=0.5 dans les expériences (réduit de moitié)
- Réduit encore la taille du modèle

**DenseNet-BC** : Combine bottleneck + compression

**Détails d'implémentation** :

*CIFAR/SVHN* :
- 3 dense blocks avec nombre égal de couches
- Convolution initiale : 16 channels (ou 2k pour DenseNet-BC)
- Zero-padding pour garder taille fixe dans les blocks
- Transition layers : 1×1 Conv + 2×2 Average Pooling
- Global average pooling + softmax à la fin
- Feature-map sizes : 32×32, 16×16, 8×8
- Configurations testées : {L=40, k=12}, {L=100, k=12}, {L=100, k=24}
- Pour DenseNet-BC : {L=100, k=12}, {L=250, k=24}, {L=190, k=40}

*ImageNet* (Table 1) :
- DenseNet-BC avec 4 dense blocks sur images 224×224
- Convolution initiale : 7×7 stride 2 avec 2k channels
- 3×3 max pooling stride 2
- Architectures : DenseNet-121, 169, 201, 264 (nombre de couches)
- Exemple DenseNet-121 : 6+12+24+16 = 58 couches conv dans les blocks + transitions + classification = 121

**Entraînement** :
- SGD avec batch size 64 (CIFAR/SVHN), 256 (ImageNet)
- CIFAR/SVHN : 300/40 epochs, LR initial 0.1, divisé par 10 à 50% et 75% des epochs
- ImageNet : 90 epochs, LR 0.1 divisé par 10 aux epochs 30 et 60
- Weight decay 10^-4, Nesterov momentum 0.9
- Dropout rate 0.2 après chaque conv (sauf première) pour datasets sans augmentation
- Weight initialization de He et al.
- Data augmentation standard sur CIFAR+/ImageNet (mirroring/shifting)

## Résultats

**CIFAR & SVHN** (Table 2) :

*CIFAR-10+* :
- DenseNet-BC (L=190, k=40, 25.6M params) : **3.46%** error (vs 4.62% ResNet-1001 avec 10.2M params)
- 30% de réduction vs FractalNet avec drop-path

*CIFAR-100+* :
- DenseNet-BC (L=190, k=40) : **17.18%** error
- DenseNet-BC (L=250, k=24, 15.3M params) : **17.60%** (vs 20.50% Wide ResNet-28 avec 36.5M params)

*Sans data augmentation* (gains encore plus impressionnants) :
- CIFAR-10 : 5.19% (réduction de 29% vs 7.33% FractalNet)
- CIFAR-100 : 19.64% (réduction de 30% vs 28.20% FractalNet)

*SVHN* :
- DenseNet (L=100, k=24) : **1.59%** error

**ImageNet** (Table 3) :

*Top-1 / Top-5 error (single-crop / 10-crop)* :
- DenseNet-121 (8M params) : 25.02% / 23.61% et 7.71% / 6.66%
- DenseNet-169 (14M params) : 23.80% / 22.08% et 6.85% / 5.92%
- DenseNet-201 (20M params) : 22.58% / 21.46% et 6.34% / 5.54%
- DenseNet-264 (33M params) : 22.15% / 20.80% et 6.12% / 5.29%

*Comparaison avec ResNets* (Figure 3) :
- DenseNet-201 (20M params) ≈ ResNet-101 (44M params) en précision
- DenseNet-201 nécessite ~2× moins de paramètres
- En FLOPs : DenseNet équivalent à ResNet-50 ≈ ResNet-101 (2× moins de compute)

**Efficacité paramétrique** :
- DenseNet-BC (L=100, k=12, 0.8M params) atteint 4.51% sur C10+ vs 4.62% pour ResNet-1001 (10.2M params)
- **90% de paramètres en moins** pour des performances comparables
- **~3× moins de paramètres** que ResNets à précision équivalente sur ImageNet

**Analyse de réutilisation de features** (Figure 5) :
- Toutes les couches utilisent des features de nombreuses couches précédentes (pas juste la précédente)
- Les early features sont directement utilisées par les couches profondes
- Les transition layers reçoivent du poids de toutes les couches du block précédent
- Les outputs des transition layers ont du poids faible (features redondantes) → justifie la compression
- La couche de classification se concentre sur les features tardives (high-level)

## Limites

1. **Complexité mémoire** : La concaténation de toutes les features peut consommer beaucoup de mémoire GPU pendant l'entraînement (résolu dans un technical report [26] avec implémentation memory-efficient)

2. **Hyperparamètres optimisés pour ResNets** : Les auteurs utilisent les mêmes hyperparamètres que ResNets, un tuning spécifique à DenseNet pourrait améliorer les résultats

3. **Gains diminuent sur datasets très faciles** : Sur SVHN, DenseNet-BC 250 couches n'améliore pas vs version plus courte (possiblement overfitting)

4. **Complexité d'implémentation** : Plus complexe à implémenter que des architectures séquentielles simples

5. **Coût computationnel** : Bien que paramètre-efficient, le nombre élevé de concaténations peut augmenter le temps d'entraînement

6. **Scalabilité extrême** : Pas testé au-delà de ~300 couches, contrairement à ResNets testés jusqu'à 1000+ couches

## Liens utiles

- **Implementation**: [DenseNet](https://github.com/simon-riou/ml-replicating/blob/master/models/DenseNet.py)
- **PDF annoté**: [DenseNet annotated](../../papers/2025/densenet.pdf)
- **Article**: [Densely Connected Convolutional Networks (PDF)](https://arxiv.org/abs/1608.06993)
- **ArXiv**: [1608.06993](https://arxiv.org/abs/1608.06993)
- **Code officiel**: [DenseNet GitHub](https://github.com/liuzhuang13/DenseNet)

## Notes perso
