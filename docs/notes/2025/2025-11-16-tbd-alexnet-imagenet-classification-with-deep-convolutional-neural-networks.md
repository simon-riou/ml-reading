---
title: "AlexNet: ImageNet Classification with Deep Convolutional Neural Networks"
authors: ["Alex Krizhevsky", "Ilya Sutskever", "Geoffrey E. Hinton"]
venue: "NIPS"
year: 2012
citekey: article 
links:
  pdf: "https://proceedings.neurips.cc/paper_files/paper/2012/file/c399862d3b9d6b76c8436e924a68c45b-Paper.pdf"
  code: "http://code.google.com/p/cuda-convnet/"
  project: ""
pdf_annotated: "../../papers/2025/AlexNet.pdf"
tags:
  areas: ["Computer Vision", "Deep Learning"]
  methods: ["CNN", "ReLU", "Dropout", "Data Augmentation", "GPU Training"]
  tasks: ["Image Classification"]
  status: "replicated"
replication:
  repo: "https://github.com/simon-riou/ml-replicating/blob/master/models/AlexNet.py"
  data: "ImageNet ILSVRC-2010/2012"
  env: ""
---

## TL;DR

AlexNet est un réseau de neurones convolutionnel profond à 8 couches (5 conv + 3 FC) qui a remporté ILSVRC-2012 avec 60M de paramètres. Innovations clés : ReLU pour accélérer l'entraînement, dropout pour réduire l'overfitting, entraînement sur 2 GPUs, et augmentation de données. Atteint 15.3% d'erreur top-5 sur ImageNet (vs 26.2% pour la 2ème place).

## Contexte

Avant AlexNet, les approches de reconnaissance d'objets utilisaient des features hand-crafted (SIFT, Fisher Vectors). Les datasets étaient petits (CIFAR, Caltech-101/256) et les CNNs étaient limités par la puissance de calcul. ImageNet (15M images, 22K catégories) a permis l'entraînement de modèles plus larges. Le défi ILSVRC utilisait un sous-ensemble de 1.2M images d'entraînement et 1000 classes.

## Idées clés

1. **ReLU Nonlinearity** : `f(x) = max(0,x)` entraîne 6x plus rapidement que tanh/sigmoid (fonctions saturantes)

2. **Entraînement Multi-GPU** : Distribution du réseau sur 2 GTX 580 3GB avec communication sélective entre couches

3. **Local Response Normalization** : Normalisation inspirée de l'inhibition latérale neuronale (remplacée par Batch Norm aujourd'hui)

4. **Overlapping Pooling** : Stride s=2, taille z=3 (s < z) réduit l'overfitting vs pooling traditionnel

5. **Dropout (p=0.5)** : Appliqué aux 2 premières FC layers, double le temps de convergence mais réduit drastiquement l'overfitting

6. **Data Augmentation** :
   - Extractions aléatoires 224×224 depuis images 256×256 + flips horizontaux (×2048 augmentation)
   - PCA color augmentation (altération RGB basée sur composantes principales)

## Méthode

**Architecture** :
- Input : 224×224×3
- Conv1 : 96 kernels 11×11×3, stride 4 → ReLU → LRN → MaxPool
- Conv2 : 256 kernels 5×5×48 → ReLU → LRN → MaxPool
- Conv3 : 384 kernels 3×3×256 → ReLU
- Conv4 : 384 kernels 3×3×192 → ReLU
- Conv5 : 256 kernels 3×3×192 → ReLU → MaxPool
- FC6 : 4096 neurons → ReLU → Dropout
- FC7 : 4096 neurons → ReLU → Dropout
- FC8 : 1000 neurons → Softmax

**Détails d'entraînement** :
- SGD avec batch size 128, momentum 0.9, weight decay 0.0005
- Learning rate 0.01, divisé par 10 quand validation error stagne
- Initialisation : Gaussian(0, 0.01) pour poids, biaises à 1 pour conv2/4/5 et FC, 0 ailleurs
- 90 epochs (~5-6 jours sur 2 GTX 580)

**Test-time** : Average des prédictions sur 10 patches (4 coins + centre + flips horizontaux)

## Résultats

**ILSVRC-2010** :
- Top-1 : 37.5% (vs 47.1% sparse coding)
- Top-5 : 17.0% (vs 28.2% sparse coding)

**ILSVRC-2012** :
- 1 CNN : 18.2% top-5
- 5 CNNs ensemble : 16.4% top-5
- 7 CNNs (avec pre-training ImageNet complet) : **15.3% top-5** (gagnant vs 26.2% 2ème place)

**Ablations** :
- Multi-GPU : -1.7% top-1, -1.2% top-5
- LRN : -1.4% top-1, -1.2% top-5
- Overlapping pooling : -0.4% top-1, -0.3% top-5
- PCA color augmentation : -1% top-1
- Retirer une seule conv layer : ~-2% top-1 (la profondeur est critique)

**ImageNet Fall 2009** (10K classes, 8.9M images) :
- Top-1 : 67.4%, Top-5 : 40.9% (vs 78.1%/60.9% précédent SOTA)

## Limites

1. **Mémoire GPU limitée** : Nécessite 2 GPUs pour 1.2M exemples d'entraînement
2. **Temps d'entraînement** : 5-6 jours sur 2 GTX 580
3. **Local Response Normalization** : Remplacée depuis par Batch Normalization (plus efficace)
4. **Architecture manuelle** : Pattern de connectivité GPU cross-validé manuellement
5. **Pas de pre-training non-supervisé** : Les auteurs mentionnent que cela pourrait aider
6. **Images statiques** : Pas d'exploitation de l'information temporelle (vidéo)

## Liens utiles

- **Implementation**: [AlexNet](https://github.com/simon-riou/ml-replicating/blob/master/models/AlexNet.py)
- **PDF annoté**: [AlexNet annotated](../../papers/2025/AlexNet.pdf)
- **Article**: [ImageNet Classification with Deep Convolutional Neural Networks (PDF)](https://proceedings.neurips.cc/paper_files/paper/2012/file/c399862d3b9d6b76c8436e924a68c45b-Paper.pdf)
- **Code original**: [cuda-convnet](http://code.google.com/p/cuda-convnet/)
- **ResearchGate**: [Publication page](https://www.researchgate.net/publication/267960550_ImageNet_Classification_with_Deep_Convolutional_Neural_Networks)

## Notes perso
