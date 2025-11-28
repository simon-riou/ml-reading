---
title: "Going deeper with convolutions"
authors: ["Christian Szegedy", "Wei Liu", "Yangqing Jia", "Pierre Sermanet", "Scott Reed", "Dragomir Anguelov", "Dumitru Erhan", "Vincent Vanhoucke", "Andrew Rabinovich"]
venue: "CVPR"
year: 2014
citekey: szegedy2014going
links:
  pdf: "https://arxiv.org/abs/1409.4842"
  code: ""
  project: ""
pdf_annotated: "../../papers/2025/inception.pdf"
tags:
  areas: ["Computer Vision", "Deep Learning"]
  methods: ["CNN", "Inception Module", "Auxiliary Classifiers", "1x1 Convolutions", "Multi-scale Processing"]
  tasks: ["Image Classification", "Object Detection"]
  status: "replicated"
replication:
  repo: ""
  data: "ImageNet ILSVRC-2014"
  env: ""
---

## TL;DR

GoogLeNet est un réseau à 22 couches introduisant les modules Inception pour une utilisation efficace des ressources de calcul. L'architecture permet d'augmenter largeur et profondeur tout en gardant un budget computationnel constant (1.5B multiply-adds). Utilise des convolutions 1×1 pour la réduction de dimensionnalité et traite l'information à différentes échelles simultanément. Atteint 6.67% d'erreur top-5 sur ILSVRC 2014 avec 12× moins de paramètres qu'AlexNet.

## Contexte

En 2014, les progrès en vision par ordinateur sont principalement dus aux CNNs profonds. La tendance est d'augmenter la taille et la profondeur des réseaux ([12], [21, 14]) avec dropout [7] pour l'overfitting. Cependant, augmenter uniformément la taille pose deux problèmes : (1) plus de paramètres = plus d'overfitting, (2) augmentation quadratique du coût computationnel. Le défi ILSVRC 2014 utilise 1.2M images d'entraînement, 50K validation, 100K test sur 1000 classes.

Les auteurs s'inspirent du principe Hebbien et du travail théorique d'Arora et al. [2] qui suggère qu'une topologie optimale peut être construite couche par couche en analysant les statistiques de corrélation des activations. L'architecture vise aussi l'efficacité pour le déploiement mobile/embarqué.

## Idées clés

1. **Module Inception** : Combine convolutions 1×1, 3×3, 5×5 et max-pooling en parallèle, permettant le traitement multi-échelle et l'extraction de features à différentes granularités

2. **Convolutions 1×1 pour réduction dimensionnelle** : Utilisées avant les convolutions 3×3 et 5×5 coûteuses pour réduire drastiquement le nombre de calculs tout en préservant l'information

3. **Architecture sparse approximée par des composants denses** : L'idée est de construire une structure sparse optimale en utilisant des blocs denses disponibles (clustering de matrices sparse)

4. **Classifieurs auxiliaires** : Ajoutés sur les couches intermédiaires (Inception 4a et 4d) pour (a) combattre le vanishing gradient, (b) encourager la discrimination dans les couches basses, (c) régulariser (poids de loss = 0.3)

5. **Efficacité computationnelle** : Budget fixe de 1.5B multiply-adds pour permettre l'utilisation pratique, même sur large datasets

6. **Average pooling avant le classifieur** : Remplace les couches fully-connected (améliore top-1 de 0.6%), dropout reste essentiel même après cette modification

## Méthode

**Architecture GoogLeNet** (Table 1) :
- 22 couches avec paramètres (27 si on compte le pooling)
- ~100 building blocks au total
- Input : 224×224 RGB avec mean subtraction
- Commence avec convolutions traditionnelles, puis 9 modules Inception empilés
- Max-pooling stride 2 pour réduire la résolution entre groupes de modules

**Module Inception** (Figure 2b) :
- Version naïve : 1×1, 3×3, 5×5 conv + 3×3 max-pool en parallèle, concaténation des filtres
- Version optimisée : ajout de 1×1 conv de réduction avant 3×3 et 5×5, et après pooling
- Toutes les convolutions utilisent ReLU
- Les ratios 3×3/5×5 augmentent dans les couches hautes (features plus abstraites, concentration spatiale diminue)

**Détails d'implémentation** :
- Receptive field : 224×224
- Depth = 2 dans chaque Inception module
- Architecture détaillée dans Table 1 (nombre de filtres pour chaque branche)

**Classifieurs auxiliaires** (sur Inception 4a et 4d) :
- Average pooling 5×5 stride 3 → 4×4×512 (ou 528)
- Conv 1×1 avec 128 filtres + ReLU
- FC 1024 units + ReLU
- Dropout 70%
- Linear layer + softmax (1000 classes)
- Utilisés pendant l'entraînement uniquement, loss pondéré à 0.3

**Entraînement** (DistBelief [4]) :
- SGD asynchrone, momentum 0.9
- Learning rate fixe, décrémenté de 4% toutes les 8 epochs
- Polyak averaging [13] pour le modèle final
- Data augmentation : patches 8%-100% de l'image, aspect ratio 3/4 à 4/3
- Distorsions photométriques [8]
- Interpolation aléatoire (bilinear, area, nearest, cubic)
- Temps d'entraînement : ~1 semaine sur quelques GPUs haut de gamme

**Testing** :
- 7 modèles GoogLeNet en ensemble (même initialisation, différentes méthodes de sampling)
- Multi-scale : 4 résolutions (256, 288, 320, 352 sur dimension courte)
- Multi-crop : 3 positions (gauche/centre/droite ou haut/centre/bas) × 5 crops (4 coins + centre) × 2 (miroir) × 4 échelles = 144 crops
- Average des softmax probabilities sur tous les crops et modèles

## Résultats

**ILSVRC 2014 Classification** :
- **Top-5 error : 6.67%** (1er, vs 7.32% VGG, 7.35% MSRA)
- Réduction de 56.5% vs SuperVision 2012 (16.4%)
- Réduction de 40% vs Clarifai 2013 (11.7%)
- Sans données externes

**Ablations** (Table 3, validation set) :
- 1 modèle, 1 crop : 10.07%
- 1 modèle, 10 crops : 9.15% (-0.92%)
- 1 modèle, 144 crops : 7.89% (-2.18%)
- 7 modèles, 1 crop : 8.09% (-1.98%)
- 7 modèles, 10 crops : 7.62% (-2.45%)
- 7 modèles, 144 crops : **6.67%** (-3.45%)

**ILSVRC 2014 Detection** :
- **mAP : 43.9%** (1er, vs 40.7% CUHK, 40.5% Deep Insight)
- Approche R-CNN avec Inception comme classifieur de régions
- Selective Search + multi-box [5] pour propositions (60% des propositions vs R-CNN, coverage 92%→93%)
- Ensemble de 6 ConvNets
- Pas de bounding box regression (manque de temps)
- Single model : 38.02% mAP

**Comparaison paramètres** :
- GoogLeNet : 6.8M paramètres (~1000K dans Table 1 linear layer suggère ~7M total)
- AlexNet : ~60M paramètres
- **12× moins de paramètres qu'AlexNet** pour de meilleures performances

## Limites

1. **Design manuel** : Les choix architecturaux (ratios de filtres, placement des modules) sont basés sur l'intuition et validation manuelle, pas complètement automatisés

2. **Justification théorique partielle** : Bien que inspiré par Arora et al. [2], difficile de prouver que les performances sont dues aux principes théoriques vs au design empirique

3. **Complexité d'implémentation** : Architecture plus complexe que VGG, nécessite infrastructure sophistiquée

4. **Tailles de filtres limitées** : Restreint à 1×1, 3×3, 5×5 par commodité plutôt que par nécessité (patch-alignment issues)

5. **Local Response Normalization** : Utilisée dans les premières couches mais sera remplacée par Batch Normalization

6. **Entraînement distribué** : Nécessite DistBelief pour l'entraînement efficace, bien qu'estimé à ~1 semaine sur quelques GPUs

7. **Testing coûteux** : 144 crops par image pour obtenir les meilleurs résultats (pas nécessaire en pratique)

## Liens utiles

- **Implementation**: [LeNet-5](https://github.com/simon-riou/ml-replicating/blob/master/models/GoogleLeNet.py)
- **PDF annoté**: [Going deeper with convolutions annotated](../../papers/2025/inception.pdf)
- **Article**: [Going deeper with convolutions (PDF)](https://arxiv.org/abs/1409.4842)
- **ArXiv**: [1409.4842](https://arxiv.org/abs/1409.4842)

## Notes perso
