---
title: "A ConvNet for the 2020s"
authors: ["Zhuang Liu", "Hanzi Mao", "Chao-Yuan Wu", "Christoph Feichtenhofer", "Trevor Darrell", "Saining Xie"]
venue: "CVPR"
year: 2022
citekey: liu2022convnet2020s
links:
  pdf: "https://arxiv.org/abs/2201.03545"
  code: "https://github.com/facebookresearch/ConvNeXt"
  project: ""
pdf_annotated: "../../papers/2025/convnext.pdf"
tags:
  areas: [computer-vision, image-classification, convolutional-networks]
  methods: [convnet, depthwise-convolution, layer-normalization, large-kernels]
  tasks: [image-classification, object-detection, semantic-segmentation]
  status: "replicated"
replication:
  repo: "https://github.com/simon-riou/ml-replicating/blob/master/models/ConvNeXt.py"
  data: ""
  env: ""
---

## TL;DR

Modernisation systématique de ResNet pour créer **ConvNeXt**, une architecture pure ConvNet compétitive avec les Transformers (Swin-T/S/B/L) sur ImageNet, COCO et ADE20K. Application progressive de design choices inspirés de Swin (training techniques, macro/micro architecture, depthwise conv 7×7, inverted bottleneck, LayerNorm, GELU, fewer normalization/activation layers) permet d'atteindre 87.8% top-1 sur ImageNet-1K, 55.2 box AP sur COCO (ResNet-200 Cascade Mask R-CNN baseline), et 54.0 mIoU sur ADE20K. Démontre que l'écart de performance ConvNets-Transformers provient principalement des training techniques et design choices modernes, pas de l'attention elle-même. Architecture simple, efficace (FLOPs comparables à Swin) et applicable à diverses tâches de vision.

## Contexte

En 2020-2021, l'avènement des Vision Transformers (ViT, Swin) a transformé le paysage de la computer vision. Swin Transformer (Liu et al. 2021) a démontré des performances state-of-the-art sur ImageNet/COCO/ADE20K, surpassant les ConvNets traditionnels. Cependant, les raisons du succès des Transformers restent débattues : est-ce l'attention multi-headed, l'architecture hiérarchique, ou simplement les training techniques modernes (AdamW, mixup, cutmix) ? Les ConvNets standards (ResNet-50/101/200) n'ont pas été modernisés depuis 2015, continuant à utiliser SGD, simples augmentations, et designs figés. L'objectif de ConvNeXt est d'explorer systématiquement les design choices qui permettent aux ConvNets de rivaliser avec les Transformers, en isolant les contributions de chaque modification.

## Idées clés

1. **Roadmap de modernisation ResNet→Swin** : Protocole expérimental contrôlé appliquant séquentiellement training techniques, macro design (stage ratios), ResNeXt-ification (grouped conv), inverted bottleneck, large kernels (7×7), et micro design (LayerNorm, GELU, fewer activations). Chaque étape améliore les performances, permettant d'identifier les contributions individuelles.

2. **Training techniques modernes compensent une partie importante du gap** : Passage de 76.1% (ResNet-50 baseline SGD) à 78.8% (+2.7%) avec AdamW, Mixup, Cutmix, RandAugment, Random Erasing, Stochastic Depth, Label Smoothing. Alignement du training avec Swin-T (90 epochs, AdamW) réduit significativement l'écart de performance avant toute modification architecturale.

3. **Macro design : stage compute ratios (3,3,9,3) → (3,3,27,3)** : Ajustement du nombre de blocs par stage pour suivre Swin-T. Le stage 3 (14×14 spatial resolution) reçoit plus de compute, reflétant l'importance des features de résolution intermédiaire. Changement de stem (7×7 stride-2 conv + maxpool) vers patchify (4×4 non-overlapping conv stride-4) réduit les features redondantes et améliore de 79.4% → 79.5% avec 0.1 GFLOPs economy.

4. **ResNeXt-ification : depthwise convolution comme extreme grouped convolution** : Adoption de depthwise conv (groups = channels) réduit les FLOPs tout en améliorant l'accuracy (79.4% → 80.5% avec 4.6G → 4.5G FLOPs). Augmentation de la largeur de réseau (64→96 channels) compense la réduction de FLOPs et utilise le FLOP budget similaire à Swin-T (80.5% @ 5.3G FLOPs vs Swin-T 81.3% @ 4.5G).

5. **Inverted bottleneck : expansion ratio 4 pour layers à large channels** : Inversion du design ResNet bottleneck (large→narrow→large) vers Transformer MLP-like (narrow→large→narrow). Depthwise conv (computationnellement cheap) opère sur channels élargis (4× expansion), réduisant les downsampling layers coûteux. Amélioration de 80.5% → 80.6% avec réduction de FLOPs (5.3G → 4.6G).

6. **Large kernel sizes (7×7) restaurent le receptive field global** : Déplacement de la depthwise conv vers le haut du bloc (avant 1×1 convs) permet l'utilisation de large kernels sans overhead excessif. Augmentation progressive 3×3 → 5×5 → 7×7 → 9×9 montre saturation à 7×7 (80.6% @ 7×7 vs 80.0% @ 9×9), aligné avec Swin window size. Large kernels compensent la réduction du receptive field due aux depthwise convs.

7. **Micro design : LayerNorm + GELU + Single activation** : Remplacement de BatchNorm par LayerNorm (améliore de 81.4% → 81.5%), activation ReLU → GELU (81.5% → 81.3%, légère baisse mais alignement Transformers), réduction des normalization/activation layers (une seule LN avant chaque bloc, GELU uniquement dans MLP) simplifie l'architecture (81.3% → 82.0%, +0.7%). Design épuré converge vers Transformer block structure.

## Méthode

**Protocole de modernisation progressive** (Figure 2, Table 1) :

**Baseline : ResNet-50** (76.1% @ 4.1G FLOPs) :
- Architecture standard : 4 stages (7×7 stem + maxpool, stage ratios (3,4,6,3), bottleneck 1×1→3×3→1×1)
- Training : SGD momentum 0.9, 90 epochs, batch 1024, learning rate 0.1 cosine decay, weight decay 0.0001

**Step 1 : Training techniques** (78.8%, +2.7%) :
- AdamW optimizer (lr 4e-3, weight decay 0.05)
- Augmentations : Mixup (α=0.8), Cutmix (α=1.0), RandAugment, Random Erasing
- Regularization : Stochastic Depth (p=0.1), Label Smoothing (ε=0.1)
- Training : 300 epochs, batch 4096

**Step 2 : Macro design - Stage compute ratios** (79.4%, +0.6%) :
- Ajustement (3,4,6,3) → (3,3,9,3) (79.4%)
- Patchify stem : 4×4 non-overlapping conv stride-4 (remplace 7×7+maxpool) (79.5%, +0.1%)
- FLOPs réduits à 4.4G

**Step 3 : ResNeXt-ification** (80.5%, +1.0%) :
- Depthwise convolution (groups = channels) (80.5%)
- Augmentation largeur : 64→96 channels pour utiliser budget FLOP similaire à Swin-T (5.3G)

**Step 4 : Inverted bottleneck** (80.6%, +0.1%) :
- Structure : 1×1 expansion conv → depthwise 3×3 → 1×1 projection
- Expansion ratio : 4 (suivant Transformer FFN ratio)
- FLOPs réduits à 4.6G

**Step 5 : Large kernel sizes** (81.4%, +0.8%) :
- Déplacement depthwise conv vers le haut du bloc (avant les 1×1)
- Tests kernel sizes : 3×3 (80.6%) → 5×5 (81.0%) → 7×7 (81.4%) → 9×9 (80.0%)
- Saturation à 7×7, aligné avec Swin window size

**Step 6 : Micro design** (82.0%, +0.6%) :
- Remplacement BatchNorm → LayerNorm (81.5%, +0.1%)
- Activation ReLU → GELU (81.3%, -0.2%)
- Fewer activation layers : une seule LN avant chaque bloc, GELU uniquement dans MLP (82.0%, +0.7%)
- Structure finale ressemble à Transformer block : LN → large-kernel depthwise conv → LN → 1×1 conv → GELU → 1×1 conv

**Architecture finale ConvNeXt** (Figure 3) :

*Bloc ConvNeXt* :
```
Input
  ↓
LayerNorm (εLN = 1e-6)
  ↓
Depthwise Conv 7×7 (large kernel)
  ↓
LayerNorm
  ↓
1×1 Conv (expansion 4×)
  ↓
GELU
  ↓
1×1 Conv (projection)
  ↓
Layer Scale (γ = 1e-6, learnable per-channel)
  ↓
Stochastic Depth
  ↓
Residual Add
```

*Variantes ConvNeXt* (Table 9) :
- **ConvNeXt-T** : C=96, layers=(3,3,9,3), 4.5G FLOPs, 28M params
- **ConvNeXt-S** : C=96, layers=(3,3,27,3), 8.7G FLOPs, 50M params
- **ConvNeXt-B** : C=128, layers=(3,3,27,3), 15.4G FLOPs, 89M params
- **ConvNeXt-L** : C=192, layers=(3,3,27,3), 34.4G FLOPs, 198M params
- **ConvNeXt-XL** : C=256, layers=(3,3,27,3), 60.9G FLOPs, 350M params

*Détails techniques* :
- Layer Scale : γ ∈ ℝ^C initialisé à 1e-6, multiplie chaque channel après conv (améliore stabilité training pour XL)
- Stochastic Depth : p=0.1 (T), 0.4 (S), 0.5 (B/L), 0.6 (XL)
- Downsampling layers : 2×2 conv stride-2, LN avant downsampling
- Stem : 4×4 conv stride-4 (patchify)

**Protocoles d'entraînement** :

*ImageNet-1K* (224×224, 1.28M images) :
- Optimizer : AdamW (β1=0.9, β2=0.999)
- Learning rate : 4e-3, cosine decay, 20 epochs warmup
- Batch size : 4096
- Epochs : 300
- Augmentations : Mixup (0.8), Cutmix (1.0), RandAugment (9, 0.5), Random Erasing (0.25)
- Regularization : Stochastic Depth, Label Smoothing (0.1)
- Weight decay : 0.05
- EMA : decay 0.9999

*ImageNet-22K→1K* (224×224, 14M images pre-training) :
- Pre-training : 90 epochs, warmup 5 epochs
- Fine-tuning : 30 epochs, layer-wise lr decay 0.8

*COCO Object Detection* (Cascade Mask R-CNN framework) :
- Backbone : ConvNeXt pré-entraîné ImageNet-1K
- Optimizer : AdamW (lr 2e-4, weight decay 0.05)
- Training : 3× schedule (36 epochs), multi-scale training
- Augmentations : LSJ (Large Scale Jittering), soft-NMS

*ADE20K Semantic Segmentation* (UperNet framework) :
- Backbone : ConvNeXt pré-entraîné ImageNet-1K
- Optimizer : AdamW (lr 1e-4, weight decay 0.05)
- Training : 160k iterations, batch 16
- Augmentations : random resize, horizontal flip

## Résultats

**ImageNet-1K Classification** (Table 2) :

*Comparison Swin Transformers (224×224, similar FLOPs)* :
- **ConvNeXt-T** : 82.1% top-1 @ 4.5G FLOPs vs Swin-T : 81.3% @ 4.5G (+0.8%)
- **ConvNeXt-S** : 83.1% @ 8.7G vs Swin-S : 83.0% @ 8.7G (+0.1%)
- **ConvNeXt-B** : 83.8% @ 15.4G vs Swin-B : 83.5% @ 15.4G (+0.3%)
- **ConvNeXt-L** : 84.3% @ 34.4G vs Swin-L : 84.5% @ 34.5G (-0.2%)

*ImageNet-22K pre-training (384×384)* :
- **ConvNeXt-B** : 86.8% vs Swin-B : 86.4% (+0.4%)
- **ConvNeXt-L** : 87.5% vs Swin-L : 87.3% (+0.2%)
- **ConvNeXt-XL** : **87.8%** @ 60.9G FLOPs (best pure ConvNet, compétitif avec Swin-L 87.3%)

*Comparison EfficientNetV2 (superior throughput)* :
- ConvNeXt-T : 3×-4× faster inference que EfficientNetV2-S (depthwise 7×7 vs inverted bottlenecks complexes)
- ConvNeXt-T : 82.1% @ 4.5G vs EfficientNetV2-S : 84.9% @ 8.4G (accuracy légèrement inférieure mais FLOP efficiency supérieure)

**COCO Object Detection** (Table 3, Cascade Mask R-CNN 3× schedule) :

*Comparison Swin Transformers* :
- **ConvNeXt-T** : 46.2% box AP, 40.1% mask AP vs Swin-T : 50.4%, 43.7% (+4.2 box, +3.6 mask en faveur de Swin-T)
- **ConvNeXt-B** : 51.8%, 44.9% vs Swin-B : 51.9%, 45.0% (performances équivalentes)
- **ConvNeXt-L** : **53.2%, 45.9%** vs Swin-L : 53.0%, 45.9% (+0.2 box AP)
- **ConvNeXt-XL** : **55.2% box AP** @ 60.9G backbone FLOPs (dépasse Swin-L 53.0%)

*Comparison ResNet Cascade Mask R-CNN baseline* :
- ConvNeXt-T (4.5G) : 46.2% vs ResNet-50 (4.1G) : 46.3% (-0.1%)
- ConvNeXt-B (15.4G) : 51.8% vs ResNet-200 (15.1G) : 51.0% (+0.8%)
- Gains modestes sur detection vs classification (Swin montre pattern similaire)

**ADE20K Semantic Segmentation** (Table 4, UperNet 160k iterations) :

*Comparison Swin Transformers* :
- **ConvNeXt-T** : 46.0% mIoU vs Swin-T : 44.5% (+1.5%)
- **ConvNeXt-S** : 48.7% vs Swin-S : 47.6% (+1.1%)
- **ConvNeXt-B** : 49.1% vs Swin-B : 48.1% (+1.0%)
- **ConvNeXt-L** : **52.9%** vs Swin-L : 52.1% (+0.8%)
- **ConvNeXt-XL** : **54.0% mIoU** (best result, dépasse Swin-L 52.1% par +1.9%)

*Comparison ConvNets traditionnels* :
- ConvNeXt-B (15.4G) : 49.1% vs ResNet-101 : 44.9% (+4.2%)
- Gains substantiels sur segmentation dense vs classification/detection

**Ablations clés** (Table 1) :

*Impact training techniques* : 76.1% → 78.8% (+2.7%, step 1)
*Impact macro design* : 78.8% → 79.5% (+0.7%, steps 2)
*Impact ResNeXt-ification* : 79.5% → 80.5% (+1.0%, step 3)
*Impact inverted bottleneck* : 80.5% → 80.6% (+0.1%, step 4)
*Impact large kernels* : 80.6% → 81.4% (+0.8%, step 5)
*Impact micro design* : 81.4% → 82.0% (+0.6%, step 6)

*Kernel size sweep* (Table 5) :
- 3×3 : 80.6%, 4.6G FLOPs
- 5×5 : 81.0%, 4.7G
- 7×7 : **81.4%**, 4.8G (optimal)
- 9×9 : 80.0%, 5.0G (degradation, kernels trop larges)
- 11×11 : 79.9%, 5.3G

**Isotropic architectures** (Table 7, exploration design space) :

*Depth vs Width trade-offs* (96 layers vs 384 channels) :
- Shallow-wide (d=96, C=384) : 80.4%
- Deep-narrow (d=384, C=96) : 80.7%
- Balanced (d=192, C=192) : **81.1%** (optimal, mais patchify stem + stage ratios (3,3,27,3) donnent 82.0%)

**Effective receptive field** (Figure 4, visualizations) :
- ConvNeXt depthwise 7×7 : ERF similaire à Swin shifted windows M=7
- Large kernels restaurent le global receptive field perdu par depthwise convs
- ConvNeXt-T : ERF spans ~75% de l'image (comparable Swin-T)

## Limites

1. **Performances detection COCO légèrement inférieures à Swin aux petits scales** : ConvNeXt-T (46.2% box AP) vs Swin-T (50.4%, +4.2%). L'écart se réduit aux scales moyens/grands (ConvNeXt-B 51.8% vs Swin-B 51.9%), suggérant que les features ConvNeXt-T manquent de capacité d'adaptation aux multi-scale object detection tasks comparé aux window-based attention de Swin.

2. **Manque d'exploration des training schedules longs** : Tous les résultats ImageNet-1K utilisent 300 epochs. Les Transformers bénéficient parfois de training plus long (1000+ epochs, RegNet). Pas de validation si ConvNeXt converge mieux/pire que Swin avec extended training.

3. **Absence de justification théorique pour certains design choices** : Le choix de Layer Scale γ=1e-6 est empirique, sans analyse de sensibilité systématique. De même, l'expansion ratio 4 suit Transformers mais pas d'ablation sur ratios 2/3/4/6. Certaines décisions semblent guidées par "alignement avec Swin" plutôt que par optimisation indépendante.

4. **Comparaison EfficientNet incomplète** : Table 2 montre que ConvNeXt-T a un throughput supérieur à EfficientNetV2-S (3×-4× faster inference) mais accuracy inférieure (82.1% vs 84.9%). Manque d'analyse approfondie sur trade-offs accuracy/speed/FLOPs pour positioning ConvNeXt vs scaling methods (compound scaling EfficientNet).

5. **Isotropic architectures sous-explorées** : Section 4.3 montre que designs isotropes (depth/width balancé) atteignent 81.1% (vs 82.0% pour hierarchical), mais l'exploration est limitée (3 configurations testées). Les Transformers isotropes (ViT) ont des propriétés intéressantes (simplicité, transferability) qui ne sont pas exploitées ici.

6. **Manque de résultats sur autres downstream tasks** : Pas de résultats sur instance segmentation seule (COCO sans boxes), panoptic segmentation, video tasks (action recognition, tracking), ou 3D vision. Les capacités de généralisation de ConvNeXt sur ces tâches restent inexplorées vs Swin qui a démontré versatilité.

7. **Analyse robustness et out-of-distribution** : Aucune évaluation sur ImageNet-A/C/R (robustness benchmarks), domain shift scenarios, ou adversarial robustness. Les large kernels et LayerNorm pourraient théoriquement améliorer la robustness, mais ce n'est pas validé expérimentalement.

8. **Dependency à l'ImageNet pre-training** : Tous les résultats COCO/ADE20K utilisent backbones pré-entraînés ImageNet-1K/22K. Pas de validation si ConvNeXt apprend mieux from scratch sur detection/segmentation que Swin (self-supervised pre-training, random initialization).

9. **Interprétabilité et attention mechanisms** : Contrairement aux Transformers où les attention maps offrent une interprétabilité (quels tokens sont importants), ConvNeXt dépthwise convs sont moins interprétables. Figure 4 montre ERF mais pas d'analyse qualitative sur ce que les features capturent vs Swin.

10. **Layer Scale et stabilité training** : Layer Scale est nécessaire pour ConvNeXt-XL (350M params) mais l'article ne documente pas les failures sans Layer Scale (instabilités, divergences). Manque de guidelines sur quand/pourquoi Layer Scale est critique (taille modèle, learning rate, batch size).

## Liens utiles

- **Implementation**: [ConvNeXt](https://github.com/simon-riou/ml-replicating/blob/master/models/ConvNeXt.py)
- **PDF annoté**: [ConvNeXt annotated](../../papers/2025/convnext.pdf)
- **Article**: [A ConvNet for the 2020s (PDF)](https://arxiv.org/pdf/2201.03545)
- **ArXiv**: [2201.03545](https://arxiv.org/abs/2201.03545)
- **Code officiel**: [facebookresearch/ConvNeXt](https://github.com/facebookresearch/ConvNeXt)

## Notes perso
