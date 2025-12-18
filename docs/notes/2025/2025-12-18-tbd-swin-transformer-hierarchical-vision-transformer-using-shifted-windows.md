---
title: "Swin Transformer: Hierarchical Vision Transformer using Shifted Windows"
authors: ["Ze Liu", "Yutong Lin", "Yue Cao", "Han Hu", "Yixuan Wei", "Zheng Zhang", "Stephen Lin", "Baining Guo"]
venue: "ICCV"
year: 2021
citekey: liu2021swintransformerhierarchicalvision
links:
  pdf: "https://arxiv.org/abs/2103.14030"
  code: "https://github.com/microsoft/Swin-Transformer"
  project: ""
pdf_annotated: "../../papers/2025/swin_transformers.pdf"
tags:
  areas: [computer-vision, transformers, object-detection, semantic-segmentation]
  methods: [shifted-windows, hierarchical-representation, self-attention, vision-transformer]
  tasks: [image-classification, object-detection, semantic-segmentation]
  status: "deep-read"
replication:
  repo: "https://github.com/microsoft/Swin-Transformer"
  data: "ImageNet-1K, ImageNet-22K, COCO, ADE20K"
  env: ""
---

## TL;DR

Swin Transformer introduit un **Transformer hiérarchique** pour la vision avec un schéma de **shifted windows** (fenêtres décalées) permettant un calcul d'attention linéaire par rapport à la taille d'image. L'architecture construit des feature maps hiérarchiques en fusionnant progressivement des patches et limite l'auto-attention à des fenêtres locales non-chevauchantes. Résultats state-of-the-art sur ImageNet-1K (87.3% top-1), COCO object detection (58.7 box AP, +2.7 vs précédent SOTA) et ADE20K semantic segmentation (53.5 mIoU, +3.2 vs SETR), démontrant le potentiel des Transformers comme backbones génériques pour la vision.

## Contexte

En 2021, les CNNs dominent la vision par ordinateur depuis AlexNet (2012), avec des architectures progressivement améliorées (VGG, ResNet, DenseNet, EfficientNet). En NLP, les Transformers sont devenus l'architecture dominante grâce à leur capacité à modéliser les dépendances long-range via l'attention. ViT (2020) a démontré des résultats prometteurs en classification d'images, mais son adaptation à la vision pose des défis : (1) **variation d'échelle** des entités visuelles (objets de tailles très variables) vs tokens de taille fixe en NLP, (2) **haute résolution** des images (millions de pixels) vs textes (centaines de mots), rendant l'auto-attention globale quadratique et computationnellement prohibitive. ViT produit des feature maps à résolution unique et a une complexité quadratique, inadapté pour les tâches de prédiction dense (détection, segmentation) ou images haute résolution.

## Idées clés

1. **Architecture hiérarchique avec patch merging** : Construction de feature maps multi-échelles en fusionnant progressivement les patches voisins dans les couches profondes (résolutions H/4, H/8, H/16, H/32), similaire aux CNNs (VGG, ResNet). Permet d'utiliser des techniques avancées comme FPN ou U-Net pour la prédiction dense et modélise les objets à différentes échelles.

2. **Shifted Window Self-Attention (SW-MSA)** : Calcul d'auto-attention limité à des **fenêtres locales non-chevauchantes** (M×M patches, M=7 par défaut), réduisant la complexité de quadratique O((hw)²) à linéaire O(M²hw). Les fenêtres partitionnent uniformément l'image, chaque fenêtre contient un nombre fixe de patches.

3. **Décalage de fenêtres entre couches consécutives** : Alternance entre partitionnement régulier (layer l) et décalé de (⌊M/2⌋, ⌊M/2⌋) pixels (layer l+1). Les fenêtres décalées **créent des connexions cross-window** en franchissant les frontières des fenêtres précédentes, augmentant significativement le pouvoir de modélisation (Table 4 : +1.1% top-1, +2.8 box AP).

4. **Batch computation efficace via cyclic shift** : Implémentation efficace du shifted windowing par **cyclic-shifting** vers top-left + masking des sub-windows non-adjacentes. Évite l'augmentation naïve du nombre de fenêtres (⌈h/M⌉×⌈w/M⌉ → (⌈h/M⌉+1)×(⌈w/M⌉+1)) et le padding coûteux. Maintient le même nombre de fenêtres avec latence minimale (Table 5).

5. **Relative position bias** : Inclusion d'un biais de position relative B ∈ R^(M²×M²) dans le calcul d'attention : Attention(Q,K,V) = SoftMax(QK^T/√d + B)V. Amélioration significative vs sans biais (+1.2% top-1) ou absolute position embedding (+0.8% top-1). Le relative position bias encourage l'invariance à la translation, cruciale pour la prédiction dense (Table 4).

6. **Complexité linéaire et trade-off efficacité/précision** : Pour une image de h×w patches :
   - Global MSA : Ω(MSA) = 4hwC² + 2(hw)²C (quadratique)
   - Window MSA : Ω(W-MSA) = 4hwC² + 2M²hwC (linéaire quand M fixe)
   - Gain : 40.8×/2.5× plus rapide que sliding window (naive/kernel) au stage 1, 4.1×/1.5× overall speedup pour Swin-T (Table 5)

## Méthode

**Architecture globale** (Figure 3, Table 7) :

*Stage 1 (H/4 × W/4)* :
- **Patch Partition + Linear Embedding** : Split image en patches 4×4 non-overlapping, embedding linéaire : 4×4×3=48 → C dimensions
- **Swin Transformer blocks** : 2 blocs pour Swin-T/S/B/L, maintiennent le nombre de tokens H/4 × W/4

*Stages 2/3/4 (H/8, H/16, H/32)* :
- **Patch Merging** : Concatène 2×2 patches voisins (4C dimensions) + linear layer → 2C dimensions, downsampling 2× de résolution
- **Swin Transformer blocks** : Nombre variable selon modèle (Swin-T : {2,2,6,2}, Swin-B : {2,2,18,2})

**Swin Transformer Block** (Figure 3b, Équation 3) :
```
ẑ^l = W-MSA(LN(z^(l-1))) + z^(l-1)           [regular windowing]
z^l = MLP(LN(ẑ^l)) + ẑ^l

ẑ^(l+1) = SW-MSA(LN(z^l)) + z^l              [shifted windowing]
z^(l+1) = MLP(LN(ẑ^(l+1))) + ẑ^(l+1)
```
- Alternance W-MSA (Window MSA) / SW-MSA (Shifted Window MSA)
- MLP 2-layers avec GELU, LayerNorm avant chaque module, residual connections

**Shifted Window Mechanism** (Figure 2, 4) :

*Regular windowing (layer l)* :
- Partition régulière depuis top-left pixel
- Feature map 8×8 → 2×2 windows de taille 4×4 (M=4)
- Self-attention calculée indépendamment dans chaque fenêtre

*Shifted windowing (layer l+1)* :
- Décalage de (⌊M/2⌋, ⌊M/2⌋) pixels = (2, 2) pour M=4
- Nouvelle partition croise les frontières des fenêtres précédentes

*Efficient batch computation* :
1. **Cyclic shift** vers top-left : déplace les régions A,B,C pour créer des fenêtres alignées
2. **Masked MSA** : masque les interactions entre sub-windows non-adjacentes dans une même fenêtre batchée
3. **Reverse cyclic shift** : repositionnement après attention
4. Nombre de fenêtres = identique au regular partitioning → efficace

**Relative Position Bias** (Équation 4) :
- Matrice B̂ ∈ R^((2M-1)×(2M-1)) paramétrisée (positions relatives dans [-M+1, M-1])
- Values dans B ∈ R^(M²×M²) extraites de B̂
- Learnable, interpolation bi-cubic pour fine-tuning avec window size différente

**Variantes architecturales** (Section 3.3) :
- **Swin-T** : C=96, layers={2,2,6,2}, 29M params, 4.5 GFLOPs (≈ResNet-50)
- **Swin-S** : C=96, layers={2,2,18,2}, 50M params, 8.7 GFLOPs (≈ResNet-101)
- **Swin-B** : C=128, layers={2,2,18,2}, 88M params, 15.4 GFLOPs (≈ViT-B/DeiT-B)
- **Swin-L** : C=192, layers={2,2,18,2}, 197M params, 103.9 GFLOPs (2× Swin-B)
- Window size M=7, query dimension d=32, MLP expansion α=4

**Protocoles d'entraînement** :

*ImageNet-1K (regular training)* :
- AdamW optimizer, 300 epochs, cosine decay LR scheduler, 20 epochs linear warmup
- Batch 1024, LR 0.001, weight decay 0.05, gradient clipping max norm 1
- Augmentation : RandAugment, Mixup, Cutmix, random erasing, stochastic depth (0.2/0.3/0.5 pour T/S/B)
- Pas de repeated augmentation ni EMA (contrairement à ViT/DeiT où crucial)

*ImageNet-22K pre-training* :
- Stage 1 (224²) : AdamW 90 epochs, linear decay LR, batch 4096, LR 0.001, weight decay 0.01
- Stage 2 (ImageNet-1K fine-tuning 224²/384²) : 30 epochs, batch 1024, constant LR 10^-5, weight decay 10^-8

*COCO object detection* :
- Frameworks : Cascade Mask R-CNN, ATSS, RepPoints v2, Sparse RCNN (ablation)
- Multi-scale training (shorter side 480-800, longer ≤1333), AdamW (LR 0.0001, weight decay 0.05, batch 16), 3x schedule (36 epochs)
- System-level : HTC++ avec instaboost, stronger multi-scale (400-1400), 6x schedule, soft-NMS, global self-attention layer après last stage, ImageNet-22K pre-trained

*ADE20K semantic segmentation* :
- Framework : UperNet, AdamW (LR 6×10^-5, weight decay 0.01), linear decay, 1500 iter warmup
- 8 GPUs × 2 images/GPU, 160K iterations
- Augmentation : random horizontal flip, random re-scaling [0.5, 2.0], random photometric distortion, stochastic depth 0.2
- Inference : multi-scale test [0.5, 0.75, 1.0, 1.25, 1.5, 1.75]×

## Résultats

**ImageNet-1K Classification** (Table 1) :

*Regular training (224²)* :
- **Swin-T** : 81.3% vs DeiT-S : 79.8% → **+1.5%**, throughput 755.2 vs 940.4 images/s
- **Swin-S** : 83.0% (8.7G FLOPs) vs RegNet/EfficientNet comparable : meilleur speed-accuracy trade-off
- **Swin-B (224²)** : 83.5% vs DeiT-B : 81.8% → **+1.7%**
- **Swin-B (384²)** : 84.5% vs DeiT-B : 83.1% → **+1.4%**, 47.0G FLOPs vs 55.4G

*ImageNet-22K pre-training* :
- **Swin-B (384²)** : 86.4% top-1 (+1.8~1.9% vs from scratch), throughput 84.7 images/s
- vs ViT-B/16 : 86.4% vs 84.0% → **+2.4%**, throughput similaire (84.7 vs 85.9), moins de FLOPs (47.0G vs 55.4G)
- **Swin-L (384²)** : **87.3% top-1** (+0.9% vs Swin-B) → nouveau SOTA

**COCO Object Detection** (Table 2) :

*Comparison to ResNe(X)t (Cascade Mask R-CNN)* :
- Swin-T vs ResNet-50 : 50.5 vs 46.3 box AP → **+4.2 box AP**, 43.7 vs 40.1 mask AP → **+3.6 mask AP**
- Swin-S vs ResNeXt-101-32x4d : 51.8 vs 48.1 box AP → **+3.7**, 44.7 vs 41.6 mask AP → **+3.1**
- Swin-B vs ResNeXt-101-64x4d : 51.9 vs 48.3 box AP → **+3.6**, 45.0 vs 41.7 mask AP → **+3.3**

*Comparison to DeiT* :
- Swin-T vs DeiT-S† : 50.5 vs 48.0 box AP → **+2.5**, 43.7 vs 41.4 mask AP → **+2.3**
- Inference speed : 15.3 vs 10.4 FPS (DeiT plus lent car complexité quadratique)

*System-level comparison (HTC++)* :
- X101-64 baseline : 52.3 box AP, 46.0 mask AP
- **Swin-B (HTC++)** : 56.4 box AP (+4.1), 49.1 mask AP (+3.1)
- **Swin-L (HTC++)** : 57.1 box AP, 49.5 mask AP sur val
- **Swin-L (HTC++)*** (multi-scale test) : **58.7 box AP, 51.1 mask AP** sur test-dev
  - **+2.7 box AP** vs Copy-paste SOTA (56.0, sans external data)
  - **+2.6 mask AP** vs DetectoRS SOTA (48.5)

**ADE20K Semantic Segmentation** (Table 3) :

*UperNet framework* :
- Swin-T : 46.1 mIoU vs DeiT-S† : 44.0 → **+2.1 mIoU**, throughput 18.5 vs 16.2 FPS
- Swin-S : 49.3 mIoU vs ResNeSt-101 (DLab.v3+) : 46.9 → **+2.4 mIoU**
- Swin-S : 49.3 vs ResNet-101 backbones (DANet 45.2, OCRNet 45.3) → **+4.0~4.1 mIoU**

*ImageNet-22K pre-training* :
- Swin-B‡ (640×640 input) : 51.6 mIoU
- **Swin-L‡** (640×640) : **53.5 mIoU** sur val, **62.8 score** sur test
- **+3.2 mIoU** vs previous SOTA SETR (50.3, T-Large‡)

**Études d'ablation** (Tables 4-6) :

*Shifted windows* (Table 4) :
- Sans shifting : 80.2% / 47.7 box AP / 43.3 mIoU
- Avec shifting : **81.3% / 50.5 box AP / 46.1 mIoU**
- Gains : **+1.1% top-1, +2.8 box AP, +2.8 mIoU** → démontre l'efficacité des connexions cross-window

*Relative position bias* (Table 4) :
- Sans position encoding : 80.1% / 49.2 box AP / 43.8 mIoU
- Absolute position : 80.5% / 49.0 box AP / 43.2 mIoU (pire pour dense tasks)
- **Relative position bias** : **81.3% / 50.5 box AP / 46.1 mIoU**
- Gains : **+1.2% top-1, +1.3 box AP, +2.3 mIoU** vs sans position

*Complexité et vitesse* (Table 5) :
- Cyclic shift vs padding : 13%, 18%, 18% speedup pour Swin-T/S/B
- vs sliding window (naive) : **4.1×, 4.0×, 3.6× faster** pour T/S/B
- vs sliding window (kernel) : **1.5× faster** avec précision similaire (Table 6)
- vs Performer : légèrement plus rapide avec **+2.3% top-1** accuracy

*Input size variations* (Table 8) :
- Swin-T : 224² (81.3%, 755 img/s) → 384² (82.2%, 220 img/s)
- Swin-B : 224² (83.3%, 278 img/s) → 384² (84.5%, 85 img/s)
- Résolution plus grande → meilleure accuracy mais throughput réduit

**Swin MLP-Mixer** (Table 10) :
- Swin-Mixer-B/D24 : 81.3% (10.4G FLOPs) vs MLP-Mixer-B/16 : 76.4% (12.7G) → **+4.9%** avec moins de compute
- vs ResMLP-B24 : 81.3% vs 81.0% → meilleur speed-accuracy trade-off
- Démontre que hierarchical design + shifted windows sont **généralisables** aux architectures all-MLP

## Limites

1. **Complexité d'implémentation du shifted windowing** : Bien que conceptuellement simple, le cyclic shifting + masking nécessite une implémentation soignée. Le gain de vitesse (13-18%) vs padding naïf est significatif mais nécessite des optimisations spécifiques.

2. **Hyperparamètre window size M** : Fixé à M=7 empiriquement pour tous les modèles. Un tuning par stage ou adaptatif pourrait améliorer les performances, mais non exploré. Le choix M=7 est un compromis entre local receptive field et computation cost.

3. **Overhead computationnel vs CNNs** : Bien que linéaire, Swin Transformer reste plus lent que ResNe(X)t optimisés (Cudnn functions). Exemple Swin-T : 755 vs ResNet-50 likely >1000 img/s. L'implémentation PyTorch built-in n'est pas complètement optimisée, des kernel optimizations réduiraient le gap.

4. **Dépendance aux augmentations pour stabilité** : Nécessite augmentations intensives (RandAugment, Mixup, Cutmix, stochastic depth) pour stabiliser l'entraînement. Contrairement à ViT (repeated augmentation crucial), Swin est plus stable mais reste sensible aux stratégies d'augmentation.

5. **Absolute position embedding contre-productif** : L'ajout d'absolute position embedding comme ViT améliore légèrement classification (+0.4%) mais dégrade detection (-0.2 AP) et segmentation (-0.6 mIoU). Suggère que translation invariance partielle via relative position bias est préférable pour dense prediction, mais analyse théorique limitée.

6. **Augmentation de paramètres avec ImageNet-22K** : Les gains sur ImageNet-22K pre-training (+1.8~1.9%) nécessitent un dataset 11× plus large (14.2M images) et deux stages d'entraînement (90 + 30 epochs). Coût computationnel significatif pour amélioration modérée.

7. **Window size fixe inadapté à certaines échelles** : M=7 peut être sous-optimal pour objets très petits ou très grands. Une approche multi-scale ou window size adaptatif améliorerait probablement la détection d'objets de tailles extrêmes.

8. **Comparaison OFFICE avec protocole différent** : System-level comparison utilise ImageNet-22K pre-training + 6x schedule + techniques avancées (instaboost, soft-NMS), rendant la comparaison moins directe avec méthodes baseline. Les gains peuvent être partiellement attribuables aux améliorations orthogonales.

9. **Manque d'analyse théorique** : Pas de justification théorique formelle sur pourquoi shifted windows améliore la modélisation (+2.8 box AP). L'explication intuitive (cross-window connections) est empirique. Une analyse type H∆H-distance ou effective receptive field serait bénéfique.

10. **Extrapolation à résolutions non-testées** : Les résultats sont principalement 224² et 384². L'interpolation bi-cubic du relative position bias pour window sizes différentes est une heuristique sans garantie théorique de préservation des propriétés apprises.

## Liens utiles

- **PDF annoté**: [Swin Transformer annotated](../../papers/2025/swin_transformers.pdf)
- **Article**: [Swin Transformer: Hierarchical Vision Transformer using Shifted Windows (PDF)](https://arxiv.org/pdf/2103.14030)
- **ArXiv**: [2103.14030](https://arxiv.org/abs/2103.14030)
- **Code officiel**: [Swin Transformer GitHub](https://github.com/microsoft/Swin-Transformer)

## Notes perso
