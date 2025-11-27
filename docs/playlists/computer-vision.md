# 🏛️ 🖼️ Computer Vision Foundations

Les papiers fondateurs qui ont révolutionné la vision par ordinateur, de l'ère des CNNs jusqu'aux Transformers

!!! tip "Parcours de lecture"
    Cette playlist contient **13 papiers** sélectionnés pour leur impact sur le domaine.

---

## 1. LeNet-5 (1998)

**Auteurs**: Yann LeCun, Léon Bottou, Yoshua Bengio, Patrick Haffner

**READ**: ✅

!!! quote "Pourquoi ce papier ?"
    Premier CNN moderne appliqué avec succès à la reconnaissance de chiffres manuscrits. Architecture fondatrice du deep learning.

**Liens**:

- 📝 [Mes notes](../notes/2025/2025-11-19-tbd-gradient-based-learning-applied-to-document-recognition.md)
- 🔗 [Article](http://yann.lecun.com/exdb/publis/pdf/lecun-01a.pdf)

---

## 2. AlexNet (2012)

**Auteurs**: Alex Krizhevsky, Ilya Sutskever, Geoffrey E. Hinton

**READ**: ✅

!!! quote "Pourquoi ce papier ?"
    Le papier qui a lancé la révolution deep learning en vision. Première victoire écrasante des CNNs sur ImageNet.

**Liens**:

- 📝 [Mes notes](../notes/2025/2025-11-16-tbd-alexnet-imagenet-classification-with-deep-convolutional-neural-networks.md)
- 🔗 [Article](https://proceedings.neurips.cc/paper_files/paper/2012/file/c399862d3b9d6b76c8436e924a68c45b-Paper.pdf)
- 💻 [Code](http://code.google.com/p/cuda-convnet/)

---

## 3. VGG (2014)

**Auteurs**: Karen Simonyan, Andrew Zisserman

**READ**: ✅

!!! quote "Pourquoi ce papier ?"
    A démontré que la profondeur des réseaux est cruciale. Architecture simple et élégante avec filtres 3x3 uniformes.

**Liens**:

- 📝 [Mes notes](../notes/2025/2025-11-04-tbd-very-deep-convolutional-networks-for-large-scale-image-recognition.md)
- 🔗 [Article](https://arxiv.org/abs/1409.1556)

---

## 4. GoogLeNet/Inception (2014)

**Auteurs**: Christian Szegedy et al.

**READ**: ✅

!!! quote "Pourquoi ce papier ?"
    Introduction des modules Inception permettant d'extraire des features à différentes échelles simultanément.

**Liens**:

- 📝 [Mes notes](../notes/2025/2025-11-27-tbd-going-deeper-with-convolutions.md)
- 🔗 [Article](https://arxiv.org/abs/1409.4842)

---

## 5. ResNet (2015)

**Auteurs**: Kaiming He, Xiangyu Zhang, Shaoqing Ren, Jian Sun

**READ**: ✅

!!! quote "Pourquoi ce papier ?"
    Les connexions résiduelles ont permis d'entraîner des réseaux ultra-profonds (152 couches). Révolution architecturale.

**Liens**:

- 📝 [Mes notes](../notes/2025/2025-10-07-tbd-deep-residual-learning-for-image-recognition.md)
- 🔗 [Article](https://arxiv.org/abs/1512.03385)

---

## 6. Batch Normalization (2015)

**Auteurs**: Sergey Ioffe, Christian Szegedy

**READ**: ✅

!!! quote "Pourquoi ce papier ?"
    Normaliser les activations par mini-batch accélère l'entraînement et régularise le modèle, permettant des learning rates plus élevés.

**Liens**:

- 📝 [Mes notes](../notes/2025/2025-11-02-tbd-batch-normalization-accelerating-deep-network-training-by-reducing-internal-covariate-shift.md)
- 🔗 [Article](https://arxiv.org/abs/1502.03167)

---

## 7. DenseNet (2016)

**Auteurs**: Gao Huang, Zhuang Liu, Laurens van der Maaten, Kilian Q. Weinberger

**READ**: ⬜

!!! quote "Pourquoi ce papier ?"
    Chaque couche est connectée à toutes les couches précédentes, favorisant la réutilisation des features et réduisant les paramètres.

**Liens**:

- 🔗 [Article](https://arxiv.org/abs/1608.06993)

---

## 8. ResNeXt (2017)

**Auteurs**: Saining Xie, Ross Girshick, Piotr Dollár, Zhuowen Tu, Kaiming He

**READ**: ⬜

!!! quote "Pourquoi ce papier ?"
    Étend ResNet avec des transformations agrégées (cardinality), améliorant les performances sans complexité additionnelle.

**Liens**:

- 🔗 [Article](https://arxiv.org/abs/1611.05431)

---

## 9. SENet (2017)

**Auteurs**: Jie Hu, Li Shen, Gang Sun

**READ**: ⬜

!!! quote "Pourquoi ce papier ?"
    Mécanisme d'attention sur les canaux (Squeeze-and-Excitation) qui recalibre dynamiquement les features channels.

**Liens**:

- 🔗 [Article](https://arxiv.org/abs/1709.01507)

---

## 10. EfficientNet (2019)

**Auteurs**: Mingxing Tan, Quoc V. Le

**READ**: ⬜

!!! quote "Pourquoi ce papier ?"
    Scaling uniforme et optimal de la profondeur, largeur et résolution pour maximiser l'efficacité computationnelle.

**Liens**:

- 🔗 [Article](https://arxiv.org/abs/1905.11946)

---

## 11. Vision Transformer (ViT) (2020)

**Auteurs**: Alexey Dosovitskiy et al.

**READ**: ✅

!!! quote "Pourquoi ce papier ?"
    Abandon des convolutions au profit des Transformers. Nouveau paradigme pour la vision avec patches d'images.

**Liens**:

- 📝 [Mes notes](../notes/2025/2025-10-05-tbd-an-image-is-worth-16x16-words-transformers-for-image-recognition-at-scale.md)
- 🔗 [Article](https://arxiv.org/abs/2010.11929)
- 💻 [Code](https://github.com/google-research/vision_transformer)

---

## 12. Swin Transformer (2021)

**Auteurs**: Ze Liu, Yutong Lin, Yue Cao, Han Hu, Yixuan Wei, Zheng Zhang, Stephen Lin, Baining Guo

**READ**: ⬜

!!! quote "Pourquoi ce papier ?"
    Transformer hiérarchique avec fenêtres décalées (shifted windows) pour une complexité linéaire et des features multi-échelles.

**Liens**:

- 🔗 [Article](https://arxiv.org/abs/2103.14030)

---

## 13. ConvNeXt (2022)

**Auteurs**: Zhuang Liu, Hanzi Mao, Chao-Yuan Wu, Christoph Feichtenhofer, Trevor Darrell, Saining Xie

**READ**: ⬜

!!! quote "Pourquoi ce papier ?"
    Modernisation des CNNs avec des designs inspirés des Transformers, prouvant que les convolutions restent compétitives.

**Liens**:

- 🔗 [Article](https://arxiv.org/abs/2201.03545)

---

