---
title: "Deep Residual Learning for Image Recognition"
authors: ["Kaiming He", "Xiangyu Zhang", "Shaoqing Ren", "Jian Sun"]
venue: "arXiv"
year: 2015
citekey: he2015deep
links:
  pdf: "https://arxiv.org/pdf/1512.03385"
  code: ""
  project: ""
pdf_annotated: "/papers/2025/resnets.pdf"
tags:
  areas: [training, computer-vision]
  methods: [Residual-connections, CNN]
  tasks: [convergence, classification]
  status: "deep-read"
replication:
  repo: ""
  data: ""
  env: ""
---

## TL;DR

Les connexions résiduelles (skip connections) permettent d'entraîner des réseaux extrêmement profonds (152 couches) en résolvant le problème de dégradation et en facilitant l'optimisation.

## Contexte

Les réseaux profonds souffrent de dégradation : au-delà d'une certaine profondeur, l'accuracy se dégrade même sur le training set, limitant les bénéfices de la profondeur.

## Idées clés

Apprendre des fonctions résiduelles F(x) = H(x) - x au lieu de H(x) directement facilite l'optimisation ; les skip connections permettent aux gradients de circuler directement.

## Méthode

Blocs résiduels : y = F(x, {Wi}) + x où F est une pile de 2-3 convolutions ; utilisation de bottleneck designs (1x1, 3x3, 1x1) pour les réseaux très profonds.

## Résultats

ResNet-152 remporte ILSVRC 2015 avec 3.57% top-5 error sur ImageNet ; gains significatifs sur COCO detection/segmentation ; le réseau à 1202 couches s'entraîne sans difficulté.

## Limites

Les gains diminuent pour des réseaux extrêmement profonds (1000+ couches) ; potentielle redondance dans les features apprises par certains blocs résiduels.

## Liens utiles

- **Article**: [Deep Residual Learning for Image Recognition (PDF)](https://arxiv.org/pdf/1512.03385)
- **ArXiv**: [https://arxiv.org/abs/1512.03385](https://arxiv.org/abs/1512.03385)

- **Paper explained**: [Yannic Kilcher](https://www.youtube.com/watch?v=GWt6Fu05voI)

## Notes perso
