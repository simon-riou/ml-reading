---
title: "Very Deep Convolutional Networks for Large-Scale Image Recognition"
authors: [Karen Simonyan, Andrew Zisserman]
venue: "arXiv"
year: 2015
citekey: simonyan2015deepconvolutionalnetworkslargescale
links:
  pdf: "https://arxiv.org/abs/1409.1556"
  code: ""
  project: ""
pdf_annotated: "/papers/2025/vgg.pdf"
tags:
  areas: [training, computer-vision, SOTA]
  methods: [CNN, VGG]
  tasks: [classification, convergence]
  status: "skimmed"
replication:
  repo: ""
  data: ""
  env: ""
---

## TL;DR

Augmenter la profondeur des CNNs (16-19 couches) avec des filtres 3x3 uniformes améliore significativement les performances en classification d'images, démontrant que la profondeur est un facteur clé.

## Contexte

Avant VGG, les CNNs utilisaient des filtres de tailles variées (7x7, 5x5) et étaient moins profonds ; la question de l'impact de la profondeur sur les performances restait à explorer systématiquement.

## Idées clés

Empiler plusieurs petits filtres 3x3 équivaut à un champ réceptif plus large mais avec moins de paramètres et plus de non-linéarités ; la profondeur est plus importante que la taille des filtres.

-> 3 filtres 3x3 <==> 1 filtre 7x7 (receptive field), les 3 filtres sont une sorte de régularization, du fait de la non-linéarité entre les filtres, du filtre 7x7.

## Méthode

Architectures VGG-16 et VGG-19 : couches convolutionnelles 3x3 exclusivement, max-pooling 2x2, trois FC layers finales ; augmentation progressive de la profondeur pour évaluer l'impact systématiquement.

## Résultats

1ère place en localisation et 2ème en classification à ILSVRC 2014 ; 7.3% top-5 error sur ImageNet ; excellente généralisation sur d'autres datasets après fine-tuning.

## Limites

Coût computationnel très élevé (nombre massif de paramètres ~138M pour VGG-16) ; temps d'entraînement long ; consommation mémoire importante limitant le déploiement.

## Liens utiles

- **PDF annoté**: [VGG - PDF annoté](../../papers/2025/vgg.pdf)
- **Article**: [Very Deep Convolutional Networks for Large-Scale Image Recognition (PDF)](https://arxiv.org/pdf/1409.1556)
- **ArXiv**: [https://arxiv.org/abs/1409.1556](https://arxiv.org/abs/1409.1556)

## Notes perso
