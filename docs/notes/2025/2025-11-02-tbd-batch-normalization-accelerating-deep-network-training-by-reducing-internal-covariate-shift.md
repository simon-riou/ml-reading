---
title: "Batch Normalization: Accelerating Deep Network Training by Reducing Internal Covariate Shift"
authors: [Sergey Ioffe, Christian Szegedy]
venue: "arXiv"
year: 2015
citekey: ioffe2015batchnormalizationacceleratingdeep
links:
  pdf: "https://arxiv.org/abs/1502.03167"
  code: ""
  project: ""
pdf_annotated: "/papers/2025/batch_norm.pdf"
tags:
  areas: [training, deep-learning, computer-vision]
  methods: [batch-normalization, normalization]
  tasks: [convergence, classification]
  status: "deep-read"
replication:
  repo: ""
  data: ""
  env: ""
---

## TL;DR

Normaliser les activations de chaque couche par mini-batch accélère significativement l'entraînement, permet d'utiliser des learning rates plus élevés, et régularise le modèle.

## Contexte

L'internal covariate shift (changement de distribution des activations internes pendant l'entraînement) ralentit l'apprentissage et nécessite des learning rates faibles et une initialisation soigneuse.

## Idées clés

Normaliser les inputs de chaque couche (moyenne 0, variance 1) par mini-batch stabilise la distribution des activations ; ajouter des paramètres apprenables γ et β préserve la capacité représentationnelle.

## Méthode

Pour chaque activation : x̂ = (x - μ_B) / √(σ²_B + ε), puis y = γx̂ + β ; utiliser les statistiques du batch en entraînement, moyenne mobile en inférence.

## Résultats

14x moins d'itérations pour atteindre la même accuracy sur ImageNet ; BN-Inception atteint l'accuracy d'Inception en 7x moins de steps et dépasse l'humain (4.9% top-5 error).

## Limites

Dépendance à la taille du batch (instable avec petits batchs) ; comportement différent train/test peut poser problème ; coût computationnel additionnel à l'inférence.

## Liens utiles

- **Article**: [Batch Normalization: Accelerating Deep Network Training by Reducing Internal Covariate Shift (PDF)](https://arxiv.org/pdf/1502.03167)
- **ArXiv**: [https://arxiv.org/abs/1502.03167](https://arxiv.org/abs/1502.03167)


## Notes perso
