---
title: "MIXED PRECISION TRAINING"
authors: []
venue: ""
year: 2026
citekey: 
links:
  pdf: ""
  code: ""
  project: ""
pdf_annotated: ""
tags:
  areas: []
  methods: []
  tasks: []
  status: "skimmed"
replication:
  repo: ""
  data: ""
  env: ""
---

## TL;DR

## Contexte

## Idées clés

## Méthode

## Résultats

## Limites

## Liens utiles

- **PDF annoté**: [PDF annoté](../../papers/2026/mixed_precision_training.pdf)
- **Article**: [Mixed Precision Training (PDF)](https://arxiv.org/abs/1710.03740)
- **ArXiv**: [https://arxiv.org/pdf/1710.03740](https://arxiv.org/pdf/1710.03740)

## Notes perso

Guide pytorch pour utiliser mixed precision training: https://docs.pytorch.org/tutorials/recipes/recipes/amp_recipe.html

L'entraînement en précision mixte réduit de moitié l'usage mémoire et accélère les calculs en utilisant le format FP16 pour les poids, activations et gradients. Trois techniques garantissent le maintien de la précision : l'usage d'une copie maître des poids en FP32 pour les mises à jour , une mise à l'échelle de la perte (loss scaling) pour éviter que les petits gradients ne deviennent nuls , et l'accumulation des produits FP16 dans des sorties FP32. Cette approche égale les performances du FP32 sur diverses architectures (classification, détection, parole, traduction) sans ajustement des hyperparamètres.
