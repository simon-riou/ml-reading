---
title: "Dropout: A Simple Way to Prevent Neural Networks from Overfitting"
authors: ["Nitish Srivastava", "Geoffrey Hinton", "Alex Krizhevsky", "Ilya Sutskever", "Ruslan Salakhutdinov"]
venue: "Journal of Machine Learning Research"
year: 2014
citekey: JMLR:v15:srivastava14a
links:
  pdf: "http://jmlr.org/papers/v15/srivastava14a.html"
  code: ""
  project: ""
pdf_annotated: "/papers/2025/srivastava14a.pdf"
tags:
  areas: [generalization, robustness]
  methods: [Dropout]
  tasks: [regularization]
  status: "deep-read"
replication:
  repo: ""
  data: ""
  env: ""
---

## TL;DR

Désactiver aléatoirement des neurones pendant l'entraînement (avec probabilité p) empêche le co-adaptation et réduit drastiquement l'overfitting, agissant comme un ensemble exponentiel de réseaux.

## Contexte

Les réseaux profonds avec beaucoup de paramètres overfittent facilement sur de petits datasets ; les méthodes de régularisation traditionnelles (L2, early stopping) ont des limites.

## Idées clés

Dropout simule l'entraînement d'un ensemble exponentiel de réseaux "thinned" partageant les poids ; empêche les neurones de trop dépendre les uns des autres, forçant des représentations robustes.

## Méthode

Pendant l'entraînement : chaque neurone est gardé avec probabilité p (typiquement 0.5 pour hidden, 0.8 pour input) ; à l'inférence : multiplier les poids par p (approximation de l'ensemble).

## Résultats

Réduction significative de l'erreur test sur MNIST, CIFAR-10, ImageNet, Street View House Numbers ; améliore speech recognition, document classification ; effet similaire à model averaging.

## Limites

Augmente le temps d'entraînement (~2-3x) car convergence plus lente ; choix de p peut être délicat selon la tâche ; moins efficace avec batch normalization moderne.

## Liens utiles

- **PDF annoté**: [Dropout - PDF annoté](../../papers/2025/srivastava14a.pdf)
- **Article**: [Dropout: A Simple Way to Prevent Neural Networks from Overfitting  (PDF)](https://www.jmlr.org/papers/volume15/srivastava14a/srivastava14a.pdf)
- **ArXiv**: [Dropout: A Simple Way to Prevent Neural Networks from Overfitting ](https://www.jmlr.org/papers/v15/srivastava14a.html)

## Notes perso
