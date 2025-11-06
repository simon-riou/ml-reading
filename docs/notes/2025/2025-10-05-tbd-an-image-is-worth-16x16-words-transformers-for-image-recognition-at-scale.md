---
title: "AN IMAGE IS WORTH 16X16 WORDS: TRANSFORMERS FOR IMAGE RECOGNITION AT SCALE"
authors: ["Alexey Dosovitskiy"]
venue: "Published as a conference paper at ICLR 2021"
year: 2021
citekey: dosovitskiy2021ViT
links:
  pdf: "https://arxiv.org/pdf/2010.11929"
  code: "https://github.com/google-research/vision_transformer"
  project: ""
pdf_annotated: "/papers/2025/ViT.pdf"
tags:
  areas: ["SOTA", "computer-vision"]
  methods: ["ViT"]
  tasks: ["classification"]
  status: "deep-read"
replication:
  repo: ""
  data: ""
  env: ""
---

## TL;DR

Les Transformers appliqués directement sur des patchs d'images atteignent des performances SOTA en classification, sans avoir besoin de biais inductifs CNN, quand pré-entraînés sur de larges datasets.

## Contexte

Les CNNs dominent la vision par ordinateur grâce à leurs biais inductifs (localité, translation equivariance), mais les Transformers ont révolutionné le NLP.

## Idées clés

Diviser les images en patchs 16x16, les traiter comme des tokens, et utiliser un Transformer pur sans convolutions permet d'atteindre SOTA avec moins de coût computationnel au pré-entraînement.

## Méthode

Images découpées en patchs linéarisés + position embeddings, passés dans un Transformer encoder standard, avec un token [CLS] pour la classification.

## Résultats

ViT-H/14 atteint 88.55% sur ImageNet (pré-entraîné sur JFT-300M), surpassant les CNNs SOTA tout en nécessitant substantiellement moins de ressources pour l'entraînement.

## Limites

Performances inférieures aux CNNs sur datasets de taille moyenne sans pré-entraînement massif ; manque de biais inductifs nécessite énormément de données pour généraliser.

## Liens utiles

- **Article**: [An Image is Worth 16x16 Words: Transformers for Image Recognition at Scale (PDF)](https://arxiv.org/pdf/2010.11929)
- **ArXiv**: [https://arxiv.org/abs/2010.11929](https://arxiv.org/abs/2010.11929)

- **Paper explained**: [Samuel Albanie](https://www.youtube.com/watch?v=vsqKGZT8Qn8)

- **Paper explained**: [Yannic Kilcher](https://www.youtube.com/watch?v=TrdevFK_am4)

## Notes perso

La vidéo de Yannic Kilcher explique le papier, tandis que la vidéo de Samuel Albanie explique en plus les concepts généraux (transformers, bias factor, ...)