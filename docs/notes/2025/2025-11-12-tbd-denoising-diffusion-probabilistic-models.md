---
title: "Denoising Diffusion Probabilistic Models"
authors: ["Jonathan Ho", "Ajay Jain", "Pieter Abbeel"]
venue: "NeurIPS 2020"
year: 2020
citekey: ho2020denoisingdiffusionprobabilisticmodels
links:
  pdf: "https://arxiv.org/pdf/2006.11239"
  code: "https://github.com/hojonathanho/diffusion"
  project: ""
pdf_annotated: "/papers/2025/DDPM.pdf"
tags:
  areas: ["deep-learning", "computer-vision"]
  methods: ["diffusion-models", "denoising"]
  tasks: ["image-generation"]
  status: "replicated"
replication:
  repo: ""
  data: ""
  env: ""
---

## TL;DR

Les modèles de diffusion probabiliste génèrent des images de haute qualité en apprenant à inverser progressivement un processus de diffusion qui ajoute du bruit gaussien, surpassant les GANs en qualité d'échantillons.

## Contexte

Les modèles génératifs profonds (GANs, VAEs, flow models) dominent la génération d'images, mais les GANs souffrent d'instabilité d'entraînement et de mode collapse, tandis que les VAEs produisent des images floues.

## Idées clés

Un processus de diffusion forward ajoute progressivement du bruit gaussien sur T étapes jusqu'à détruire complètement l'image ; le modèle apprend le processus reverse pour débruiter progressivement du bruit pur en une image cohérente ; formulation équivalente à un score matching denoising avec un objectif simplifié.

## Méthode

Forward process : q(x_t|x_{t-1}) = N(√(1-β_t) x_{t-1}, β_t I) avec schedule de variance β_t fixe ;
reverse process : p_θ(x_{t-1}|x_t) paramétré par un U-Net qui prédit le bruit ε ; 
objectif d'entraînement simplifié L_simple = E[||ε - ε_θ(x_t, t)||²] ;
échantillonnage itératif sur T~1000 étapes.

## Résultats

FID de 3.17 sur CIFAR-10 (256x256 conditionnelles), surpassant les GANs state-of-the-art ; qualité perceptuelle exceptionnelle sur CelebA-HQ et LSUN ; diversité d'échantillons supérieure aux GANs sans mode collapse ; inception score compétitif.

## Limites

Échantillonnage très lent (~1000 forward passes vs 1 pour GAN) ; coût computationnel élevé en inférence ; nécessite beaucoup de mémoire pour l'entraînement ; difficulté à générer des images haute résolution directement.

## Liens utiles

- **Implementation**: [DDPM](https://github.com/simon-riou/ml-replicating/blob/master/models/DDPM.py)
- **PDF annoté**: [DDPM - PDF annoté](../../papers/2025/DDPM.pdf)
- **Article**: [Denoising Diffusion Probabilistic Models (PDF)](https://arxiv.org/pdf/2006.11239)
- **ArXiv**: [https://arxiv.org/abs/2006.11239](https://arxiv.org/abs/2006.11239)
- **Code officiel**: [https://github.com/hojonathanho/diffusion](https://github.com/hojonathanho/diffusion)
- **Paper explainer**: [Diffusion Models: DDPM | Generative AI Animated](https://www.youtube.com/watch?v=EhndHhIvWWw)

## Notes perso
