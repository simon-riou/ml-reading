---
title: "LLaMA: Open and Efficient Foundation Language Models"
authors: ["Hugo Touvron", "Thibaut Lavril", "Gautier Izacard", "Xavier Martinet", "Marie-Anne Lachaux", "Timothee Lacroix", "Baptiste Rozière", "Naman Goyal", "Eric Hambro", "Faisal Azhar", "Aurelien Rodriguez", "Armand Joulin", "Edouard Grave", "Guillaume Lample"]
venue: "arXiv"
year: 2023
citekey: touvron2023llama
links:
  pdf: "https://arxiv.org/abs/2302.13971"
  code: "https://github.com/facebookresearch/llama"
  project: ""
pdf_annotated: "../../papers/2026/LLaMA.pdf"
tags:
  areas: ["NLP", "Language Modeling", "Open Source"]
  methods: ["Transformer", "RMSNorm", "SwiGLU", "RoPE", "Scaling Laws"]
  tasks: ["Language Modeling", "Question Answering", "Common Sense Reasoning", "Code Generation"]
  status: "deep-read"
replication:
  repo: "https://github.com/facebookresearch/llama"
  data: "CommonCrawl, C4, GitHub, Wikipedia, Books, ArXiv, StackExchange"
  env: ""
---

## TL;DR

LLaMA est une famille de LLMs (7B-65B paramètres) entraînés **uniquement sur des données publiques** qui atteint des performances SOTA. Résultat clé : **LLaMA-13B surpasse GPT-3 (175B) sur la plupart des benchmarks** tout en étant 10x plus petit. LLaMA-65B est compétitif avec Chinchilla-70B et PaLM-540B. L'insight principal : entraîner des modèles plus petits sur **beaucoup plus de tokens** (1-1.4T) que ce que suggèrent les scaling laws de Chinchilla, car le coût d'inférence compte plus que le coût d'entraînement en production.

## Contexte

**Scaling laws de Chinchilla** : Hoffmann et al. (2022) montrent qu'il faut équilibrer taille du modèle et quantité de données. Mais ces lois optimisent le **budget d'entraînement**, pas le **coût d'inférence**.

**Problème** : Les meilleurs LLMs (GPT-3, PaLM, Chinchilla) utilisent des données propriétaires et ne sont pas accessibles à la recherche.

**Modèles ouverts existants** : OPT, BLOOM, GPT-NeoX existent mais ne rivalisent pas avec PaLM ou Chinchilla.

**Motivation de LLaMA** : Créer des modèles performants, entraînés sur données publiques uniquement, et les rendre accessibles à la communauté de recherche.

## Idées clés

1. **Optimiser pour l'inférence, pas l'entraînement** : Un modèle plus petit entraîné plus longtemps est plus efficace à l'inférence qu'un gros modèle entraîné moins. LLaMA-7B continue à s'améliorer après 1T tokens (vs 200B recommandés par Chinchilla pour 10B).

2. **Données publiques uniquement** : Contrairement à GPT-3/PaLM qui utilisent "Books - 2TB" ou des "conversations de réseaux sociaux" non documentées, LLaMA n'utilise que des sources publiques et documentées.

3. **Scaling au-delà de Chinchilla** : Entraîner sur 1-1.4T tokens même pour des modèles relativement petits (7B-65B), bien au-delà du ratio "optimal" N ≈ D.

## Méthode

### Architecture

Transformer decoder-only avec améliorations récentes :
- **Pre-normalization** (GPT-3) : RMSNorm à l'entrée de chaque sub-layer
- **SwiGLU** (PaLM) : Activation SwiGLU au lieu de ReLU, dimension 2/3 × 4d
- **RoPE** (GPTNeo) : Rotary Positional Embeddings au lieu d'embeddings absolus

| Modèle | Params | Layers | d_model | Heads | Tokens |
|--------|--------|--------|---------|-------|--------|
| LLaMA-7B | 6.7B | 32 | 4096 | 32 | 1.0T |
| LLaMA-13B | 13B | 40 | 5120 | 40 | 1.0T |
| LLaMA-33B | 32.5B | 60 | 6656 | 52 | 1.4T |
| LLaMA-65B | 65.2B | 80 | 8192 | 64 | 1.4T |

### Données d'entraînement (1.4T tokens)

| Source | Proportion |
|--------|------------|
| CommonCrawl (filtré CCNet) | 67% |
| C4 | 15% |
| GitHub | 4.5% |
| Wikipedia (20 langues) | 4.5% |
| Books (Gutenberg + Books3) | 4.5% |
| ArXiv | 2.5% |
| StackExchange | 2% |

Tokenizer : BPE (SentencePiece), nombres splittés en digits individuels.

### Entraînement

- Optimiseur : AdamW (β1=0.9, β2=0.95)
- Cosine LR schedule, warmup 2000 steps
- Weight decay 0.1, gradient clipping 1.0
- LLaMA-65B : 21 jours sur 2048 A100-80GB (~380 tokens/sec/GPU)

## Résultats

### Common Sense Reasoning (Zero-shot)

| Modèle | BoolQ | PIQA | HellaSwag | WinoGrande | ARC-c |
|--------|-------|------|-----------|------------|-------|
| GPT-3 175B | 60.5 | 81.0 | 78.9 | 70.2 | 51.4 |
| Chinchilla 70B | 83.7 | 81.8 | 80.8 | 74.9 | - |
| **LLaMA-13B** | 78.1 | 80.1 | 79.2 | 73.0 | 52.7 |
| **LLaMA-65B** | 85.3 | 82.8 | 84.2 | 77.0 | 56.0 |

**LLaMA-13B surpasse GPT-3 175B** sur la majorité des benchmarks.

### Question Answering (Closed-book)

| Modèle | NaturalQuestions (5-shot) | TriviaQA (5-shot) |
|--------|---------------------------|-------------------|
| GPT-3 175B | - | - |
| Chinchilla 70B | 31.5 | 64.1 |
| LLaMA-13B | 28.1 | 63.1 |
| LLaMA-65B | **35.0** | **72.6** |

### MMLU (5-shot)

| Modèle | Average |
|--------|---------|
| GPT-3 175B | 43.9 |
| Chinchilla 70B | 67.5 |
| PaLM 540B | 69.3 |
| LLaMA-65B | 63.4 |
| **LLaMA-I 65B** (instruction-tuned) | **68.9** |

LLaMA-65B légèrement en dessous de Chinchilla/PaLM sur MMLU (moins de livres/papers académiques dans les données).

### Code Generation (pass@1)

| Modèle | HumanEval | MBPP |
|--------|-----------|------|
| PaLM 62B | 15.9 | 21.4 |
| PaLM 540B | 26.2 | 36.8 |
| LLaMA-13B | 15.8 | 22.0 |
| LLaMA-65B | **23.7** | **37.7** |

## Limites

1. **MMLU en dessous de Chinchilla/PaLM** : Moins de livres/papiers académiques (177GB vs 2TB pour Gopher).

2. **Biais et toxicité** : Scores de toxicité augmentent avec la taille du modèle. Biais de genre/religion présents (CommonCrawl).

3. **Pas d'instruction tuning** : Le modèle de base n'est pas aligné pour suivre des instructions (contrairement à InstructGPT/ChatGPT).

4. **Licence restrictive** : Initialement distribué sous licence recherche uniquement, pas pour usage commercial.

## Liens utiles

- **PDF annoté**: [PDF annoté](../../papers/2026/LLaMA.pdf)
- **Article**: [LLaMA: Open and Efficient Foundation Language Models (arXiv)](https://arxiv.org/abs/2302.13971)
- **Code**: [GitHub](https://github.com/facebookresearch/llama)
