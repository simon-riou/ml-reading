---
title: "LoRA: Low-Rank Adaptation of Large Language Models"
authors: ["Edward Hu", "Yelong Shen", "Phillip Wallis", "Zeyuan Allen-Zhu", "Yuanzhi Li", "Shean Wang", "Lu Wang", "Weizhu Chen"]
venue: "ICLR"
year: 2022
citekey: hu2022lora
links:
  pdf: "https://arxiv.org/abs/2106.09685"
  code: "https://github.com/microsoft/LoRA"
  project: ""
pdf_annotated: "../../papers/2026/LoRA.pdf"
tags:
  areas: ["NLP", "Parameter-Efficient Fine-Tuning", "Large Language Models"]
  methods: ["Low-Rank Adaptation", "Transfer Learning"]
  tasks: ["Fine-Tuning", "NLU", "NLG"]
  status: "deep-read"
replication:
  repo: "https://github.com/microsoft/LoRA"
  data: "GLUE, E2E, WikiSQL, SAMSum"
  env: ""
---

## TL;DR

LoRA gèle les poids pré-entraînés et injecte des **matrices de décomposition low-rank** (A et B) dans les couches du Transformer. Au lieu d'entraîner W directement, on entraîne ΔW = BA où B ∈ ℝ^(d×r) et A ∈ ℝ^(r×k) avec r << d. Résultat clé : **10,000x moins de paramètres entraînables**, **3x moins de VRAM**, et **aucune latence d'inférence** (on peut fusionner W + BA). Performances égales ou supérieures au fine-tuning complet sur GPT-3 175B.

## Contexte

**Problème** : Le fine-tuning complet de LLMs géants (GPT-3 175B) est prohibitif :
- Stocker un checkpoint de 350GB par tâche
- Mémoire GPU pour les optimizer states (Adam stocke 2x les paramètres)
- Déployer plusieurs modèles fine-tunés = coûteux

**Alternatives existantes** :
- **Adapters** (Houlsby 2019) : ajoutent de la latence d'inférence (layers séquentiels)
- **Prefix-tuning** (Li & Liang 2021) : difficile à optimiser, réduit la longueur de séquence utilisable

**Hypothèse clé** : Les updates de poids pendant l'adaptation ont un faible "rang intrinsèque" (basé sur Aghajanyan 2020 qui montre que les LLMs pré-entraînés résident sur une dimension intrinsèque basse).

## Idées clés

1. **Low-rank update** : ΔW = BA avec rang r << min(d, k). Pour GPT-3 175B, r = 1 ou 2 suffit même quand d = 12,288.

2. **Pas de latence à l'inférence** : On peut fusionner W' = W₀ + BA au déploiement. Pour changer de tâche : soustraire BA, ajouter B'A'.

3. **Initialisation** : A ~ N(0, σ²), B = 0. Donc ΔW = 0 au début de l'entraînement.

4. **Quelles matrices adapter ?** : Wq et Wv donnent les meilleurs résultats. Adapter plusieurs matrices avec r petit > une seule avec r grand.

5. **Scaling factor** : ΔWx est multiplié par α/r pour stabiliser les hyperparamètres quand r varie.

## Méthode

**Forward pass modifié** :
```
h = W₀x + ΔWx = W₀x + BAx
```

**Application au Transformer** :
- LoRA sur Wq, Wk, Wv, Wo (attention)
- MLP gelé dans les expériences

**Budget paramètres** : |Θ| = 2 × L × d_model × r (pour L layers)

| Modèle | Params totaux | Params LoRA (r=4, Wq+Wv) |
|--------|---------------|--------------------------|
| GPT-3 175B | 175B | 4.7M (0.003%) |

## Résultats

### GPT-3 175B

| Méthode | # Params | WikiSQL | MNLI-m | SAMSum |
|---------|----------|---------|--------|--------|
| Fine-Tune | 175B | 73.8 | 89.5 | 52.0/28.0/44.5 |
| Adapter | 7.1M | 71.9 | 89.8 | 53.0/28.9/44.8 |
| **LoRA** | **4.7M** | **73.4** | **91.7** | **53.8/29.8/45.9** |

LoRA **surpasse le fine-tuning complet** avec 0.003% des paramètres.

### Effet du rang r

r = 1 ou 2 suffit sur la plupart des tâches. Augmenter r n'améliore pas significativement les performances → ΔW a vraiment un rang intrinsèque très bas.

### Latence d'inférence

| Config | Fine-Tune/LoRA | Adapter |
|--------|----------------|---------|
| Batch=1, Seq=128 | 19.8ms | 25.8ms (+30%) |

## Limites

1. **Batching multi-tâches** : Difficile de batcher des inputs pour différentes tâches (différents A, B) si on fusionne les poids.

2. **Choix des matrices** : Heuristique (Wq, Wv). Pas de méthode principielle pour sélectionner où appliquer LoRA.

3. **Rank optimal** : Dépend de la tâche. r = 1 marche pour GPT-3 sur MNLI mais pas nécessairement pour des tâches très différentes du pré-entraînement.

## Liens utiles

- **PDF annoté**: [PDF annoté](../../papers/2026/LoRA.pdf)
- **Article**: [LoRA: Low-Rank Adaptation of Large Language Models (arXiv)](https://arxiv.org/abs/2106.09685)
- **Code**: [GitHub](https://github.com/microsoft/LoRA)
