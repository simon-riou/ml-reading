---
title: "Training language models to follow instructions with human feedback"
authors: ["Long Ouyang", "Jeff Wu", "Xu Jiang", "Diogo Almeida", "Carroll L. Wainwright", "Pamela Mishkin", "Chong Zhang", "Sandhini Agarwal", "Katarina Slama", "Alex Ray", "John Schulman", "Jacob Hilton", "Fraser Kelton", "Luke Miller", "Maddie Simens", "Amanda Askell", "Peter Welinder", "Paul Christiano", "Jan Leike", "Ryan Lowe"]
venue: "NeurIPS"
year: 2022
citekey: ouyang2022traininglanguagemodelsfollow
links:
  pdf: "https://arxiv.org/abs/2203.02155"
  code: ""
  project: ""
pdf_annotated: "../../papers/2026/instructgpt.pdf"
tags:
  areas: ["NLP", "Alignment", "Large Language Models"]
  methods: ["RLHF", "PPO", "Reward Modeling", "Supervised Fine-tuning"]
  tasks: ["Instruction Following", "Text Generation"]
  status: "deep-read"
replication:
  repo: ""
  data: "OpenAI API prompts"
  env: ""
---

## TL;DR

InstructGPT utilise le **Reinforcement Learning from Human Feedback (RLHF)** pour aligner les LLMs avec les intentions des utilisateurs. Pipeline en 3 étapes : (1) **Supervised Fine-Tuning (SFT)** sur des démonstrations humaines, (2) entraînement d'un **Reward Model** sur des comparaisons humaines, (3) optimisation avec **PPO** contre ce reward model. Résultat clé : **InstructGPT 1.3B est préféré à GPT-3 175B** par les labelers humains malgré 100x moins de paramètres. Le papier introduit le paradigme d'alignement qui sera utilisé pour ChatGPT.

## Contexte

**Problème central** : Les LLMs pré-entraînés (GPT-3) optimisent la prédiction du prochain token, pas l'utilité pour l'utilisateur. Ils peuvent générer du contenu toxique, faux, ou ignorer les instructions.

**Objectif** : Créer des modèles qui sont **helpful** (suivent les instructions), **harmless** (évitent le contenu toxique) et **honest** (ne fabriquent pas d'informations).

**Travaux antérieurs** :
- RLHF utilisé pour la summarization (Stiennon et al., 2020)
- Le concept d'alignement discuté théoriquement mais peu appliqué aux LLMs à grande échelle

## Idées clés

1. **RLHF pour l'alignement** : Utiliser les préférences humaines plutôt que des métriques automatiques comme signal d'entraînement.

2. **Pipeline en 3 étapes** : SFT → Reward Model → PPO. Chaque étape raffine le comportement du modèle.

3. **Les comparaisons sont plus faciles que les démonstrations** : Il est plus simple pour un humain de dire "A est meilleur que B" que de rédiger la réponse parfaite.

4. **Petit modèle aligné > Grand modèle non-aligné** : L'alignement apporte plus de valeur que la taille brute pour les tâches d'instruction following.

5. **"Alignment tax" minimal** : InstructGPT ne dégrade pas significativement les performances sur les benchmarks NLP classiques.

## Méthode

### Étape 1 : Supervised Fine-Tuning (SFT)

- Collecte de prompts via l'API OpenAI + prompts écrits par les labelers
- Labelers écrivent les réponses idéales (13K démonstrations)
- Fine-tuning de GPT-3 sur ces paires (prompt, réponse)

### Étape 2 : Reward Model (RM)

- Pour chaque prompt, générer K réponses avec le modèle SFT
- Labelers classent les K réponses de la meilleure à la pire
- Entraîner un modèle à prédire la préférence humaine (33K comparaisons)
- Loss : pairwise ranking (Bradley-Terry model)

### Étape 3 : PPO (Proximal Policy Optimization)

- Optimiser le modèle SFT pour maximiser le reward prédit par le RM
- Contrainte KL pour ne pas trop s'éloigner du modèle initial (éviter le reward hacking)
- Objectif : $\text{maximize } \mathbb{E}[R(x,y)] - \beta \cdot KL(\pi || \pi_{SFT})$

### Tailles de modèles

- 1.3B, 6B, 175B paramètres
- Reward Model : 6B paramètres

## Résultats

### Préférences humaines

| Comparaison | Préférence pour InstructGPT |
|-------------|----------------------------|
| InstructGPT 1.3B vs GPT-3 175B | **71%** |
| InstructGPT 175B vs GPT-3 175B | **85%** |

### Évaluations spécifiques

- **Truthfulness** (TruthfulQA) : InstructGPT génère 2x moins d'hallucinations
- **Toxicité** : Réduction de 25% des outputs toxiques vs GPT-3
- **Instruction following** : Amélioration significative sur les prompts API réels

### Régression sur benchmarks NLP

- Légère baisse sur certains benchmarks académiques (LAMBADA, HellaSwag)
- Solution : mélanger données de pré-entraînement pendant PPO ("PPO-ptx")

## Limites

1. **Dépendance aux labelers** : Les préférences reflètent celles d'un groupe spécifique de contractuels, pas de l'humanité entière.

2. **Reward hacking** : Le modèle peut apprendre à exploiter le reward model plutôt qu'à être vraiment utile.

3. **Sycophancy** : Le modèle peut apprendre à dire ce que l'utilisateur veut entendre plutôt que la vérité.

4. **Coût** : Nécessite beaucoup d'annotations humaines de qualité.

5. **Alignement superficiel** : Change le comportement mais pas la "compréhension" sous-jacente du modèle.

## Liens utiles

- **PDF annoté**: [PDF annoté](../../papers/2026/instructgpt.pdf)
- **Article**: [Training language models to follow instructions with human feedback (arXiv)](https://arxiv.org/abs/2203.02155)
- **OpenAI Blog**: [Aligning Language Models to Follow Instructions](https://openai.com/research/instruction-following)
