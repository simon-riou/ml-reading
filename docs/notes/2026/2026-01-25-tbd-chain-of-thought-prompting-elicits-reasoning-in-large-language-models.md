---
title: "Chain-of-Thought Prompting Elicits Reasoning in Large Language Models"
authors: ["Jason Wei", "Xuezhi Wang", "Dale Schuurmans", "Maarten Bosma", "Brian Ichter", "Fei Xia", "Ed Chi", "Quoc Le", "Denny Zhou"]
venue: "NeurIPS"
year: 2022
citekey: wei2022chainofthoughtpromptingelicitsreasoning
links:
  pdf: "https://arxiv.org/abs/2201.11903"
  code: ""
  project: ""
pdf_annotated: "../../papers/2026/chain_of_thought.pdf"
tags:
  areas: ["NLP", "Prompting", "Reasoning", "Large Language Models"]
  methods: ["Chain-of-Thought", "Few-shot Prompting", "In-context Learning"]
  tasks: ["Arithmetic Reasoning", "Commonsense Reasoning", "Symbolic Reasoning", "Math Word Problems"]
  status: "deep-read"
replication:
  repo: ""
  data: "GSM8K, SVAMP, ASDiv, AQuA, MAWPS, StrategyQA, CommonsenseQA, Sports Understanding, SayCan, Date Understanding, Last Letter Concatenation, Coin Flip"
  env: ""
---

## TL;DR

Ce papier introduit le **Chain-of-Thought (CoT) prompting**, une technique simple mais révolutionnaire qui améliore drastiquement les capacités de raisonnement des LLMs. L'idée clé : au lieu de demander directement la réponse, on fournit des exemples few-shot incluant des **étapes de raisonnement intermédiaires**. Sur le benchmark GSM8K (problèmes mathématiques), PaLM 540B passe de **17.9% (standard prompting)** à **56.9% (CoT)**, et atteint **74.4%** avec self-consistency, surpassant le SOTA fine-tuné (55%). Les résultats montrent que le CoT est une **capacité émergente** : il n'aide pas les petits modèles (<10B) mais produit des gains spectaculaires aux grandes échelles. Le papier établit que les LLMs peuvent "raisonner" de manière compositionnelle quand on leur fournit le bon format de démonstration, ouvrant la voie au prompting moderne et à l'ère de l'"augmented LLM".

## Contexte

En 2022, malgré les succès impressionnants des LLMs (GPT-3, PaLM), plusieurs limitations persistent :

**Échec sur le raisonnement arithmétique** : Les LLMs, même très grands, échouent sur des problèmes mathématiques simples de niveau école primaire. GPT-3 175B atteint seulement ~35% sur GSM8K, bien en dessous des performances humaines (~95%). Le raisonnement multi-étapes semble hors de portée.

**Gap entre scaling et raisonnement** : Les scaling laws montrent que plus de compute améliore la perplexité, mais les capacités de raisonnement ne semblent pas suivre la même courbe. Le raisonnement compositionnel reste un défi fondamental.

**Fine-tuning coûteux** : Les approches SOTA sur les benchmarks de raisonnement (comme les "verifiers" ou "calculators") nécessitent un fine-tuning sur des données annotées avec des traces de raisonnement, ce qui est coûteux et ne généralise pas.

**Prompting standard limité** : Le few-shot prompting de GPT-3 fonctionne bien pour des tâches de classification ou QA simple, mais échoue sur les tâches nécessitant plusieurs étapes de raisonnement.

**Travaux connexes** :
- **Scratchpads** (Nye et al., 2021) : Montrent que l'entraînement avec des traces de calcul intermédiaires améliore les performances sur l'arithmétique.
- **Rationales** : Des travaux montrent que générer des explications avant les réponses peut aider, mais nécessitent du fine-tuning.
- **In-context learning** : GPT-3 a établi que les LLMs peuvent apprendre de nouvelles tâches via des exemples, mais sans exploration du raisonnement multi-étapes.

**Question centrale** : Peut-on éliciter des capacités de raisonnement latentes dans les LLMs simplement en modifiant le format des prompts, sans aucun fine-tuning ?

## Idées clés

1. **Chain-of-Thought comme format de prompting** : Au lieu de `[Question] → [Réponse]`, utiliser `[Question] → [Étapes de raisonnement] → [Réponse]`. Les exemples few-shot incluent explicitement le raisonnement intermédiaire, incitant le modèle à générer son propre raisonnement avant de répondre.

2. **Aucun fine-tuning requis** : Le CoT est purement une technique de prompting. On utilise le même modèle pré-entraîné, en changeant seulement les exemples fournis dans le contexte. Cela permet une adaptation instantanée à de nouvelles tâches sans collecte de données.

3. **Décomposition naturelle des problèmes** : Le CoT force le modèle à décomposer un problème complexe en sous-problèmes plus simples, chacun pouvant être résolu avec les capacités existantes du modèle. C'est analogue au raisonnement humain étape par étape.

4. **Capacité émergente avec le scale** : Le CoT n'améliore pas (voire dégrade) les performances des modèles <10B paramètres. Les gains n'apparaissent qu'aux grandes échelles (~100B+), suggérant que le CoT exploite des capacités latentes qui n'émergent qu'avec le scaling.

5. **Généralisation robuste** : Les mêmes 8 exemples CoT fonctionnent sur de multiples benchmarks de raisonnement arithmétique. Le modèle apprend le "format" du raisonnement, pas des solutions spécifiques.

6. **Interprétabilité accrue** : Le CoT produit une trace explicite du raisonnement, permettant de diagnostiquer où le modèle se trompe. On peut identifier si l'erreur vient d'une étape de calcul, d'une mauvaise compréhension, ou d'une erreur logique.

7. **Applicabilité large** : Le CoT s'applique à trois types de raisonnement : arithmétique (math word problems), commonsense (StrategyQA), et symbolique (last letter concatenation). La technique est générale, pas spécifique à un domaine.

8. **Complémentarité avec self-consistency** : Le CoT peut être combiné avec le sampling multiple et le vote majoritaire (self-consistency) pour des gains supplémentaires significatifs.

## Méthode

### Chain-of-Thought Prompting

**Format standard (baseline)** :
```
Q: Roger a 5 balles de tennis. Il achète 2 boîtes de 3 balles chacune.
Combien de balles a-t-il maintenant?
A: 11

Q: [nouvelle question]
A:
```

**Format Chain-of-Thought** :
```
Q: Roger a 5 balles de tennis. Il achète 2 boîtes de 3 balles chacune.
Combien de balles a-t-il maintenant?
A: Roger a commencé avec 5 balles. 2 boîtes de 3 balles font 2 * 3 = 6 balles.
5 + 6 = 11. La réponse est 11.

Q: [nouvelle question]
A:
```

**Caractéristiques clés** :
- Les chains sont écrites manuellement par les auteurs (8 exemplaires pour math)
- Langage naturel, pas de format rigide
- Chaque chain se termine par "La réponse est [X]" pour extraction facile
- Les mêmes chains sont utilisées pour tous les benchmarks d'un domaine

### Modèles évalués

**5 LLMs de différentes échelles** :
- GPT-3 : 350M, 1.3B, 6.7B, 175B (davinci)
- LaMDA : 422M, 2B, 8B, 68B, 137B
- PaLM : 8B, 62B, 540B
- UL2 : 20B
- Codex (code-davinci-002)

### Benchmarks

**Raisonnement arithmétique (5 benchmarks)** :
- **GSM8K** : 8.5K problèmes de math niveau école primaire (7-8 étapes)
- **SVAMP** : Variations sur les word problems classiques
- **ASDiv** : Problèmes de math diversifiés
- **AQuA** : Problèmes algébriques (choix multiple)
- **MAWPS** : Suite de benchmarks arithmétiques

**Raisonnement commonsense (5 benchmarks)** :
- **CommonsenseQA** : QA nécessitant connaissance du monde
- **StrategyQA** : Questions nécessitant stratégie de décomposition
- **Sports Understanding** : Compréhension de plausibilité sportive
- **SayCan** : Planification robotique
- **Date Understanding** : Raisonnement sur les dates

**Raisonnement symbolique (2 tâches)** :
- **Last Letter Concatenation** : Concaténer les dernières lettres de N mots
- **Coin Flip** : Suivre l'état d'une pièce après N flips

### Protocole expérimental

- **Few-shot** : 8 exemplaires pour math, 4-6 pour autres tâches
- **Decoding** : Greedy (sauf pour self-consistency)
- **Évaluation** : Exact match après extraction de la réponse finale
- **Pas de fine-tuning** : Tous les résultats sont zero-gradient

## Résultats

### Raisonnement arithmétique (GSM8K)

| Modèle | Standard | Chain-of-Thought | Gain |
|--------|----------|------------------|------|
| GPT-3 175B | 17.9% | 35.8% | +17.9% |
| LaMDA 137B | 17.1% | 27.7% | +10.6% |
| PaLM 62B | 33.0% | 44.2% | +11.2% |
| **PaLM 540B** | 17.9% | **56.9%** | **+39.0%** |
| Codex | 19.7% | 63.1% | +43.4% |

**Avec self-consistency (PaLM 540B)** : **74.4%** (40 paths)

**Comparaison SOTA** :
- Fine-tuned GPT-3 + verifier : 55%
- **CoT + self-consistency : 74.4%** (nouveau SOTA)

### Capacité émergente

| Taille modèle | GSM8K Standard | GSM8K CoT | Différence |
|---------------|----------------|-----------|------------|
| PaLM 8B | 4.6% | 2.9% | -1.7% |
| PaLM 62B | 33.0% | 44.2% | +11.2% |
| PaLM 540B | 17.9% | 56.9% | +39.0% |

**Observation clé** : Le CoT **dégrade** les performances des petits modèles mais produit des gains **massifs** pour les très grands modèles. C'est une capacité émergente liée au scale.

### Raisonnement commonsense

| Benchmark | PaLM Standard | PaLM CoT | SOTA fine-tuné |
|-----------|---------------|----------|----------------|
| StrategyQA | 73.9% | **79.0%** | 69.0% |
| Sports | 95.4% | **95.9%** | - |
| CommonsenseQA | 79.0% | **79.9%** | 79.0% |
| Date | 58.0% | **67.5%** | - |

Le CoT surpasse le SOTA fine-tuné sur StrategyQA et égale/dépasse sur les autres.

### Raisonnement symbolique

**Last Letter Concatenation** (4 mots, OOD) :
- Standard : 8.6%
- **CoT : 74.0%**

**Coin Flip** (4 flips, OOD) :
- Standard : 50.0%
- **CoT : 98.0%**

Le CoT permet une généralisation OOD spectaculaire sur les tâches symboliques.

### Analyse des erreurs (GSM8K, PaLM 62B)

| Type d'erreur | Pourcentage |
|---------------|-------------|
| Erreur de calcul | 8% |
| Symbole manquant | 16% |
| Une étape manquante | 22% |
| Erreur sémantique | 54% |

La majorité des erreurs sont des erreurs de compréhension sémantique, pas de calcul pur.

### Ablations

**Importance de chaque composante** :
| Variante | GSM8K |
|----------|-------|
| Équation seulement (pas de NL) | 41.3% |
| Variables après équation | 43.2% |
| **Chain-of-Thought complet** | **49.6%** |

Le langage naturel dans les chains améliore significativement les performances.

**Robustesse aux exemples** :
- Différents annotateurs → résultats similaires
- Différents styles de chains → résultats similaires
- Le format importe plus que le contenu exact

## Limites

### Limites méthodologiques

1. **Capacité émergente non comprise** : On ne sait pas pourquoi le CoT ne fonctionne qu'aux grandes échelles. Quelles capacités spécifiques doivent émerger pour que le CoT soit efficace ?

2. **Pas de garantie de correction** : Le modèle peut générer des chains plausibles mais incorrectes. Un raisonnement convaincant ne garantit pas la bonne réponse.

3. **Coût d'inférence accru** : Générer les étapes intermédiaires augmente significativement le nombre de tokens générés, donc le coût et la latence.

4. **Prompts manuels** : Les exemplaires CoT sont écrits manuellement. Trouver les bons exemplaires peut nécessiter de l'expérimentation.

5. **Évaluation sur benchmarks simples** : GSM8K reste des problèmes de niveau école primaire. Les performances sur des mathématiques plus avancées ne sont pas évaluées en détail.

### Limites sur le raisonnement

6. **Pas de vrai raisonnement symbolique** : Le modèle peut faire des erreurs de calcul triviales. Le "raisonnement" reste probabiliste, pas symbolique.

7. **Corrélation vs causalité** : On ne sait pas si le modèle "raisonne vraiment" ou s'il produit du texte qui ressemble à du raisonnement et arrive à la bonne réponse par corrélation.

8. **Erreurs systématiques** : Certains types d'erreurs (multi-step, erreurs sémantiques) persistent même avec le CoT.

9. **Pas de révision** : Le modèle génère séquentiellement sans pouvoir revenir en arrière corriger une erreur dans son raisonnement.

### Limites pratiques

10. **Dépendance au scale** : Nécessite des modèles >100B paramètres pour des gains significatifs. Inaccessible pour la plupart des applications.

11. **Sensibilité au prompt** : Les performances peuvent varier selon la formulation exacte des exemplaires.

12. **Pas de garantie de généralisation** : Les résultats sur les benchmarks peuvent ne pas transférer à des cas réels plus complexes.

## Liens utiles

- **PDF annoté**: [PDF annoté](../../papers/2026/chain_of_thought.pdf)
- **Article**: [Chain-of-Thought Prompting Elicits Reasoning in Large Language Models (arXiv)](https://arxiv.org/abs/2201.11903)
- **NeurIPS 2022**: [Paper](https://openreview.net/forum?id=_VjQlMeSB_J)
- **Google AI Blog**: [Language Models Perform Reasoning via Chain of Thought](https://ai.googleblog.com/2022/05/language-models-perform-reasoning-via.html)

## Notes perso

Le papier est important car il formalise le passage d'une ere ou on cherche a finetuner a celle ou on comprend que les Fundations models ont des capacités émergentes (des capacités qui arrivent toutes seuls grace a la scale)
