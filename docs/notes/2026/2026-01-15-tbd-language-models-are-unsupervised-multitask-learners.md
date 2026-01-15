---
title: "Language Models are Unsupervised Multitask Learners"
authors: ["Alec Radford", "Jeffrey Wu", "Rewon Child", "David Luan", "Dario Amodei", "Ilya Sutskever"]
venue: "OpenAI"
year: 2019
citekey: Radford2019LanguageMA
links:
  pdf: "https://cdn.openai.com/better-language-models/language_models_are_unsupervised_multitask_learners.pdf"
  code: "https://github.com/openai/gpt-2"
  project: ""
pdf_annotated: "../../papers/2026/gpt2.pdf"
tags:
  areas: ["NLP", "Language Modeling", "Transfer Learning", "Zero-shot Learning"]
  methods: ["GPT-2", "Transformer Decoder", "Byte Pair Encoding", "Unsupervised Multitask Learning"]
  tasks: ["Language Modeling", "Reading Comprehension", "Machine Translation", "Summarization", "Question Answering"]
  status: "deep-read"
replication:
  repo: "https://github.com/openai/gpt-2"
  data: "WebText"
  env: ""
---

## TL;DR

GPT-2 démontre que les modèles de langage, lorsqu'ils sont entraînés sur un corpus suffisamment large et divers, commencent à apprendre des tâches NLP **sans supervision explicite** (zero-shot). Le modèle est un **Transformer decoder** scalé à **1.5 milliard de paramètres** (10x GPT-1), entraîné sur **WebText** (40GB de texte web filtré par qualité via Reddit karma ≥3). L'innovation clé est le concept d'**apprentissage multitâche non supervisé** : en modélisant p(output|input, task) via le langage naturel, le modèle apprend implicitement à effectuer diverses tâches pour mieux prédire le texte. GPT-2 atteint le **state-of-the-art zero-shot sur 7/8 benchmarks de language modeling** (perplexité 8.6 sur LAMBADA, 18.3 sur WikiText-103). Sur des tâches downstream sans fine-tuning : 55 F1 sur CoQA (reading comprehension), 63.2% sur LAMBADA (accuracy), 70.7% sur Winograd Schema Challenge (+7% SOTA), et des capacités émergentes en traduction (11.5 BLEU Fr→En) et résumé. Le papier établit que la **capacité du modèle est essentielle** au transfer zero-shot, avec des performances qui s'améliorent log-linéairement avec la taille.

## Contexte

En 2019, le paradigme dominant en NLP est le **pré-entraînement + fine-tuning supervisé** (GPT-1, BERT). Cependant, plusieurs problèmes persistent :

**Systèmes étroits et fragiles** : Les modèles ML actuels excellent sur leurs tâches d'entraînement mais sont sensibles aux changements de distribution de données et de spécification de tâche. Ce sont des "experts étroits" plutôt que des "généralistes compétents".

**Limitations du multitask learning** : Le multitask learning promet d'améliorer la généralisation, mais les efforts les plus ambitieux n'ont entraîné que sur 10-17 paires (dataset, objectif). D'un point de vue méta-learning, chaque paire est un seul exemple d'entraînement - insuffisant pour induire des fonctions généralisables.

**Coût de création de datasets** : Créer et annoter manuellement des datasets pour chaque tâche est coûteux et ne scale pas. Il serait souhaitable de développer des systèmes capables d'apprendre directement à partir de démonstrations naturelles dans le texte.

**Évolution du transfer learning** :
1. Word vectors (Word2Vec, GloVe) → inputs pour architectures task-specific
2. Représentations contextuelles (ELMo) → features de RNNs transférées
3. Self-attention blocks (GPT-1, BERT) → fine-tuning sans architectures task-specific

Mais toutes ces approches nécessitent encore un **entraînement supervisé** pour effectuer une tâche.

**Travaux préliminaires prometteurs** : Des travaux antérieurs (Schwartz 2017, Radford 2017) ont montré que les modèles de langage peuvent effectuer des tâches spécifiques (commonsense reasoning, sentiment analysis) avec peu ou pas de données supervisées.

**Hypothèse centrale du papier** : Un modèle de langage avec une capacité suffisante commencera à **inférer et effectuer les tâches démontrées dans les séquences de langage naturel** afin de mieux les prédire, réalisant ainsi un apprentissage multitâche non supervisé.

## Idées clés

1. **Apprentissage multitâche non supervisé via language modeling** : Au lieu de conditionner explicitement sur la tâche (p(output|input, task) avec encodeurs task-specific), utiliser le langage naturel pour spécifier tâches, inputs et outputs dans une séquence unifiée. Exemples : "(translate to french, english text, french text)" ou "(answer the question, document, question, answer)". Le modèle apprend implicitement ces patterns en optimisant l'objectif de language modeling.

2. **L'objectif non supervisé converge vers l'objectif supervisé** : Formellement, l'objectif supervisé est l'objectif non supervisé évalué sur un sous-ensemble de la séquence. Donc le minimum global de l'objectif non supervisé est aussi le minimum global de l'objectif supervisé. La question devient : peut-on optimiser l'objectif non supervisé jusqu'à convergence en pratique ?

3. **WebText : dataset web filtré par qualité humaine** : Plutôt que d'utiliser Common Crawl (problèmes de qualité) ou des corpus single-domain (news, Wikipedia, livres), créer un nouveau dataset en scrappant tous les liens sortants de Reddit avec karma ≥3 (heuristique de qualité/intérêt). Résultat : 45M liens → 8M documents → 40GB de texte après déduplication et nettoyage. Wikipedia exclu pour éviter les chevauchements avec les benchmarks.

4. **Byte-level BPE pour représentation universelle** : Problème : les LMs word-level nécessitent tokenization et gèrent mal l'OOV; les LMs byte-level sont moins performants. Solution : BPE appliqué aux bytes (vocabulaire de base de 256) avec une modification pour empêcher les merges entre catégories de caractères (sauf espaces). Résultat : vocabulaire de 50,257 tokens capable de représenter n'importe quelle chaîne Unicode sans perte d'information.

5. **Scaling du Transformer decoder** : Architecture GPT-1 avec modifications :
   - Layer normalization déplacée à l'**entrée** de chaque sub-block (pre-activation, comme ResNet)
   - Layer normalization additionnelle après le dernier bloc self-attention
   - Initialisation modifiée : poids des couches résiduelles scalés par 1/√N (N = nombre de couches résiduelles)
   - Contexte étendu de 512 à **1024 tokens**
   - 4 tailles de modèles : 117M (=GPT-1), 345M (=BERT-large), 762M, **1542M** (GPT-2)

6. **Zero-shot task transfer sans modification** : Évaluer les tâches downstream **sans aucune modification de paramètres ou d'architecture**. Les tâches sont induites par le contexte (few-shot prompting pour traduction/QA) ou par des hints textuels ("TL;DR:" pour summarization). Cela teste ce que le modèle a appris durant le pré-entraînement pur.

7. **La capacité du modèle est essentielle** : Les performances zero-shot s'améliorent **log-linéairement** avec la taille du modèle sur toutes les tâches testées. Le plus petit modèle (117M) ne dépasse souvent pas les baselines triviales, tandis que GPT-2 (1.5B) atteint des performances compétitives ou SOTA.

8. **Démonstrations naturelles de tâches dans WebText** : Le dataset contient naturellement des exemples de traduction ("In French: ..."), de QA, de summarization, etc. Le modèle apprend ces patterns pour mieux prédire le texte, acquérant ainsi des capacités de transfer.

## Méthode

### Architecture du modèle

**Transformer Decoder (4 tailles)** :

| Paramètres | Couches | d_model | Contexte |
|------------|---------|---------|----------|
| 117M       | 12      | 768     | 1024     |
| 345M       | 24      | 1024    | 1024     |
| 762M       | 36      | 1280    | 1024     |
| **1542M**  | 48      | 1600    | 1024     |

**Modifications par rapport à GPT-1** :
- **Pre-norm** : Layer normalization à l'entrée de chaque sub-block (vs post-norm)
- **Final layer norm** : LayerNorm additionnelle après le dernier bloc d'attention
- **Scaled initialization** : Poids des couches résiduelles initialisés avec facteur 1/√N
- **Vocabulaire élargi** : 50,257 tokens (vs 40k pour GPT-1)
- **Contexte doublé** : 1024 tokens (vs 512)
- **Batch size augmenté** : 512

### Dataset WebText

**Construction** :
1. Scraper tous les liens sortants de Reddit avec karma ≥3
2. Extraire le texte avec Dragnet + Newspaper
3. Déduplication et nettoyage heuristique
4. Supprimer tous les documents Wikipedia

**Statistiques** :
- 45M liens → 8M documents
- 40GB de texte total
- Cutoff : décembre 2017

**Justification du karma Reddit** : Heuristique pour filtrer le contenu que des humains ont trouvé intéressant, éducatif ou amusant, sans faire d'hypothèses sur les tâches cibles.

### Input Representation (Byte-level BPE)

**Problème avec BPE standard sur bytes** :
- BPE greedy crée des tokens sous-optimaux (ex: "dog.", "dog!", "dog?" comme tokens séparés)
- Allocation inefficace du vocabulaire

**Solution** :
- Empêcher BPE de merger entre catégories de caractères
- Exception pour les espaces (améliore compression)
- Vocabulaire de base : 256 bytes
- Vocabulaire final : 50,257 tokens

**Avantages** :
- Peut assigner une probabilité à n'importe quelle chaîne Unicode
- Évaluation sur n'importe quel benchmark sans pré-processing
- Pas de token `<UNK>` (apparaît 26 fois sur 40B bytes dans WebText)

### Entraînement

- Learning rate tuné manuellement pour minimiser la perplexité sur 5% held-out de WebText
- Tous les modèles **underfittent** encore WebText (perplexité continue de baisser avec plus d'entraînement)
- Pas de détails sur nombre d'epochs, optimiseur, etc. dans le papier

### Évaluation Zero-shot

**Language Modeling** : Calculer log-probabilité du dataset selon WebText LM, diviser par nombre d'unités canoniques (caractères, bytes, mots selon le benchmark). Utilisation de de-tokenizers inversibles pour adapter aux artefacts de preprocessing des benchmarks.

**Reading Comprehension (CoQA)** : Conditionner sur document + historique de conversation + "A:", décoder greedy.

**Summarization (CNN/DM)** : Ajouter "TL;DR:" après l'article, générer 100 tokens avec Top-k sampling (k=2), prendre les 3 premières phrases.

**Translation** : Conditionner sur exemples "english sentence = french sentence", puis "english sentence =", décoder greedy.

**Question Answering (Natural Questions)** : Conditionner sur exemples Q/A, décoder greedy.

## Résultats

### Language Modeling (Table 3)

**Zero-shot SOTA sur 7/8 datasets** :

| Dataset | SOTA précédent | GPT-2 (1.5B) | Amélioration |
|---------|----------------|--------------|--------------|
| **LAMBADA** (PPL) | 99.8 | **8.63** | 91.2 points |
| **LAMBADA** (ACC) | 59.23% | **63.24%** | +4.0% |
| **CBT-CN** (ACC) | 85.7% | **93.30%** | +7.6% |
| **CBT-NE** (ACC) | 82.3% | **89.05%** | +6.8% |
| **WikiText-2** (PPL) | 39.14 | **18.34** | 20.8 points |
| **PTB** (PPL) | 46.54 | **35.76** | 10.8 points |
| **enwik8** (BPB) | 0.99 | **0.93** | 0.06 |
| **text8** (BPC) | 1.08 | **0.98** | 0.10 |
| **WikiText-103** (PPL) | 18.3 | **17.48** | 0.8 points |
| 1BW (PPL) | **21.8** | 42.16 | -20.4 (pire) |

**Observations** :
- Gains massifs sur LAMBADA (dépendances longue-distance) : perplexité 99.8 → 8.6
- Améliorations importantes sur petits datasets (PTB, WikiText-2)
- Seul échec : 1 Billion Word Benchmark (shuffled au niveau des phrases, détruit la structure longue-distance)
- De-tokenizers inversibles apportent 2.5-5 points de perplexité supplémentaires

### Children's Book Test (Figure 2)

- **Common Nouns** : 93.3% (vs 85.7% SOTA, proche de 96% humain)
- **Named Entities** : 89.1% (vs 82.3% SOTA, proche de 92% humain)
- Performance augmente régulièrement avec la taille du modèle

### Winograd Schema Challenge (Figure 3)

- **70.70%** accuracy (vs 63.7% SOTA précédent)
- Amélioration de **+7%** sur le state-of-the-art
- Capacité de raisonnement de sens commun émergente

### Reading Comprehension - CoQA (Section 3.5)

- **55 F1** sur le development set (greedy decoding)
- Égale ou dépasse **3/4 systèmes baseline** sans utiliser les 127k+ exemples d'entraînement
- SOTA supervisé (BERT) : 89 F1 (proche des 89 F1 humains)
- Analyse des erreurs : GPT-2 utilise souvent des heuristiques simples (répondre avec un nom du document pour les questions "who")

### Summarization - CNN/Daily Mail (Table 4)

| Méthode | R-1 | R-2 | R-L | R-AVG |
|---------|-----|-----|-----|-------|
| Bottom-Up Sum (SOTA) | **41.22** | **18.68** | **38.34** | **32.75** |
| Lede-3 | 40.38 | 17.66 | 36.62 | 31.55 |
| GPT-2 TL;DR: | 29.34 | 8.27 | 26.58 | 21.40 |
| Random-3 | 28.78 | 8.63 | 25.52 | 20.98 |
| GPT-2 no hint | 21.58 | 4.03 | 19.47 | 15.03 |

- Performance qualitativement correcte mais quantitativement faible
- Le hint "TL;DR:" apporte **+6.4 R-AVG** vs sans hint → démontre la capacité à invoquer des comportements task-specific via langage naturel
- Problèmes : focus sur contenu récent, confusion de détails spécifiques

### Translation (Section 3.7)

| Direction | GPT-2 | Baseline non supervisé | SOTA non supervisé |
|-----------|-------|------------------------|-------------------|
| En→Fr | 5 BLEU | Word-by-word : 5 BLEU | 33.5 BLEU |
| Fr→En | **11.5 BLEU** | 9-10 BLEU | 33.5 BLEU |

- Surprenant car WebText a été filtré pour exclure les pages non-anglaises
- Seulement ~10MB de français détecté dans WebText (500x moins que corpus MT standard)
- Fr→En meilleur grâce au fort modèle de langage anglais

### Question Answering - Natural Questions (Table 5)

- **4.1%** exact match (vs 1.0% baseline triviale, vs 30-50% systèmes hybrides retrieval+QA)
- Performance **5.3x** meilleure que le plus petit modèle
- Probabilités **bien calibrées** : 63.1% accuracy sur le 1% de questions les plus confiantes
- 23/30 réponses les plus confiantes sont correctes

### Scaling (Figure 1 & 4)

- Performance zero-shot s'améliore **log-linéairement** avec la taille du modèle sur toutes les tâches
- Le plus petit modèle (117M) ≈ baselines triviales
- GPT-2 (1.5B) ≈ performances compétitives/SOTA
- Même GPT-2 underfit encore WebText (train et test perplexity similaires et continuent de baisser)

### Analyse de mémorisation (Section 4 & Figure 5)

**Chevauchement 8-grams avec WebText** :
- Datasets de test : 1-6% overlap avec WebText train (moyenne 3.2%)
- Les datasets ont souvent plus d'overlap avec leur propre train (moyenne 5.9%)
- Exemple problématique : 1BW a 13.2% d'overlap avec son propre train

**Impact sur les résultats** :
- LAMBADA : recalcul sans overlap → 8.6 → 8.7 PPL, 63.2% → 62.9% ACC (négligeable)
- CoQA : ~3 F1 de mieux sur documents déjà dans WebText
- Winograd : seulement 1/273 schémas avec contexte révélant la réponse

**Mémorisation vs généralisation** :
- Échantillons générés ont **moins** d'overlap avec WebText train que les vrais articles test (Figure 5)
- 30%+ des échantillons n'ont aucun overlap
- Suggère que GPT-2 généralise plutôt que mémorise

## Limites

1. **Zero-shot loin d'être utilisable en pratique** : Malgré des résultats prometteurs en recherche, les performances zero-shot de GPT-2 sont "still far from use-able" pour des applications pratiques. Sur summarization et translation, les métriques quantitatives restent faibles même si la sortie est qualitativement correcte.

2. **Architecture unidirectionnelle** : Comme GPT-1, GPT-2 utilise un decoder unidirectionnel (gauche→droite). BERT a démontré que les représentations bidirectionnelles sont plus efficaces pour de nombreuses tâches de compréhension. Le papier reconnaît cette limitation et mentionne que le fine-tuning pourrait ne pas surmonter les "inefficiencies of uni-directional representations".

3. **Pas de fine-tuning exploré** : Le papier se concentre exclusivement sur le zero-shot. Les performances avec fine-tuning (comme BERT sur GLUE) ne sont pas rapportées, rendant difficile la comparaison directe avec les approches pré-entraînement + fine-tuning.

4. **Échec sur 1 Billion Word Benchmark** : GPT-2 sous-performe significativement sur 1BW (42.2 vs 21.8 SOTA). Les auteurs attribuent cela au shuffling au niveau des phrases qui détruit la structure longue-distance, mais cela révèle une dépendance à cette structure.

5. **Heuristiques simples sur certaines tâches** : L'analyse des erreurs sur CoQA montre que GPT-2 utilise souvent des heuristiques retrieval-based (répondre avec un nom du document pour "who questions") plutôt qu'un vrai raisonnement.

6. **Coût computationnel non reporté** : Le papier ne fournit pas de détails sur le coût d'entraînement (GPU-hours, nombre d'epochs, etc.), rendant difficile l'évaluation de la reproductibilité.

7. **WebText non public** : Le dataset WebText n'est pas publié, citant des préoccupations de reproductibilité des expériences de chevauchement. Cela limite la reproductibilité de la recherche.

8. **Sensibilité au prompting** : Les performances dépendent fortement de la formulation du prompt (ex: "TL;DR:" améliore de 6.4 points). Il n'y a pas d'étude systématique de cette sensibilité.

9. **Calibration des probabilités inexploitée** : Bien que les probabilités soient bien calibrées (63% accuracy sur le top 1% confident), cette propriété n'est pas exploitée pour améliorer les performances (ex: abstention sur les prédictions incertaines).

10. **Évaluation limitée sur certaines tâches** : Pas d'évaluation sur GLUE, pas de comparaison avec BERT fine-tuné, pas d'évaluation cross-linguale ou sur des domaines spécialisés.

11. **Risques de génération de texte** : Le papier mentionne brièvement les préoccupations de sécurité (génération de fausses nouvelles) mais ne les analyse pas en profondeur.

12. **Underfitting persistant** : Même le plus grand modèle underfit encore WebText, suggérant que des modèles plus grands et/ou plus d'entraînement pourraient améliorer les résultats, mais aussi que l'approche n'a pas atteint sa limite.

## Liens utiles

- **PDF annoté**: [PDF annoté](../../papers/2026/gpt2.pdf)
- **Article**: [Language Models are Unsupervised Multitask Learners](https://cdn.openai.com/better-language-models/language_models_are_unsupervised_multitask_learners.pdf)
- **Code**: [https://github.com/openai/gpt-2](https://github.com/openai/gpt-2)
- **OpenAI Blog**: [https://openai.com/blog/better-language-models/](https://openai.com/blog/better-language-models/)

## Notes perso

GPT-2 vs GPT-1 : évolution naturelle

- **Même architecture fondamentale** : Transformer decoder avec masked self-attention
- **Scaling massif** : 117M → 1.5B paramètres (10x), contexte 512 → 1024 (2x)
- **Changement de paradigme** : GPT-1 = pré-entraînement + fine-tuning, GPT-2 = zero-shot transfer
- **Nouveau dataset** : BooksCorpus (7k livres) → WebText (8M pages web, 40GB)

Contributions conceptuelles majeures :

1. **"Language models are unsupervised multitask learners"** : L'idée que p(output|input, task) peut être apprise via language modeling est fondamentale. Le langage naturel permet de spécifier n'importe quelle tâche, et un LM suffisamment capable apprendra à les effectuer pour mieux prédire.

2. **Zero-shot comme mesure de pré-entraînement** : Plutôt que de mesurer les représentations via fine-tuning (qui peut masquer les faiblesses), le zero-shot teste directement ce qui a été appris.

3. **Le scaling améliore les capacités émergentes** : Les courbes log-linéaires suggèrent que des modèles plus grands auront des capacités zero-shot encore meilleures - préfigurant GPT-3.

Comparaison avec ELMo et GPT-1 :

| Aspect | ELMo | GPT-1 | GPT-2 |
|--------|------|-------|-------|
| Architecture | biLSTM | Transformer decoder | Transformer decoder |
| Paramètres | 94M | 117M | 1.5B |
| Corpus | 1B Word | BooksCorpus | WebText |
| Utilisation | Features | Fine-tuning | Zero-shot |
| Objectif | biLM | LM unidirectionnel | LM unidirectionnel |

Points intéressants du papier :

- La note de bas de page humoristique : "Alec, who previously thought of himself as good at random trivia, answered 17 of 100 randomly sampled examples correctly when tested in the same setting as GPT-2. He actually only got 14 right but he should have gotten those other 3"

- L'exemple de génération sur les "talking unicorns" (Table 13) démontre la capacité à générer du texte cohérent et stylistiquement approprié sur des sujets absurdes

- Le filtrage par karma Reddit comme proxy de qualité est une idée simple mais efficace - précurseur des techniques de filtrage de données qui deviendront cruciales pour les LLMs

Ce papier marque le début de l'ère "scaling laws" : la démonstration empirique que les capacités des LMs s'améliorent de manière prévisible avec la taille, ouvrant la voie à GPT-3 et au-delà.
