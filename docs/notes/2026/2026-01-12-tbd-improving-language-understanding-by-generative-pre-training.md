---
title: "Improving Language Understanding by Generative Pre-Training"
authors: ["Alec Radford", "Karthik Narasimhan", "Tim Salimans", "Ilya Sutskever"]
venue: "OpenAI"
year: 2018
citekey: Radford2018ImprovingLU
links:
  pdf: "https://api.semanticscholar.org/CorpusID:49313245"
  code: ""
  project: ""
pdf_annotated: "../../papers/2026/GPT1.pdf"
tags:
  areas: ["NLP", "Transfer Learning", "Language Modeling"]
  methods: ["GPT", "Transformer Decoder", "Generative Pre-training", "Fine-tuning"]
  tasks: ["Natural Language Inference", "Question Answering", "Semantic Similarity", "Text Classification"]
  status: "deep-read"
replication:
  repo: ""
  data: "BooksCorpus"
  env: ""
---

## TL;DR

GPT (Generative Pre-Training Transformer) introduit un framework pour le transfer learning en NLP combinant **pré-entraînement génératif non supervisé** (modèle de langage) sur un large corpus, suivi d'un **fine-tuning discriminatif** sur des tâches spécifiques. L'architecture repose sur un **Transformer decoder** (12 couches, 768 dimensions, 12 têtes d'attention) pré-entraîné sur BooksCorpus avec un objectif de modélisation de langage unidirectionnel. L'innovation clé est l'utilisation de **transformations d'entrée task-aware** (traversal-style) qui convertissent toutes les tâches structurées en séquences de tokens, permettant un transfer avec **modifications architecturales minimales** (seulement une couche linear+softmax). Le fine-tuning utilise un **objectif auxiliaire de modélisation de langage** (λ=0.5) en plus de l'objectif supervisé pour améliorer la généralisation. GPT atteint des **nouveaux state-of-the-art sur 9/12 tâches** testées : +8.9% sur Story Cloze (86.5%), +5.7% sur RACE (59.0%), +1.5% sur MultiNLI (82.1%), et 72.8 sur le benchmark GLUE. Les analyses zero-shot révèlent que le modèle acquiert des capacités linguistiques (sentiment, acceptabilité) durant le pré-entraînement. Les études d'ablation montrent que chaque couche Transformer contribue jusqu'à 9% d'amélioration, et que l'absence de pré-entraînement cause une baisse de 14.8%.

## Contexte

En 2018, le deep learning en NLP souffre de deux problèmes majeurs :

**Dépendance aux données supervisées** : Les modèles discriminatifs nécessitent de grandes quantités de données annotées, qui sont rares, coûteuses et limitent l'applicabilité dans de nombreux domaines. Bien que les word embeddings pré-entraînés (Word2Vec, GloVe) soient largement utilisés, ils ne transfèrent que des informations au niveau des mots, pas de sémantique de niveau supérieur.

**Incertitudes sur le transfer learning** :
1. **Objectif de pré-entraînement optimal** : Quel objectif apprend les meilleures représentations transférables ? Les recherches ont exploré diverses approches (language modeling, machine translation, discourse coherence), chaque méthode surpassant les autres sur différentes tâches.
2. **Méthode de transfer efficace** : Comment transférer ces représentations ? Les techniques existantes impliquent des modifications architecturales task-specific, des schémas d'apprentissage complexes, ou des objectifs auxiliaires, rendant le développement d'approches semi-supervisées difficile.

**Approches existantes et leurs limites** :
- **ELMo (Peters et al. 2018)** : Utilise des biLSTMs et les représentations comme features auxiliaires, nécessitant des architectures task-specific et de nouveaux paramètres pour chaque tâche.
- **ULMFiT (Howard & Ruder 2018)** : Fine-tuning de LSTM pour classification de texte, mais les LSTMs limitent la prédiction à courte portée.
- **CoVe (McCann et al. 2017)** : Utilise un encodeur de traduction automatique comme extracteur de features, mais nécessite des données supervisées parallèles.

**Motivation pour GPT** : Développer un modèle **task-agnostic** qui apprend une représentation universelle transférable avec une adaptation minimale, en choisissant le Transformer pour sa capacité à capturer les dépendances longue-distance et permettre une parallélisation massive contrairement aux RNNs.

## Idées clés

1. **Pré-entraînement génératif + Fine-tuning discriminatif** : Procédure en deux étapes séparées. D'abord, apprendre un modèle de langage génératif sur un large corpus non supervisé (BooksCorpus, 7000 livres, séquences contiguës longues). Ensuite, adapter les paramètres aux tâches supervisées. Cette séparation claire permet un pré-entraînement coûteux une fois, puis un fine-tuning rapide sur chaque tâche (3 epochs suffisent).

2. **Transformer Decoder comme architecture unifiée** : Utilisation d'un décodeur Transformer (12 couches, masked self-attention) plutôt que des LSTMs. Avantages :
   - Mémoire structurée pour dépendances longue-distance
   - Parallélisation massive (O(1) opérations séquentielles vs O(n) pour RNNs)
   - Inductive bias favorable au transfer (variance zero-shot plus stable que LSTMs)

3. **Task-aware input transformations (traversal-style)** : Au lieu de créer des architectures task-specific, convertir **toutes** les entrées structurées en séquences de tokens avec des delimiters :
   - **Classification** : [Start] Text [Extract]
   - **Entailment** : [Start] Premise [Delim] Hypothesis [Extract]
   - **Similarity** : Traiter les deux ordres [Start] Text1 [Delim] Text2 [Extract] + [Start] Text2 [Delim] Text1 [Extract], additionner les représentations
   - **QA/Multiple choice** : [Start] Context [Delim] Question [Delim] Answer_i [Extract] pour chaque réponse, softmax sur les scores

   Cette approche permet de fine-tuner avec **modifications minimales** (seulement Wy et embeddings des delimiters).

4. **Objectif auxiliaire de modélisation de langage durant le fine-tuning** :
   ```
   L₃(C) = L₂(C) + λ * L₁(C)
   ```
   où L₂ est l'objectif supervisé et L₁ l'objectif de modélisation de langage. Bénéfices :
   - Améliore la généralisation (régularisation)
   - Accélère la convergence
   - Particulièrement utile sur les grands datasets (NLI, QQP)
   - λ=0.5 empiriquement optimal

5. **Apprentissage de capacités zero-shot durant le pré-entraînement** : Le modèle génératif apprend à effectuer de nombreuses tâches pour améliorer sa capacité de modélisation de langage :
   - **Sentiment** : Ajouter "very" et comparer p("positive") vs p("negative")
   - **Acceptabilité linguistique (CoLA)** : Score = log-probabilité moyenne par token
   - **QA** : Sélectionner la réponse avec la plus haute log-probabilité conditionnelle
   - Les performances zero-shot augmentent régulièrement avec le pré-entraînement, suggérant l'acquisition de connaissances linguistiques transférables

6. **Chaque couche Transformer contient des fonctionnalités utiles** : Transfer progressif de 0 à 12 couches montre des gains jusqu'à 9% sur MultiNLI. Contrairement aux word embeddings qui transfèrent seulement la surface, les couches profondes capturent des abstractions linguistiques de plus haut niveau.

7. **Texte continu longue-distance crucial** : BooksCorpus contient de longues séquences contiguës, permettant au modèle d'apprendre à se conditionner sur des informations longue-distance. Le 1B Word Benchmark (utilisé par ELMo) est mélangé au niveau des phrases, détruisant la structure à long terme.

8. **Fine-tuning rapide et efficace** : 3 epochs suffisent pour la plupart des tâches avec un learning rate de 6.25e-5, batch size 32, et linear decay avec warmup sur 0.2% du training. Le modèle s'adapte rapidement grâce aux représentations riches pré-entraînées.

## Méthode

### Architecture du modèle

**Transformer Decoder (12 couches)** :
- **Dimensions** : d_model=768, d_ff=3072, 12 attention heads (d_k=64 par tête)
- **Self-attention masquée** : Chaque position ne peut attendre qu'aux positions précédentes (causal masking)
- **Position-wise FFN** : FFN(x) = max(0, xW₁+b₁)W₂+b₂
- **Residual connections + LayerNorm** : Appliqué autour de chaque sub-layer
- **Activation** : GELU (Gaussian Error Linear Unit) au lieu de ReLU
- **Position embeddings** : Learned position embeddings (pas sinusoïdaux comme Transformer original)
- **Régularisation** :
  - Dropout 0.1 sur residual, embedding et attention
  - L2 regularization modifiée (w=0.01) sur tous les poids sauf bias/gain
  - Label smoothing non utilisé

**Nombre de paramètres** : ~117M paramètres

### Pré-entraînement non supervisé

**Objectif de modélisation de langage** :
```
L₁(U) = Σᵢ log P(uᵢ | uᵢ₋ₖ, ..., uᵢ₋₁; Θ)
```
Modèle de langage unidirectionnel (forward) maximisant la log-vraisemblance conditionnelle.

**Corpus** : BooksCorpus (Zhu et al. 2015)
- 7000+ livres uniques non publiés
- Genres variés : Adventure, Fantasy, Romance
- Avantage : Longues séquences contiguës (vs 1B Word Benchmark mélangé)
- Perplexité finale : 18.4 (très faible)

**Tokenization** : Byte Pair Encoding (BPE) avec 40,000 merges, vocabulaire de ~40k tokens

**Entraînement** :
- 100 epochs
- Batch size : 64
- Séquences de 512 tokens contiguës échantillonnées aléatoirement
- Optimiseur : Adam (lr_max=2.5e-4, β₁=0.9, β₂=0.999, ε=1e-8)
- Learning rate schedule : Linear warmup sur 2000 steps, puis cosine annealing vers 0
- Weight initialization : N(0, 0.02)
- Librairies : ftfy pour nettoyage de texte, spaCy tokenizer

### Fine-tuning supervisé

**Pour chaque tâche** :
1. Transformer les entrées en séquences avec delimiters (voir section Idées clés #3)
2. Passer la séquence à travers le modèle pré-entraîné (poids initialisés)
3. Extraire l'activation de la dernière couche Transformer au token final : h_m^l
4. Ajouter une couche linear : P(y|x¹,...,xᵐ) = softmax(h_m^l W_y)
5. Optimiser l'objectif combiné : L₃(C) = L₂(C) + 0.5 * L₁(C)

**Hyperparamètres de fine-tuning** :
- Learning rate : 6.25e-5
- Batch size : 32
- Epochs : 3 (suffisant pour la plupart des tâches)
- Dropout classifier : 0.1
- Linear decay avec warmup sur 0.2% du training
- λ=0.5 pour l'objectif auxiliaire LM

**Seuls paramètres ajoutés** : W_y (couche de sortie task-specific) + embeddings des tokens delimiters ([Start], [Delim], [Extract])

### Transformations d'entrée task-specific

**Classification** : [Start] + Text + [Extract] → Linear(h_extract)

**Entailment** : [Start] + Premise + [Delim] + Hypothesis + [Extract] → Linear(h_extract)

**Similarity** :
- Séquence 1 : [Start] + Sent1 + [Delim] + Sent2 + [Extract] → h_m¹
- Séquence 2 : [Start] + Sent2 + [Delim] + Sent1 + [Extract] → h_m²
- Représentation finale : h_m¹ + h_m² (element-wise addition)

**Multiple Choice (QA, Story Cloze)** :
- Pour chaque réponse aₖ : [Start] + Context + [Delim] + Question + [Delim] + aₖ + [Extract]
- Score chaque séquence indépendamment
- Softmax sur tous les scores pour obtenir la distribution sur les réponses

## Résultats

### Performance globale

**9 nouveaux state-of-the-art sur 12 tâches** testées, surpassant souvent des ensembles de modèles task-specific avec un seul modèle task-agnostic.

**GLUE benchmark** : Score global de **72.8** (vs 68.9 précédent SOTA, +3.9 points)

### Natural Language Inference (Tableau 2)

| Tâche | Baseline SOTA | GPT | Amélioration |
|-------|---------------|-----|--------------|
| **MultiNLI-m** | 80.6 (ensemble) | **82.1** | +1.5% |
| **MultiNLI-mm** | 80.1 | **81.4** | +1.3% |
| **SNLI** | 89.3 (ESIM+ELMo ensemble) | **89.9** | +0.6% |
| **SciTail** | 83.3 | **88.3** | +5.0% |
| **QNLI** | 82.3 | **88.1** | +5.8% |
| **RTE** | 61.7 | 56.0 | -5.7% (petit dataset) |

**Observations** :
- Surpasse les baselines sur 4/5 datasets NLI
- RTE (2490 exemples) : Performance inférieure, probablement bénéficierait du multi-task training
- Démontre la capacité à raisonner sur plusieurs phrases et gérer l'ambiguïté linguistique

### Question Answering et Commonsense Reasoning (Tableau 3)

| Tâche | Baseline SOTA | GPT | Amélioration |
|-------|---------------|-----|--------------|
| **Story Cloze** | 77.6 | **86.5** | +8.9% (énorme) |
| **RACE** (overall) | 53.3 (ensemble 9x) | **59.0** | +5.7% |
| **RACE-m** (middle) | 60.2 | **62.9** | +2.7% |
| **RACE-h** (high) | 50.3 | **57.4** | +7.1% |

**Observations** :
- Story Cloze : Gain massif de 8.9%, démontrant une excellente compréhension narrative
- RACE : Dataset difficile (examens middle/high school) avec plus de raisonnement que CNN/SQuAD
- Capacité exceptionnelle à gérer les contextes longue-distance

### Semantic Similarity (Tableau 4)

| Tâche | Baseline SOTA | GPT | Amélioration |
|-------|---------------|-----|--------------|
| **STS-B** (Pearson) | 81.0 | **82.0** | +1.0 |
| **MRPC** (F1) | 83.5 | 82.3 | -1.2 |
| **QQP** (F1) | 66.1 | **70.3** | +4.2% |

**Observations** :
- QQP : Gain significatif de 4.2% sur la détection de paraphrases
- SOTA sur 2/3 tâches de similarité sémantique

### Classification (Tableau 4)

| Tâche | Métrique | Baseline SOTA | GPT | Amélioration |
|-------|----------|---------------|-----|--------------|
| **CoLA** | Matthews Corr. | 35.0 | **45.4** | +10.4 (énorme) |
| **SST-2** | Accuracy | 91.6 | 91.3 | -0.3 (compétitif) |

**Observations** :
- CoLA : Saut massif de 35.0 → 45.4, démontrant le biais linguistique inné appris durant le pré-entraînement
- SST-2 : Performance compétitive avec le SOTA

### Analyses et ablations

**Impact du nombre de couches transférées** (Figure 2 gauche) :
- Transfer d'embeddings seulement : ~40% sur MultiNLI
- Transfer progressif de 1→12 couches : Amélioration continue
- Transfer complet (12 couches) : ~82% sur MultiNLI (+9% vs embeddings seuls)
- **Conclusion** : Chaque couche Transformer contient des fonctionnalités utiles pour les tâches downstream

**Zero-shot behaviors** (Figure 2 droite) :
- Performances zero-shot augmentent régulièrement durant le pré-entraînement
- Tâches testées : Sentiment analysis, Winograd schema, Linguistic acceptability, QA
- Le modèle génératif apprend des capacités task-relevant pour améliorer sa modélisation de langage
- Transformer montre une variance plus faible que LSTM, suggérant un meilleur inductive bias pour le transfer

**Études d'ablation** (Tableau 5) :

| Configuration | Score moyen | Impact |
|---------------|-------------|--------|
| **GPT complet** (w/ aux LM) | **74.7** | Baseline |
| Sans pré-entraînement | 59.9 | **-14.8%** (impact énorme) |
| Sans objectif auxiliaire LM | 75.0 | +0.3% (aide surtout sur grands datasets) |
| LSTM au lieu de Transformer | 69.1 | **-5.6%** (Transformer crucial) |

**Observations** :
- **Pré-entraînement** : Impact massif (-14.8% sans), validant l'approche centrale
- **Objectif auxiliaire** : Aide sur NLI et QQP (grands datasets), neutre/négatif sur petits datasets
- **Transformer vs LSTM** : Transformer surpasse systématiquement sauf sur MRPC
- **Architecture** : Les attention heads structurées du Transformer sont cruciales pour le transfer

**Efficacité sur différentes tailles de datasets** :
- Fonctionne bien de STS-B (5.7k exemples) à SNLI (550k exemples)
- Contrairement à d'autres approches, GPT ne nécessite pas de très grands datasets supervisés

## Limites

1. **Architecture unidirectionnelle (forward only)** : Le modèle de langage est uniquement forward (gauche→droite), ne capturant pas le contexte bidirectionnel. ELMo utilise des biLSTMs bidirectionnels, ce qui pourrait être avantageux pour certaines tâches nécessitant un contexte complet. BERT (2018, publié quelques mois après) résoudra cela avec un masked language model bidirectionnel.

2. **Longueur de séquence limitée** : Entraîné sur séquences de 512 tokens maximum. Pour des documents très longs (milliers de tokens), le modèle doit tronquer ou découper le texte, perdant potentiellement des informations contextuelles importantes.

3. **Performances variables sur petits datasets** : Sur RTE (2490 exemples), GPT sous-performe (56.0% vs 61.7%). Les petits datasets bénéficieraient probablement de multi-task training ou de techniques de data augmentation non explorées ici.

4. **Objectif auxiliaire pas toujours bénéfique** : L'objectif de modélisation de langage durant le fine-tuning aide sur les grands datasets (MNLI, QQP) mais pas systématiquement sur les petits. Le poids λ=0.5 est fixe, un scheduling ou un tuning task-specific pourrait améliorer.

5. **Coût computationnel du pré-entraînement** : Bien que non chiffré précisément dans le paper, pré-entraîner un Transformer 12-couches sur BooksCorpus pour 100 epochs nécessite des ressources GPU importantes, limitant la reproducibilité pour les chercheurs sans accès à de larges ressources de calcul.

6. **Transformations d'entrée manuelles** : Les input transformations (start/delim tokens, ordre des séquences) sont conçues manuellement pour chaque type de tâche. Une approche plus automatique ou apprise serait souhaitable.

7. **Comparaison limitée avec multi-task learning** : Le paper ne compare pas avec des approches multi-task qui pourraient bénéficier au modèle, particulièrement sur les petits datasets. L'entraînement est strictement séquentiel (pré-training puis fine-tuning).

8. **Pas d'analyse d'interprétabilité** : Contrairement à ELMo qui analyse explicitement quelle couche capture quoi (syntaxe vs sémantique), GPT ne fournit pas d'analyse détaillée des représentations apprises, seulement des expériences zero-shot heuristiques.

9. **Vocabulaire BPE fixe** : Le vocabulaire de 40k tokens est figé après le pré-entraînement. Pour des domaines très spécialisés (médical, légal), des tokens out-of-vocabulary nécessitent une décomposition en sub-words, potentiellement sous-optimale.

10. **Génération vs compréhension** : Le modèle est pré-entraîné pour la génération mais évalué sur la compréhension. Bien que cela fonctionne empiriquement, il n'y a pas de garantie théorique que l'objectif génératif soit optimal pour toutes les tâches discriminatives.

11. **Pas d'évaluation cross-lingual** : Toutes les expériences sont en anglais. La capacité du modèle à transférer à d'autres langues ou dans des contextes multilingues n'est pas explorée.

12. **Fine-tuning catastrophic forgetting** : Le fine-tuning sur une tâche spécifique pourrait potentiellement détériorer les capacités générales acquises durant le pré-entraînement, non investigué ici.

13. **Dépendance au BooksCorpus** : Le corpus de livres peut introduire des biais stylistiques ou thématiques. Les performances sur des textes très différents (tweets, code, transcriptions orales) sont inconnues.

## Liens utiles

- **PDF annoté**: [PDF annoté](../../papers/2026/GPT1.pdf)
- **Article**: [Improving Language Understanding by Generative Pre-Training](https://api.semanticscholar.org/CorpusID:49313245)
- **OpenAI Blog**: [https://openai.com/blog/language-unsupervised/](https://openai.com/blog/language-unsupervised/)

## Notes perso

ELMo avait a peu pres le meme objectif que GPT 

- L'objectif de GPT est d'apprendre une représentation universelle qui se transfère avec une adaptation minimale à une large gamme de tâches.
- L'objectif d'ELMo est de générer des plongements de mots contextualisés (word vectors) de haute qualité.

mais il y a des diffs:

Architecture et Mémoire

    Modèle : ELMo utilise des architectures LSTM récurrentes. GPT utilise des Transformers.

Portée temporelle : Les modèles LSTM utilisés par ELMo restreignent leur capacité de prédiction à une courte portée. Le Transformer de GPT permet de capturer des structures linguistiques à plus longue portée grâce à une mémoire attentionnelle plus structurée.

Données de Pré-entraînement

    Corpus : ELMo est entraîné sur le 1B Word Benchmark. GPT utilise le BooksCorpus.

Cohérence du texte : Le corpus d'ELMo est mélangé au niveau des phrases, ce qui détruit la structure sémantique à long terme. GPT est entraîné sur de longues séquences de texte contiguës, permettant au modèle d'apprendre à se conditionner sur des informations à longue portée.

Méthode de Transfert et Intégration

    Utilisation des représentations : ELMo utilise les représentations cachées du modèle pré-entraîné comme caractéristiques auxiliaires lors de l'entraînement d'un modèle supervisé séparé. GPT adapte directement les paramètres pré-entraînés via un fine-tuning discriminatif sur la tâche cible.

Changements architecturaux : ELMo nécessite l'ajout d'une quantité substantielle de nouveaux paramètres et d'architectures spécifiques pour chaque tâche. GPT utilise des transformations d'entrée de style traversée pour traiter les textes structurés comme des séquences continues, ce qui permet de maintenir une architecture agnostique à la tâche avec des modifications minimales.

Performance

    Efficacité comparative : GPT surpasse les approches basées sur ELMo sur plusieurs benchmarks de compréhension du langage, notamment sur les tâches d'inférence de langage naturel (MNLI, QNLI, SciTail) et de réponse à des questions (RACE).

Zéro-shot : GPT démontre des capacités de transfert zéro-shot, acquérant des connaissances linguistiques utilisables pour des tâches en aval dès la phase de pré-entraînement génératif, une propriété non exploitée par le cadre ELMo.