---
title: "BERT: Pre-training of Deep Bidirectional Transformers for Language Understanding"
authors: ["Jacob Devlin", "Ming-Wei Chang", "Kenton Lee", "Kristina Toutanova"]
venue: "NAACL"
year: 2019
citekey: Devlin2019BERTPO
links:
  pdf: "https://arxiv.org/abs/1810.04805"
  code: "https://github.com/google-research/bert"
  project: ""
pdf_annotated: "../../papers/2026/bert.pdf"
tags:
  areas: ["NLP", "Transfer Learning", "Language Modeling"]
  methods: ["BERT", "Transformer Encoder", "Masked Language Model", "Next Sentence Prediction", "Bidirectional Pre-training"]
  tasks: ["Question Answering", "Natural Language Inference", "Named Entity Recognition", "Text Classification", "Semantic Similarity"]
  status: "deep-read"
replication:
  repo: "https://github.com/google-research/bert"
  data: "BooksCorpus + Wikipedia"
  env: ""
---

## TL;DR

BERT (Bidirectional Encoder Representations from Transformers) introduit un modèle de représentation de langage **bidirectionnel profond** pré-entraîné via deux tâches non supervisées : **Masked Language Model (MLM)** qui masque aléatoirement 15% des tokens et prédit les tokens masqués en se conditionnant sur le contexte gauche ET droit dans toutes les couches, et **Next Sentence Prediction (NSP)** qui prédit si deux phrases sont consécutives. L'architecture repose sur un **Transformer encoder** (12 ou 24 couches, attention bidirectionnelle) pré-entraîné sur BooksCorpus (800M mots) + Wikipedia (2,500M mots). Contrairement à GPT (unidirectionnel forward) et ELMo (concaténation shallow de LSTMs indépendants), BERT fusionne véritablement les contextes gauche et droit en profondeur. Le fine-tuning nécessite seulement **une couche de sortie additionnelle** sans modifications architecturales majeures. BERT atteint des **nouveaux state-of-the-art sur 11 tâches NLP** : GLUE score à 80.5% (+7.7%), MultiNLI à 86.7% (+4.6%), SQuAD v1.1 à 93.2 F1 (+1.5), SQuAD v2.0 à 83.1 F1 (+5.1), SWAG à 86.3% (+8.3% vs GPT). Les études d'ablation montrent que la bidirectionnalité et les deux tâches de pré-entraînement sont cruciales : supprimer NSP dégrade QNLI/MNLI/SQuAD de -0.5 à -4.4%, et un modèle left-to-right sans NSP perd jusqu'à -10.7% sur MRPC et -7.1% sur SQuAD.

## Contexte

En 2018-2019, le transfer learning en NLP connaît un essor majeur avec ELMo et GPT-1, mais des limitations persistent :

**Limites des modèles unidirectionnels** : GPT utilise un modèle de langage **left-to-right** (forward uniquement) où chaque token ne peut attendre qu'aux tokens précédents dans les couches d'attention du Transformer. Cette restriction est :
- **Sub-optimale pour les tâches sentence-level** (NLI, paraphrase) qui bénéficient d'une analyse holistique bidirectionnelle
- **Très dommageable pour les tâches token-level** (QA, NER) où il est crucial d'incorporer le contexte des deux directions pour prédire précisément

**Limites de la bidirectionnalité shallow d'ELMo** : ELMo utilise une **concaténation superficielle** de LSTMs forward et backward **entraînés indépendamment**. Chaque direction n'a aucune information sur l'autre pendant l'entraînement, limitant la capacité à capturer les interactions complexes entre contextes gauche et droit.

**Approches existantes** :
- **ELMo (Peters et al. 2018)** : BiLSTMs avec représentations feature-based, nécessite des architectures task-specific et nouveaux paramètres pour chaque tâche
- **GPT (Radford et al. 2018)** : Transformer left-to-right, fine-tuning avec modifications minimales, mais unidirectionnel
- **ULMFiT (Howard & Ruder 2018)** : Fine-tuning de LSTM pour classification, portée limitée

**Motivation pour BERT** : Développer un modèle qui combine les avantages du fine-tuning (minimal task-specific parameters) avec une **vraie bidirectionnalité profonde** où chaque couche peut attendre simultanément aux contextes gauche ET droit, permettant d'apprendre des représentations plus riches.

## Idées clés

1. **Masked Language Model (MLM) pour bidirectionnalité profonde** : Masquer aléatoirement 15% des tokens d'entrée et prédire les tokens masqués uniquement sur la base du contexte (gauche + droit). Cela contourne le problème des modèles de langage bidirectionnels standard où chaque mot pourrait indirectement "se voir lui-même" dans un contexte multi-couches, rendant la prédiction triviale. L'objectif MLM force le modèle à maintenir une représentation contextuelle distributionnelle de **chaque** token d'entrée.

2. **Procédure de masking mixte** : Pour réduire le mismatch pre-training/fine-tuning (le token [MASK] n'apparaît jamais en fine-tuning), BERT utilise une stratégie mixte pour les 15% de tokens choisis :
   - **80% du temps** : remplacer par [MASK] (e.g., "my dog is hairy" → "my dog is [MASK]")
   - **10% du temps** : remplacer par un token aléatoire (e.g., "my dog is hairy" → "my dog is apple")
   - **10% du temps** : garder inchangé (e.g., "my dog is hairy" → "my dog is hairy")

   Comme le remplacement aléatoire n'affecte que 1.5% des tokens totaux (10% de 15%), cela ne nuit pas aux capacités de compréhension.

3. **Next Sentence Prediction (NSP) pour comprendre les relations inter-phrases** : Beaucoup de tâches importantes (QA, NLI) reposent sur la compréhension de la **relation entre deux phrases**, non directement capturée par le language modeling. NSP est une tâche binaire : étant donné deux phrases A et B, prédire si B est la phrase suivante réelle de A (IsNext) ou une phrase aléatoire du corpus (NotNext). 50% des exemples sont IsNext, 50% NotNext. Le vecteur [CLS] final est utilisé pour cette prédiction (97-98% accuracy finale).

4. **Architecture unifiée avec modifications minimales** : Une seule architecture Transformer encoder bidirectionnelle pour toutes les tâches. Le fine-tuning nécessite seulement :
   - **Pour classification** : Couche linear sur [CLS]
   - **Pour QA** : Vecteurs start S et end E pour prédire le span de réponse
   - **Pour tagging** : Couche linear sur chaque token Ti

   Tous les paramètres sont fine-tunés end-to-end avec les nouvelles couches.

5. **Représentation d'entrée flexible** :
   - **Input embedding** = Token embeddings + Segment embeddings (A/B) + Position embeddings (learned)
   - **[CLS]** : Token spécial au début de chaque séquence, sa représentation finale sert pour classification
   - **[SEP]** : Token séparateur entre phrases
   - Peut représenter une phrase seule ou une paire de phrases dans une séquence unifiée
   - Longueur max : 512 tokens

6. **Self-attention pour cross attention bidirectionnelle** : Contrairement aux approches qui encodent indépendamment les paires de texte puis appliquent une cross attention bidirectionnelle (Parikh et al. 2016, Seo et al. 2017), BERT utilise le **mécanisme de self-attention** pour unifier ces deux étapes. Encoder une paire de texte concaténée avec self-attention inclut effectivement une cross attention bidirectionnelle entre les deux phrases.

7. **Pré-entraînement à large échelle sur corpus de documents** :
   - **BooksCorpus** (800M mots) + **English Wikipedia** (2,500M mots, texte seulement)
   - Crucial d'utiliser un corpus **document-level** plutôt que sentence-level shuffled (comme 1B Word Benchmark) pour extraire des séquences contiguës longues et apprendre les dépendances long-range
   - 1M steps, batch size 256 sequences (128,000 tokens/batch), ~40 epochs
   - WordPiece tokenization (vocabulaire 30,000 tokens)

8. **Deux tailles de modèles** :
   - **BERT_BASE** : L=12 couches, H=768 hidden, A=12 heads, 110M paramètres (comparable à GPT)
   - **BERT_LARGE** : L=24 couches, H=1024 hidden, A=16 heads, 340M paramètres

   La seule différence architecturale avec GPT : BERT utilise **bidirectional self-attention**, GPT utilise **constrained self-attention** (masked) où chaque token n'attend qu'au contexte à gauche.

9. **Fine-tuning rapide et peu coûteux** :
   - Typiquement 2-4 epochs suffisent
   - Learning rate task-specific (5e-5, 4e-5, 3e-5, 2e-5)
   - Batch size 16-32
   - Fine-tuning complet en 1 heure sur 1 Cloud TPU ou quelques heures sur GPU
   - Par exemple, BERT SQuAD fine-tuné en ~30 minutes pour 91.0% Dev F1

10. **Scaling massif efficace même sur petits datasets** : Première démonstration convaincante que scaler à des tailles de modèles extrêmes améliore fortement même les **très petites tâches**, à condition que le modèle soit suffisamment pré-entraîné. BERT_LARGE surpasse BERT_BASE sur toutes les tâches, surtout celles avec très peu de données d'entraînement (e.g., MRPC 3,600 exemples : +1.1%, CoLA 8,500 exemples : +8.4%).

## Méthode

### Architecture du modèle

**Transformer Encoder bidirectionnel** basé sur Vaswani et al. (2017) :

**BERT_BASE** :
- L=12 couches Transformer
- H=768 hidden size
- A=12 attention heads (64 dimensions par head)
- Feed-forward size = 4H = 3072
- **110M paramètres**

**BERT_LARGE** :
- L=24 couches Transformer
- H=1024 hidden size
- A=16 attention heads (64 dimensions par head)
- Feed-forward size = 4H = 4096
- **340M paramètres**

**Composants** :
- **Bidirectional self-attention** : Chaque token peut attendre à tous les autres tokens (pas de masking causal)
- **Position-wise FFN** : Dense layer avec activation GELU
- **Residual connections + LayerNorm** autour de chaque sub-layer
- **Activation** : GELU (Gaussian Error Linear Unit) au lieu de ReLU
- **Position embeddings** : Learned (pas sinusoïdaux)
- **Régularisation** : Dropout 0.1 sur tous les layers, L2 weight decay 0.01

### Représentation d'entrée

Pour un token donné, son input embedding = **Token embedding + Segment embedding + Position embedding**

**Tokens spéciaux** :
- **[CLS]** : Toujours le premier token, sa représentation finale C ∈ R^H sert pour classification
- **[SEP]** : Sépare les phrases dans une paire

**Segment embeddings** :
- Embedding learned E_A pour tokens de la phrase A
- Embedding learned E_B pour tokens de la phrase B
- Permet au modèle de différencier les phrases dans une paire

**Format pour différentes tâches** :
- **Phrase unique** : [CLS] + Sentence + [SEP]
- **Paire de phrases** : [CLS] + Sentence A + [SEP] + Sentence B + [SEP]

### Pré-entraînement

**Tâche #1 : Masked Language Model (MLM)**

Objectif : Prédire les tokens masqués basé uniquement sur leur contexte.

**Procédure** :
1. Sélectionner aléatoirement 15% des positions de tokens pour prédiction
2. Pour chaque position i sélectionnée :
   - 80% : remplacer par [MASK]
   - 10% : remplacer par un token aléatoire
   - 10% : garder le token original inchangé
3. Prédire le token original avec cross entropy loss sur le vocabulaire

**Avantages** :
- Permet de pré-entraîner une représentation bidirectionnelle profonde
- Force le modèle à maintenir une représentation contextuelle de chaque token

**Inconvénient** :
- Mismatch pre-training/fine-tuning (le token [MASK] n'apparaît pas en fine-tuning), atténué par la stratégie mixte
- Seulement 15% des tokens sont prédits par batch, donc convergence plus lente qu'un LM standard (mais largement compensé par les gains empiriques)

**Tâche #2 : Next Sentence Prediction (NSP)**

Objectif : Comprendre les relations entre phrases pour des tâches comme QA et NLI.

**Procédure** :
1. Pour chaque exemple de pré-entraînement, choisir phrases A et B :
   - 50% du temps : B est la phrase suivante réelle de A (label = IsNext)
   - 50% du temps : B est une phrase aléatoire du corpus (label = NotNext)
2. Le vecteur final [CLS] (noté C) est utilisé pour prédiction binaire NSP
3. Loss : Cross entropy pour prédiction IsNext/NotNext

**Performance** : Le modèle final atteint 97-98% accuracy sur NSP.

**Importance** : Les ablations (Section 5.1) montrent que supprimer NSP nuit significativement à QNLI, MNLI et SQuAD.

**Données de pré-entraînement** :
- **BooksCorpus** (Zhu et al. 2015) : 800M mots, 7000+ livres non publiés
- **English Wikipedia** : 2,500M mots (texte uniquement, pas listes/tables/headers)
- **Total** : 3.3 milliards de mots
- **Tokenization** : WordPiece avec vocabulaire 30,000 tokens

**Entraînement** :
- 1,000,000 steps (~40 epochs)
- Batch size : 256 sequences (128,000 tokens/batch)
- Séquences de 512 tokens
- Optimiseur : Adam (lr=1e-4, β₁=0.9, β₂=0.999, L2 decay=0.01)
- Learning rate : Warmup sur 10,000 steps, puis linear decay
- Dropout : 0.1 sur tous les layers
- Activation : GELU
- Training loss : Somme de mean masked LM likelihood + mean NSP likelihood
- **BERT_BASE** : 4 jours sur 4 Cloud TPUs (16 TPU chips)
- **BERT_LARGE** : 4 jours sur 16 Cloud TPUs (64 TPU chips)

**Astuce d'optimisation** :
- 90% des steps avec séquences de longueur 128 (attention quadratique, plus rapide)
- 10% des steps avec séquences de longueur 512 (pour apprendre les position embeddings)

### Fine-tuning

**Procédure générale** :
1. Initialiser avec les paramètres pré-entraînés BERT
2. Ajouter une couche de sortie task-specific
3. Fine-tuner **tous** les paramètres end-to-end sur la tâche cible

**Hyperparamètres typiques** :
- Batch size : 16 ou 32
- Learning rate : 5e-5, 4e-5, 3e-5, ou 2e-5 (task-specific)
- Epochs : 2, 3, ou 4
- Dropout : 0.1

**Pour BERT_LARGE** : Fine-tuning parfois instable sur petits datasets, donc plusieurs random restarts et sélection du meilleur modèle sur Dev set.

**Adaptations task-specific** :

**Classification (sentence-level)** :
- Exemples : MNLI, QQP, QNLI, SST-2, CoLA, STS-B, MRPC, RTE
- Input : [CLS] + Sentence (A) + [SEP] (+ Sentence (B) + [SEP] si paire)
- Output : Linear layer W ∈ R^(K×H) sur représentation [CLS] finale C
- Prédiction : P(y|x) = softmax(C·W^T)
- Loss : Standard classification loss

**Question Answering (SQuAD)** :
- Input : [CLS] + Question + [SEP] + Paragraph + [SEP]
- Question reçoit segment embedding A, paragraph reçoit B
- Ajouter deux vecteurs : Start S ∈ R^H et End E ∈ R^H
- Pour chaque mot i dans le paragraph :
  - P_start(i) = e^(S·Ti) / Σ_j e^(S·Tj)
  - P_end(i) = e^(E·Ti) / Σ_j e^(E·Tj)
- Score d'un span candidat de position i à j : S·Ti + E·Tj
- Prédiction : span avec score maximum où j ≥ i
- Loss : Somme des log-likelihoods des positions start et end correctes

**SQuAD v2.0 (avec questions sans réponse)** :
- Traiter les questions sans réponse comme ayant start et end au token [CLS]
- Score no-answer : s_null = S·C + E·C
- Prédire non-null answer si max_score > s_null + τ (τ choisi sur Dev set pour maximiser F1)

**Sequence Tagging (NER)** :
- Input : [CLS] + Tokens + [SEP]
- Output : Linear layer sur chaque représentation token Ti
- Prédiction : Tag pour chaque token
- Pas de CRF layer utilisé

**Multiple Choice (SWAG)** :
- Construire 4 séquences : [CLS] + Sentence + [SEP] + Continuation_k + [SEP] pour k=1..4
- Score chaque séquence indépendamment via dot product d'un vecteur avec représentation [CLS]
- Softmax sur les 4 scores

## Résultats

### GLUE Benchmark

**Performance globale** : BERT_LARGE atteint **80.5 sur GLUE leaderboard** vs 72.8 pour GPT (date de soumission).

**Résultats détaillés (Tableau 1)** :

| Tâche | # exemples | Baseline SOTA | GPT | BERT_BASE | BERT_LARGE | Amélioration |
|-------|-----------|---------------|-----|-----------|------------|--------------|
| **MNLI-m/mm** | 392k | 80.6/80.1 | 82.1/81.4 | 84.6/83.4 | **86.7/85.9** | +4.6%/+4.5% |
| **QQP** | 363k | 66.1 | 70.3 | 71.2 | **72.1** | +1.8% |
| **QNLI** | 108k | 82.3 | 87.4 | 90.5 | **92.7** | +5.3% |
| **SST-2** | 67k | 93.2 | 91.3 | 93.5 | **94.9** | +1.7% |
| **CoLA** | 8.5k | 35.0 | 45.4 | 52.1 | **60.5** | +15.1% |
| **STS-B** | 5.7k | 81.0 | 80.0 | 85.8 | **86.5** | +5.5% |
| **MRPC** | 3.5k | 86.0 | 82.3 | 88.9 | **89.3** | +3.3% |
| **RTE** | 2.5k | 61.7 | 56.0 | 66.4 | **70.1** | +8.4% |

**Moyenne** : 74.0 (Pre-SOTA) → 75.1 (GPT) → 79.6 (BERT_BASE) → **82.1 (BERT_LARGE)**

**Observations** :
- BERT_BASE et BERT_LARGE surpassent tous les systèmes sur toutes les tâches avec des marges substantielles
- Amélioration moyenne de +4.5% (BERT_BASE) et +7.0% (BERT_LARGE) vs SOTA précédent
- BERT_BASE et GPT sont quasi identiques architecturalement sauf le masking d'attention
- BERT_LARGE surpasse significativement BERT_BASE sur toutes les tâches, **surtout celles avec très peu de données** (e.g., CoLA +8.4%, RTE +3.7%)

### SQuAD v1.1 Question Answering

**Performance** (Tableau 2) :

| Système | Dev EM/F1 | Test EM/F1 |
|---------|----------|------------|
| Human | - | 82.3/91.2 |
| Top Ensemble (nlnet) | - | 86.0/91.7 |
| BiDAF+ELMo (Single) | -/85.6 | -/85.8 |
| **BERT_BASE** (Single) | 80.8/88.5 | - |
| **BERT_LARGE** (Single) | 84.1/90.9 | - |
| **BERT_LARGE** (Ensemble 7x) | 85.8/91.8 | - |
| **BERT_LARGE** (Sgl.+TriviaQA) | 84.2/91.1 | **85.1/91.8** |
| **BERT_LARGE** (Ens.+TriviaQA) | 86.2/92.2 | **87.4/93.2** |

**Observations** :
- **+1.5 F1 absolu** vs top leaderboard system en ensemble
- **+1.3 F1** en single model
- Le single BERT model surpasse le top ensemble system
- Sans TriviaQA fine-tuning, perd seulement 0.1-0.4 F1
- Avec TriviaQA augmentation, gains massifs

### SQuAD v2.0 (avec questions sans réponse)

**Performance** (Tableau 3) :

| Système | Dev EM/F1 | Test EM/F1 |
|---------|----------|------------|
| Human | 86.3/89.0 | 86.9/89.5 |
| Top System (MIR-MRC) | - | 74.8/78.0 |
| unet (Ensemble) | - | 71.4/74.9 |
| **BERT_LARGE** (Single) | 78.7/81.9 | **80.0/83.1** |

**Observations** :
- **+5.1 F1 absolu** vs système précédent meilleur
- Performance remarquable en single model sans ensemble
- Pas de TriviaQA data utilisé pour ce modèle

### SWAG (Commonsense Reasoning)

**Performance** (Tableau 4) :

| Système | Dev | Test |
|---------|-----|------|
| ESIM+GloVe | 51.9 | 52.7 |
| ESIM+ELMo | 59.1 | 59.2 |
| OpenAI GPT | - | 78.0 |
| BERT_BASE | 81.6 | - |
| **BERT_LARGE** | 86.6 | **86.3** |
| Human (expert) | - | 85.0 |
| Human (5 annotations) | - | 88.0 |

**Observations** :
- **+27.1% vs ESIM+ELMo baseline**
- **+8.3% vs OpenAI GPT**
- Performance proche de l'humain expert (85.0%)
- Démontre des capacités exceptionnelles de raisonnement de bon sens

### Études d'ablation

**Impact des tâches de pré-entraînement** (Tableau 5, BERT_BASE) :

| Configuration | MNLI-m | QNLI | MRPC | SST-2 | SQuAD F1 |
|---------------|--------|------|------|-------|----------|
| **BERT_BASE complet** | 84.4 | 88.4 | 86.7 | 92.7 | 88.5 |
| No NSP | 83.9 (-0.5) | 84.9 (-3.5) | 86.5 (-0.2) | 92.6 (-0.1) | 87.9 (-0.6) |
| LTR & No NSP | 82.1 (-2.3) | 84.3 (-4.1) | 77.5 (-9.2) | 92.1 (-0.6) | 77.8 (-10.7) |
| + BiLSTM | 82.1 | 84.1 | 75.7 | 91.6 | 84.9 |

**Observations** :
- **Supprimer NSP** nuit significativement à QNLI (-3.5%), MNLI (-0.5%), SQuAD (-0.6%)
- **Modèle Left-to-Right** (comparable à GPT) performe beaucoup moins bien : -9.2% sur MRPC, -10.7% sur SQuAD, -4.1% sur QNLI
- **Ajouter BiLSTM** au modèle LTR améliore SQuAD (+7.1%) mais reste bien en-deçà du MLM bidirectionnel, et nuit aux tâches GLUE
- Démontre l'importance cruciale de la bidirectionnalité profonde et de NSP

**Impact de la taille du modèle** (Tableau 6) :

| #L | #H | #A | LM ppl | MNLI-m | MRPC | SST-2 |
|----|----|----|--------|--------|------|-------|
| 3 | 768 | 12 | 5.84 | 77.9 | 79.8 | 88.4 |
| 6 | 768 | 3 | 5.24 | 80.6 | 82.2 | 90.7 |
| 6 | 768 | 12 | 4.68 | 81.9 | 84.8 | 91.3 |
| 12 | 768 | 12 | 3.99 | 84.4 | 86.7 | 92.9 |
| 12 | 1024 | 16 | 3.54 | 85.7 | 86.9 | 93.3 |
| **24** | **1024** | **16** | 3.23 | **86.6** | **87.8** | **93.7** |

**Observations** :
- Les modèles plus grands améliorent strictement l'accuracy sur tous les datasets
- Amélioration même sur **MRPC avec seulement 3,600 exemples** (+8.0% de 3L à 24L)
- **Première démonstration convaincante** que scaler à des tailles extrêmes améliore même les très petites tâches, si le modèle est suffisamment pré-entraîné
- Perplexité LM améliore continûment (5.84 → 3.23)

**Approche feature-based vs fine-tuning** (Tableau 7, CoNLL-2003 NER) :

| Approche | Dev F1 | Test F1 |
|----------|--------|---------|
| ELMo | 95.7 | 92.2 |
| CVT | - | 92.6 |
| CSE | - | 93.1 |
| **Fine-tuning** | | |
| BERT_LARGE | 96.6 | **92.8** |
| BERT_BASE | 96.4 | 92.4 |
| **Feature-based (BERT_BASE)** | | |
| Embeddings only | 91.0 | - |
| Second-to-Last Hidden | 95.6 | - |
| Last Hidden | 94.9 | - |
| Weighted Sum Last 4 | 95.9 | - |
| **Concat Last 4** | **96.1** | - |
| Weighted Sum All 12 | 95.5 | - |

**Observations** :
- Fine-tuning performe légèrement mieux (96.6 vs 96.1 Dev F1)
- **Meilleure approche feature-based** : Concaténer les représentations des 4 dernières couches (seulement -0.3 F1 vs fine-tuning)
- Démontre que BERT est efficace pour **les deux approches**
- Avantages feature-based : pas besoin d'architecture Transformer pour la tâche, pré-calcul des représentations coûteuses une fois

**Convergence MLM vs LTR** (Figure 5) :

- MLM converge légèrement plus lentement que LTR (prédit seulement 15% des tokens par batch)
- **Mais MLM commence à surpasser LTR presque immédiatement** en termes d'accuracy absolue
- À 1M steps : BERT_BASE (MLM) ~84% vs ~82% pour LTR sur MNLI
- Les améliorations empiriques du MLM **dépassent largement** le coût d'entraînement accru

**Stratégies de masking** (Tableau 8, ablation MASK/SAME/RND) :

| MASK | SAME | RND | MNLI | NER Fine-tune | NER Feature-based |
|------|------|-----|------|---------------|-------------------|
| 80% | 10% | 10% | 84.2 | 95.4 | 94.9 |
| 100% | 0% | 0% | 84.3 | 94.9 | 94.0 |
| 80% | 0% | 20% | 84.1 | 95.2 | 94.6 |
| 80% | 20% | 0% | 84.4 | 95.2 | 94.7 |
| 0% | 0% | 100% | 83.6 | 94.9 | 94.6 |

**Observations** :
- Fine-tuning est **étonnamment robuste** aux différentes stratégies de masking
- Utiliser seulement MASK (100%) est problématique pour l'approche feature-based sur NER (-0.9 F1)
- Utiliser seulement RND (100%) performe bien pire que la stratégie mixte
- Stratégie BERT (80/10/10) est un bon compromis

## Limites

1. **Mismatch pré-entraînement/fine-tuning avec [MASK]** : Le token [MASK] n'apparaît jamais durant le fine-tuning, créant un décalage. Bien que la stratégie mixte (80/10/10) atténue cela, ce n'est pas une solution parfaite. Des approches alternatives comme SpanBERT ou ELECTRA tenteront de résoudre ce problème.

2. **Convergence plus lente du MLM** : Comme seulement 15% des tokens sont prédits par batch (vs 100% pour un LM standard), le MLM nécessite plus de steps pour converger. Bien que les gains empiriques compensent largement, cela augmente le coût computationnel du pré-entraînement.

3. **Coût computationnel du pré-entraînement** : BERT_BASE nécessite 4 jours sur 16 TPU chips, BERT_LARGE 4 jours sur 64 TPU chips. Ce coût est prohibitif pour la plupart des chercheurs académiques. Cependant, les modèles pré-entraînés sont publics, et le fine-tuning est rapide (heures sur GPU).

4. **NSP peut-être trop simple** : La tâche NSP atteint 97-98% accuracy, suggérant qu'elle est relativement facile. Des travaux ultérieurs (RoBERTa) remettront en question l'utilité de NSP et proposeront des alternatives (Sentence Order Prediction).

5. **Longueur de séquence limitée à 512 tokens** : Pour des documents très longs (milliers de tokens), BERT doit tronquer ou découper le texte, perdant potentiellement des informations contextuelles. Des modèles comme Longformer ou BigBird tenteront d'adresser cette limitation.

6. **Quadratic attention complexity** : L'attention du Transformer est O(n²) en longueur de séquence, limitant l'applicabilité à de très longues séquences. C'est pourquoi BERT pré-entraîne 90% des steps avec longueur 128 pour accélérer.

7. **Fine-tuning instable sur petits datasets** : Pour BERT_LARGE sur petits datasets, le fine-tuning peut être instable, nécessitant plusieurs random restarts et sélection du meilleur modèle sur Dev set. Ajoute de la complexité et du coût computationnel.

8. **Pas d'analyse d'interprétabilité détaillée** : Contrairement à ELMo qui analyse explicitement quelle couche capture quoi (syntaxe vs sémantique), BERT ne fournit pas d'analyse profonde des représentations apprises. Les visualisations d'attention existent mais l'interprétation reste difficile.

9. **Vocabulaire WordPiece fixe** : Le vocabulaire de 30,000 tokens est figé après pré-entraînement. Pour des domaines très spécialisés (médical, légal), des tokens out-of-vocabulary nécessitent une décomposition en sub-words, potentiellement sous-optimale.

10. **Pas d'évaluation multilingue dans le paper** : Toutes les expériences sont en anglais. Bien que Google publie plus tard multilingual BERT, le paper original ne démontre pas les capacités cross-lingual ou multilingues.

11. **Segment embeddings binaires A/B** : Le modèle supporte seulement 2 segments (A et B). Pour des tâches nécessitant plus de 2 phrases distinctes, des modifications architecturales seraient nécessaires.

12. **Pas de comparaison avec multi-task learning durant pré-entraînement** : Le pré-entraînement utilise seulement MLM + NSP. Des objectifs auxiliaires supplémentaires (e.g., NER, POS tagging) durant le pré-entraînement pourraient potentiellement améliorer les représentations.

13. **Dependency on pre-training data quality and biases** : BERT hérite des biais présents dans BooksCorpus et Wikipedia (biais de genre, race, culture). Le paper ne discute pas ces aspects éthiques importants.

14. **Pas de génération de texte native** : BERT est conçu pour la compréhension, pas la génération. Contrairement à GPT qui peut naturellement générer du texte, BERT nécessiterait des modifications significatives pour la génération (bien que des variantes comme MASS ou BART tenteront d'adresser cela).

15. **Feature-based approach légèrement inférieure** : Bien que BERT fonctionne en approche feature-based (Concat last 4 layers : -0.3 F1 sur NER), le fine-tuning reste légèrement meilleur. Pour certaines applications nécessitant des architectures non-Transformer, cette perte peut être significative.

## Liens utiles

- **PDF annoté**: [PDF annoté](../../papers/2026/bert.pdf)
- **Article**: [BERT: Pre-training of Deep Bidirectional Transformers for Language Understanding (PDF)](https://arxiv.org/abs/1810.04805)
- **ArXiv**: [https://arxiv.org/abs/1810.04805](https://arxiv.org/abs/1810.04805)
- **Code et modèles pré-entraînés**: [https://github.com/google-research/bert](https://github.com/google-research/bert)

## Notes perso

BERT représente une évolution majeure par rapport à ELMo et GPT-1 en combinant les avantages des deux approches :

**vs ELMo** :
- **Bidirectionnalité profonde** : BERT fusionne véritablement contextes gauche et droit dans toutes les couches via self-attention bidirectionnelle, contrairement à ELMo qui concatène des LSTMs indépendants entraînés séparément
- **Architecture unifiée** : Fine-tuning end-to-end avec modifications minimales, vs feature-based approach nécessitant des architectures task-specific pour ELMo
- **Transformer vs LSTM** : Meilleure capture des dépendances long-range et parallélisation massive
- **Performance** : BERT surpasse systématiquement ELMo sur tous les benchmarks (e.g., SQuAD : 93.2 F1 vs 85.8 F1)

**vs GPT-1** :
- **Bidirectionnel vs Unidirectionnel** : La différence clé. GPT utilise masked self-attention (left-to-right), BERT utilise bidirectional self-attention
- **MLM vs LM** : L'objectif MLM permet la bidirectionnalité en masquant des tokens, évitant le problème d'information leakage
- **NSP task** : BERT ajoute NSP pour apprendre les relations inter-phrases, crucial pour QA et NLI
- **Données** : BERT utilise BooksCorpus + Wikipedia (3.3B mots) vs seulement BooksCorpus (800M mots) pour GPT
- **Performance** : BERT surpasse largement GPT sur GLUE (+7.0%), SQuAD, et toutes les tâches testées

**Points communs ELMo/GPT/BERT** :
- Tous utilisent le transfer learning : pré-entraînement non supervisé sur large corpus, puis adaptation aux tâches cibles
- Tous démontrent l'importance d'un pré-entraînement riche pour améliorer les performances downstream

**Impact** :
BERT a lancé une révolution dans le NLP avec des modèles toujours plus grands (RoBERTa, ALBERT, ELECTRA, T5, GPT-2/3) qui suivront ses principes tout en tentant d'améliorer ses limitations.
