---
title: "Exploring the Limits of Transfer Learning with a Unified Text-to-Text Transformer"
authors: ["Colin Raffel", "Noam Shazeer", "Adam Roberts", "Katherine Lee", "Sharan Narang", "Michael Matena", "Yanqi Zhou", "Wei Li", "Peter J. Liu"]
venue: "JMLR"
year: 2020
citekey: Raffel2020ExploringTL
links:
  pdf: "https://arxiv.org/abs/1910.10683"
  code: "https://github.com/google-research/text-to-text-transfer-transformer"
  project: "https://research.google/blog/exploring-transfer-learning-with-t5-the-text-to-text-transfer-transformer/"
pdf_annotated: "../../papers/2026/t5.pdf"
tags:
  areas: ["NLP", "Transfer Learning", "Language Modeling"]
  methods: ["T5", "Text-to-Text Framework", "Encoder-Decoder Transformer", "Span Corruption"]
  tasks: ["GLUE", "SuperGLUE", "SQuAD", "CNN/DM", "Translation", "Summarization", "Question Answering"]
  status: "deep-read"
replication:
  repo: "https://github.com/google-research/text-to-text-transfer-transformer"
  data: "C4 (Colossal Clean Crawled Corpus)"
  env: ""
---

## TL;DR

T5 (Text-to-Text Transfer Transformer) unifie toutes les tâches NLP dans un **framework text-to-text** : chaque tâche (traduction, résumé, classification, QA) est reformulée comme "générer du texte à partir de texte" via des préfixes textuels (ex: "translate English to German: ...", "summarize: ..."). Cette approche permet d'utiliser la **même architecture encoder-decoder Transformer, le même objectif (language modeling), et les mêmes hyperparamètres** pour toutes les tâches. Le papier présente une **étude empirique systématique à large échelle** comparant architectures (encoder-decoder vs decoder-only), objectifs de pré-entraînement (denoising vs language modeling), datasets non supervisés, et stratégies de transfer. Contributions clés : création du **C4 dataset** (750GB de texte web nettoyé via Common Crawl), et démonstration que l'objectif **span corruption denoising** (masquer des spans de tokens aléatoires) surpasse les alternatives. Le modèle **T5-11B** (11 milliards de paramètres) atteint le **state-of-the-art sur GLUE, SuperGLUE (88.9, proche du 89.8 humain), SQuAD, et CNN/DM**. L'étude révèle que les **encoder-decoder surpassent les decoder-only**, que **scaler le modèle et les données améliore log-linéairement les performances**, et que le pré-entraînement multitâche supervisé n'apporte pas de gains significatifs vs l'approche non supervisée simple.

## Contexte

En 2019-2020, le transfer learning en NLP connaît une explosion avec BERT, GPT-2, XLNet, RoBERTa, mais chaque approche utilise des **architectures, objectifs, et datasets différents**, rendant difficile la comparaison et la compréhension de ce qui fonctionne vraiment :

**Fragmentation méthodologique** : BERT utilise un encoder bidirectionnel avec MLM+NSP, GPT-2 un decoder unidirectionnel avec LM, XLNet un autoregressive avec permutations, ELMo des BiLSTMs, etc. Impossible de déterminer si les différences de performance viennent de l'architecture, de l'objectif, du dataset, ou d'autres facteurs.

**Formats de tâches hétérogènes** : Les tâches NLP traditionnelles ont des formats incompatibles : classification (label unique), QA (span extraction), traduction (séquence), NER (tagging), etc. Chaque approche nécessite des adaptations task-specific différentes.

**Manque d'études systématiques** : Les papiers comparent généralement 2-3 variantes et reportent un seul score final. Pas de vue d'ensemble sur l'espace des choix de design (architectures × objectifs × datasets × scaling × transfer strategies = milliers de combinaisons possibles).

**Coût computationnel prohibitif** : Pré-entraîner un seul modèle BERT-large coûte des dizaines de milliers de dollars en TPUs. Explorer systématiquement l'espace de design nécessite des ressources massives que seules les grandes entreprises peuvent se permettre.

**Questions non résolues** :
- Encoder-decoder (BERT) vs decoder-only (GPT) : lequel est meilleur ?
- Quel objectif de pré-entraînement : MLM, LM, denoising, deshuffling ?
- Quel dataset : Wikipedia, Common Crawl, livres, web filtré ?
- Quelle stratégie : adapter des couches, fine-tuning complet, multi-task ?
- Combien de données non supervisées suffisent ? Combien de paramètres ?

**Motivation pour T5** : Google dispose des ressources pour mener une **étude systématique exhaustive** testant des dizaines de variantes sur plusieurs benchmarks, afin d'identifier les meilleures pratiques et de créer un modèle unifié state-of-the-art. Le framework text-to-text permet d'évaluer toutes les tâches avec la même infrastructure.

## Idées clés

1. **Framework text-to-text unifié** : Reformuler toutes les tâches NLP comme "consommer du texte en entrée, produire du texte en sortie". Exemples : traduction = "translate English to German: That is good. → Das ist gut.", classification = "sst2 sentence: This movie is great! → positive", QA = "question: ... context: ... → answer span", résumé = "summarize: article... → summary...". Avantages : même architecture, même loss (likelihood du texte cible), même décoding, même évaluation directe du texte généré.

2. **Étude empirique systématique à large échelle** : Comparer de manière contrôlée des dizaines de variantes sur les dimensions clés : architectures (encoder-decoder, decoder-only, prefix LM), objectifs (BERT-style MLM, span corruption avec différentes longueurs, deshuffling, etc.), datasets (C4, Wikipedia, Common Crawl filtré), stratégies de transfer (fine-tuning, adapter layers, gradual unfreezing), et scaling (taille modèle, taille dataset, durée entraînement). Toutes les expériences utilisent la même infrastructure et les mêmes benchmarks (GLUE, SuperGLUE, SQuAD, CNN/DM).

3. **C4 : Colossal Clean Crawled Corpus** : Créer un nouveau dataset massif en filtrant Common Crawl d'avril 2019 avec des heuristiques de qualité : ne garder que les phrases se terminant par ponctuation, supprimer les pages avec mots grossiers, déduplication au niveau des lignes, ne conserver que l'anglais détecté avec langdetect. Résultat : **~750GB de texte anglais propre** (vs 2.5GB Wikipedia, 40GB WebText GPT-2). Public et reproductible contrairement à WebText.

4. **Span corruption denoising comme objectif optimal** : Parmi tous les objectifs testés (MLM, prefix LM, deshuffling, mass masking), le **span corruption** fonctionne le mieux : masquer des spans contiguës de tokens (longueur moyenne 3) représentant 15% du texte, remplacer chaque span par un sentinel token unique (`<X>`, `<Y>`, etc.), et entraîner le modèle à reconstruire seulement les spans masqués (pas le texte complet). Cela combine efficacité computationnelle (prédire seulement 15% des tokens) et capacité de débruitage.

5. **Encoder-decoder supérieur aux autres architectures** : Comparaison contrôlée sur plusieurs tâches montre que l'architecture encoder-decoder (style BERT/BART) surpasse le decoder-only (style GPT) et le prefix LM. L'encoder bidirectionnel capture mieux le contexte d'entrée, tandis que le decoder autoregressive génère la sortie. Le partage de paramètres entre encoder et decoder réduit le nombre total de paramètres tout en maintenant les performances.

6. **Scaling laws empiriques** : Les performances s'améliorent **log-linéairement** avec (1) la taille du modèle (60M → 11B paramètres), (2) la taille du dataset de pré-entraînement (10M → 1T tokens), et (3) l'ensemble training (plus de steps). Mais il existe des rendements décroissants : doubler le modèle/dataset donne des gains de plus en plus faibles. L'étude identifie le point optimal : T5-11B pré-entraîné sur C4 complet (~1T tokens).

7. **Pré-entraînement multitâche supervisé n'aide pas** : Ajouter des tâches supervisées (GLUE, SQuAD, etc.) durant le pré-entraînement (en plus du denoising non supervisé) ne donne **aucun gain significatif** vs pré-entraînement non supervisé seul suivi de fine-tuning. Suggère que l'objectif de denoising capture déjà les structures linguistiques nécessaires.

8. **Fine-tuning complet > approches légères** : Comparaison des stratégies de transfer : (1) fine-tuning de tous les paramètres, (2) adapter layers (ajouter des couches denses dans chaque bloc Transformer), (3) gradual unfreezing (dégeler les couches progressivement). Résultat : **fine-tuning complet de tous les paramètres surpasse les alternatives**, même sur petits datasets. Les approches légères économisent du stockage (1 set de poids au lieu de 1 par tâche) mais sacrifient les performances.

## Méthode

### Architecture du modèle

**Transformer encoder-decoder** basé sur Vaswani et al. (2017) avec modifications mineures :
- **Simplification de LayerNorm** : Suppression du biais additif, placement de LayerNorm en dehors du residual path (pre-norm)
- **Positional embeddings relatifs** : Utilise relative positional embeddings (shared buckets) au lieu d'embeddings absolus appris
- **Dropout** : Appliqué partout (attention, feed-forward, skip connections)

**Tailles de modèles évaluées** :

| Nom | Paramètres | Couches | d_model | d_ff | Heads |
|-----|-----------|---------|---------|------|-------|
| T5-Small | 60M | 6 | 512 | 2048 | 8 |
| T5-Base | 220M | 12 | 768 | 3072 | 12 |
| T5-Large | 770M | 24 | 1024 | 4096 | 16 |
| T5-3B | 3B | 24 | 1024 | 16384 | 32 |
| **T5-11B** | **11B** | 24 | 1024 | 65536 | 128 |

### Dataset C4 (Colossal Clean Crawled Corpus)

**Construction** :
1. Partir de Common Crawl d'avril 2019 (~20TB de texte web)
2. Filtrer pour ne garder que les pages anglaises (langdetect avec probabilité >0.99)
3. Supprimer lignes ne se terminant pas par ponctuation de phrase
4. Rejeter pages contenant des mots de la liste "Dirty, Naughty, Obscene, and Otherwise Bad Words"
5. Supprimer lignes contenant le mot "JavaScript" (artefacts de pages mal extraites)
6. Déduplication au niveau des lignes (si 3+ phrases identiques apparaissent, garder 1 seule occurrence)
7. Supprimer tout code détecté

**Résultat** : ~750GB de texte anglais propre (~1 trillion de tokens)

### Objectif de pré-entraînement : Span Corruption

**Procédure** :
1. Sélectionner aléatoirement 15% des tokens pour corruption
2. Grouper les tokens consécutifs sélectionnés en spans
3. Remplacer chaque span par un unique sentinel token (`<extra_id_0>`, `<extra_id_1>`, etc.)
4. **Input** : Texte avec spans remplacés par sentinels
5. **Target** : Seulement les spans masqués, séparés par les mêmes sentinels, suivi de `<extra_id_final>`

**Exemple** :
- Original : "Thank you for inviting me to your party last week."
- Corrompu (input) : "Thank you `<X>` me to your party `<Y>` week."
- Cible (target) : "`<X>` for inviting `<Y>` last `<Z>`"

**Longueur moyenne des spans** : 3 tokens (déterminé par étude d'ablation)

### Pré-entraînement

- **Dataset** : C4 (~1 trillion tokens)
- **Séquence length** : 512 tokens (input) + 512 tokens (target)
- **Batch size** : 2^11 = 2048 sequences
- **Steps** : 2^19 = ~524k steps (~1T tokens vus, soit 1 epoch de C4)
- **Optimiseur** : AdaFactor (variant d'Adam avec factorized second moments pour économie mémoire)
- **Learning rate** : Inverse square root schedule avec warmup de 10k steps
- **Dropout** : 0.1
- **Vocabulaire** : SentencePiece avec 32,000 wordpieces
- **Coût** : T5-11B pré-entraîné en ~1 semaine sur 1024 TPU v3 cores

### Fine-tuning

**Procédure générale** :
1. Reformuler la tâche en format text-to-text avec préfixe task-specific
2. Initialiser avec poids pré-entraînés T5
3. Fine-tuner tous les paramètres end-to-end
4. Utiliser les mêmes hyperparamètres pour toutes les tâches (sauf learning rate)

**Format par type de tâche** :

**Classification (GLUE, SuperGLUE)** :
- Input : "mnli premise: ... hypothesis: ... → entailment/neutral/contradiction"
- Output : Le texte du label (pas un logit)

**Question Answering (SQuAD)** :
- Input : "question: What is...? context: The text... →"
- Output : Span de réponse extrait

**Summarization (CNN/DM)** :
- Input : "summarize: article text..."
- Output : Résumé généré

**Translation (WMT)** :
- Input : "translate English to German: That is good."
- Output : "Das ist gut."

**Hyperparamètres fine-tuning** :
- **Learning rate** : Grid search sur {1e-4, 3e-4, 1e-3, 3e-3} selon la tâche
- **Steps** : 2^18 = ~262k steps maximum (early stopping sur validation)
- **Batch size** : 128 sequences
- **Dropout** : 0.1
- **Checkpoint** : Sauvegarder toutes les 5k steps, sélectionner meilleur sur validation

## Résultats

**GLUE Benchmark** : T5-11B atteint **89.7 average score**, dépassant le précédent SOTA (88.4 de RoBERTa-large).

**SuperGLUE** : T5-11B atteint **89.3 average**, puis **88.9 en single-model** (vs 89.8 baseline humain, 84.6 RoBERTa). Proche des performances humaines.

**SQuAD v1.1** : T5-11B atteint **95.3 F1** en single model (vs 94.6 SOTA précédent).

**CNN/Daily Mail Summarization** : T5-11B atteint **43.52 ROUGE-2** (nouveau SOTA à l'époque).

**Études d'ablation clés** :

**Architectures** : Encoder-decoder > Decoder-only > Prefix LM sur la plupart des tâches. L'encoder-decoder avec partage de paramètres offre le meilleur compromis performance/paramètres.

**Objectifs de pré-entraînement** : Span corruption (mean length 3) performe le mieux. Masquer des spans > masquer des tokens individuels (BERT-style). Prédire seulement les spans masqués (pas tout le texte) > prédire la séquence complète.

**Dataset size** : Les performances s'améliorent log-linéairement avec la taille du dataset jusqu'à ~1T tokens, puis plateau. C4 complet (750GB) > subsets plus petits.

**Model size** : Scaling de 60M → 11B améliore continûment toutes les tâches. Rendements décroissants mais pas de plateau observé à 11B.

**Pré-entraînement multitâche supervisé** : Ajouter des tâches supervisées durant pré-entraînement n'aide **pas** vs denoising non supervisé seul + fine-tuning. Suggère que le denoising capture déjà les bonnes structures.

**Stratégies de fine-tuning** : Fine-tuning complet > Adapter layers > Gradual unfreezing. La flexibilité de tous les paramètres surpasse les approches économes en mémoire.

## Limites

**Coût computationnel massif** : T5-11B nécessite ~1 semaine sur 1024 TPU v3 cores (~$50k en cloud), rendant la reproduction inaccessible aux laboratoires académiques. Seules les grandes entreprises peuvent se permettre ces expériences.

**Architecture encoder-decoder plus lourde** : Nécessite encoder ET decoder pour l'inférence, doublant la mémoire et ralentissant vs decoder-only (GPT). Pour la génération pure, GPT-style peut être plus efficace.

**Objectif span corruption non optimal pour génération** : T5 est pré-entraîné pour reconstruire des spans masqués, pas pour génération autoregressif libre. GPT-2/3 peuvent mieux généraliser à la génération longue et créative.

**Dataset C4 contient des biais** : Le filtrage par liste de "bad words" peut introduire des biais (supprimer contenu LGBTQ+, discussions médicales, etc.). Common Crawl hérite des biais du web anglophone (majoritairement US/UK).

**Pas de multilingue** : T5 original est anglais uniquement. mT5 sera publié plus tard pour le multilingue, mais le papier n'explore pas les capacités cross-lingual.

**Étude limitée aux tâches supervisées** : Pas d'évaluation zero-shot ou few-shot comme GPT-2/3. Toutes les tâches nécessitent fine-tuning complet, limitant la flexibilité.

**Comparaisons parfois injustes** : T5-11B (11B params, 1T tokens) comparé à BERT-large (340M params, 3.3B tokens) ou RoBERTa (355M params, 160GB). Les gains viennent en partie du scale, pas seulement de l'approche text-to-text.

**Génération de texte évaluée seulement par métriques automatiques** : ROUGE, BLEU ne capturent pas la qualité, cohérence, factualité. Pas d'évaluation humaine détaillée.

**Format text-to-text peut être inefficace** : Pour classification binaire, générer "positive"/"negative" (5-8 tokens) vs prédire 1 logit est wasteful. Le format unifié sacrifie l'efficacité task-specific.

**Pas d'analyse d'interprétabilité** : Quelles représentations apprend T5 ? Comment les différentes couches fonctionnent ? Pas de visualisations ou d'analyses approfondies.

## Liens utiles

- **PDF annoté**: [PDF annoté](../../papers/2026/t5.pdf)
- **Article**: [Exploring the Limits of Transfer Learning with a Unified Text-to-Text Transformer (PDF)](https://arxiv.org/abs/1910.10683)
- **ArXiv**: [https://arxiv.org/abs/1910.10683](https://arxiv.org/abs/1910.10683)
- **Code et modèles**: [https://github.com/google-research/text-to-text-transfer-transformer](https://github.com/google-research/text-to-text-transfer-transformer)
- **Blog Google Research**: [https://research.google/blog/exploring-transfer-learning-with-t5-the-text-to-text-transfer-transformer/](https://research.google/blog/exploring-transfer-learning-with-t5-the-text-to-text-transfer-transformer/)
- **HuggingFace**: [https://huggingface.co/google-t5/t5-base](https://huggingface.co/google-t5/t5-base)

## Notes perso
