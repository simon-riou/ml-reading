---
title: "Language Models are Few-Shot Learners"
authors: ["Tom B. Brown", "Benjamin Mann", "Nick Ryder", "Melanie Subbiah", "Jared Kaplan", "Prafulla Dhariwal", "Arvind Neelakantan", "Pranav Shyam", "Girish Sastry", "Amanda Askell", "Sandhini Agarwal", "Ariel Herbert-Voss", "Gretchen Krueger", "Tom Henighan", "Rewon Child", "Aditya Ramesh", "Daniel M. Ziegler", "Jeffrey Wu", "Clemens Winter", "Christopher Hesse", "Mark Chen", "Eric Sigler", "Mateusz Litwin", "Scott Gray", "Benjamin Chess", "Jack Clark", "Christopher Berner", "Sam McCandlish", "Alec Radford", "Ilya Sutskever", "Dario Amodei"]
venue: "NeurIPS"
year: 2020
citekey: brown2020languagemodelsfewshotlearners
links:
  pdf: "https://arxiv.org/abs/2005.14165"
  code: ""
  project: "https://openai.com/blog/gpt-3-apps/"
pdf_annotated: "../../papers/2026/gpt3.pdf"
tags:
  areas: ["NLP", "Language Modeling", "Transfer Learning", "Few-shot Learning", "In-context Learning"]
  methods: ["GPT-3", "Transformer Decoder", "Few-shot Learning", "In-context Learning", "Scaling Laws"]
  tasks: ["Language Modeling", "Question Answering", "Machine Translation", "Reading Comprehension", "Arithmetic", "Common Sense Reasoning"]
  status: "deep-read"
replication:
  repo: ""
  data: "Common Crawl, WebText2, Books1, Books2, Wikipedia"
  env: ""
---

## TL;DR

GPT-3 démontre que le **scaling massif** des modèles de langage (jusqu'à **175 milliards de paramètres**, 100x GPT-2) permet un **apprentissage few-shot performant sans fine-tuning**. L'idée clé est l'**in-context learning** : le modèle apprend à effectuer des tâches uniquement à partir d'exemples fournis dans le contexte (prompt), sans mise à jour des poids. GPT-3 est évalué sur plus de **42 benchmarks** dans trois régimes : zero-shot, one-shot et few-shot. Résultats majeurs : **86.4% sur LAMBADA** (vs 76.6% SOTA supervisé), **71.2% sur TriviaQA** (SOTA closed-book), performances compétitives sur SuperGLUE égalant BERT-Large fine-tuné, et capacités émergentes sur des tâches de raisonnement (arithmétique 2-3 chiffres, word scrambling, utilisation de mots nouveaux). Le papier établit que les performances few-shot augmentent **plus rapidement que zero-shot** avec la taille du modèle, suggérant que les grands modèles sont meilleurs pour l'in-context learning. GPT-3 a été entraîné sur **~300 milliards de tokens** (Common Crawl filtré + WebText2 + Books + Wikipedia) avec un coût estimé à **>4 millions de dollars**.

## Contexte

En 2020, le paradigme dominant en NLP reste le **pré-entraînement + fine-tuning supervisé** (BERT, RoBERTa, T5). Cependant, plusieurs limitations persistent :

**Dépendance aux données annotées** : Le fine-tuning nécessite des milliers à des centaines de milliers d'exemples annotés par tâche. Créer ces datasets est coûteux et ne scale pas.

**Généralisation hors distribution limitée** : Les modèles fine-tunés performent bien sur leur distribution d'entraînement mais sont fragiles face aux variations. Les humains, eux, peuvent apprendre de nouvelles tâches à partir de quelques exemples ou d'instructions.

**Potentiel de mémorisation** : Avec des modèles de plus en plus grands et des datasets task-specific relativement petits, le risque de mémorisation du train set augmente, créant une surestimation des capacités de généralisation.

**GPT-2 et le zero-shot** : Le papier GPT-2 (Radford et al., 2019) a montré que les grands modèles de langage peuvent effectuer des tâches en zero-shot, mais les performances restaient bien en deçà du fine-tuning supervisé sur la plupart des tâches.

**Méta-learning et in-context learning** : Des travaux récents suggèrent que les Transformers peuvent apprendre de nouvelles tâches "dans le contexte" - en conditionnant sur des exemples plutôt qu'en mettant à jour les poids. Cette capacité pourrait émerger naturellement du pré-entraînement sur du texte diversifié.

**Scaling laws** : Des travaux parallèles (Kaplan et al., 2020) ont établi des lois de scaling prévisibles reliant la taille du modèle, la taille du dataset et le compute aux performances. Cela motive l'exploration de modèles beaucoup plus grands.

**Hypothèse centrale** : En scalant suffisamment les modèles de langage, les capacités de few-shot learning (in-context learning) pourraient devenir assez performantes pour rivaliser avec le fine-tuning supervisé, sans nécessiter de mise à jour des poids.

## Idées clés

1. **In-context learning comme paradigme alternatif au fine-tuning** : Au lieu de mettre à jour les poids du modèle sur des données task-specific, fournir des exemples de la tâche directement dans le prompt. Le modèle "apprend" la tâche à l'inférence, pas à l'entraînement. Cela permet une adaptation instantanée à de nouvelles tâches sans collecte de données ni entraînement supplémentaire.

2. **Trois régimes d'évaluation sans gradient updates** :
   - **Zero-shot** : Seulement une description de la tâche, sans exemples
   - **One-shot** : Un seul exemple de démonstration
   - **Few-shot** : K exemples (typiquement 10-100, limité par la fenêtre de contexte)

   Aucun de ces régimes n'implique de mise à jour des poids - tout est dans le conditionnement textuel.

3. **Scaling massif comme levier principal** : GPT-3 (175B paramètres) est 100x plus grand que GPT-2 (1.5B) et 10x plus grand que le plus grand modèle dense précédent. L'hypothèse est que le scaling améliore non seulement les performances de base mais aussi la capacité d'in-context learning.

4. **8 tailles de modèles pour étudier le scaling** : De 125M à 175B paramètres, permettant d'analyser comment les capacités zero-shot, one-shot et few-shot évoluent avec la taille.

5. **Dataset d'entraînement massif et diversifié** : ~300B tokens vus pendant l'entraînement, provenant de :
   - **Common Crawl filtré** (410B tokens, 60% du poids)
   - **WebText2** (19B tokens, 22%)
   - **Books1 & Books2** (67B tokens, 16%)
   - **Wikipedia** (3B tokens, 3%)

   Le downsampling de Common Crawl et le sursampling des sources de haute qualité améliore la qualité.

6. **Few-shot learning scale mieux que zero-shot** : Observation clé du papier : les gains en few-shot augmentent plus rapidement avec la taille du modèle que les gains en zero-shot. Les grands modèles sont meilleurs pour exploiter les exemples in-context.

7. **Capacités émergentes** : Certaines capacités n'apparaissent qu'aux grandes échelles :
   - Arithmétique (addition/soustraction 2-3 chiffres)
   - Word scrambling (anagrammes, insertion de caractères)
   - Utilisation de mots nouveaux après une seule définition
   - Génération de code à partir de descriptions
   - Correction grammaticale

8. **Benchmark diversifié (42+ tâches)** : Évaluation systématique sur language modeling, closed-book QA, translation, SuperGLUE, reading comprehension, common sense reasoning, et tâches synthétiques (arithmétique, anagrammes).

## Méthode

### Architecture

**GPT-3 utilise la même architecture que GPT-2** avec scaling :
- Transformer decoder-only avec masked self-attention
- Pre-norm (LayerNorm à l'entrée de chaque sub-block)
- Vocabulaire de **50,257 tokens** (BPE)
- Fenêtre de contexte de **2048 tokens**
- Alternance d'attention dense et sparse (comme Sparse Transformer)

**8 tailles de modèles** :

| Nom | Paramètres | Couches | d_model | Heads | d_head | Batch size |
|-----|------------|---------|---------|-------|--------|------------|
| GPT-3 Small | 125M | 12 | 768 | 12 | 64 | 0.5M |
| GPT-3 Medium | 350M | 24 | 1024 | 16 | 64 | 0.5M |
| GPT-3 Large | 760M | 24 | 1536 | 16 | 96 | 0.5M |
| GPT-3 XL | 1.3B | 24 | 2048 | 24 | 128 | 1M |
| GPT-3 2.7B | 2.7B | 32 | 2560 | 32 | 80 | 1M |
| GPT-3 6.7B | 6.7B | 32 | 4096 | 32 | 128 | 2M |
| GPT-3 13B | 13B | 40 | 5140 | 40 | 128 | 2M |
| **GPT-3 175B** | **175B** | **96** | **12288** | **96** | **128** | **3.2M** |

**GPT-3 175B spécifiquement** :
- 96 couches Transformer
- Dimension 12,288
- 96 heads d'attention × 128 dimensions chacun
- FFN hidden size = 4 × 12,288 = 49,152
- ~173B paramètres dans les couches Transformer
- ~0.6B paramètres dans les embeddings de tokens
- ~0.025B paramètres dans les embeddings de position

### Dataset d'entraînement

**5 sources, pondérées différemment** :

| Dataset | Tokens | Poids dans le mix |
|---------|--------|-------------------|
| Common Crawl (filtré) | 410B | 60% |
| WebText2 | 19B | 22% |
| Books1 | 12B | 8% |
| Books2 | 55B | 8% |
| Wikipedia | 3B | 3% |

**Total** : ~499B tokens disponibles, **~300B tokens vus** pendant l'entraînement (certaines sources vues plusieurs fois, d'autres sous-échantillonnées).

**Filtrage de Common Crawl** :
- Entraînement d'un classifieur pour distinguer contenu de haute qualité (similaire à WebText) vs bas qualité
- Fuzzy deduplication avec LSH
- Ajout de sources connues de haute qualité (WebText, Books, Wikipedia)

### Entraînement

- **Tous les modèles entraînés pour 300B tokens**
- Optimiseur Adam
- Learning rate warmup puis decay cosine
- Batch size progressif (augmente pendant l'entraînement pour les gros modèles)
- **V100 GPUs** sur un cluster haute bande passante
- Partitionnement du modèle en profondeur et en largeur pour minimiser la communication
- **Coût estimé : >4 millions de dollars** (à 2020)
- **Temps d'entraînement** : Non spécifié, mais estimé à plusieurs semaines/mois

### Évaluation

**Protocole few-shot** :
1. Sélectionner K exemples du train set (ou écrire manuellement)
2. Formater comme : `[exemple 1]\n[exemple 2]\n...[exemple K]\n[test input]`
3. Générer ou scorer les continuations possibles
4. **Aucune mise à jour de gradient** - le modèle ne voit les exemples qu'à l'inférence

**Métriques** :
- Accuracy pour classification/QA
- Perplexité pour language modeling
- BLEU pour traduction
- F1 pour certaines tâches de compréhension

## Résultats

### Language Modeling

| Dataset | GPT-3 Zero-shot | GPT-3 Few-shot | SOTA précédent |
|---------|-----------------|----------------|----------------|
| **LAMBADA** (acc) | 76.2% | **86.4%** | 68.0% (GPT-2) |
| **LAMBADA** (ppl) | 3.0 | **1.92** | 8.63 (GPT-2) |
| PTB (ppl) | **20.5** | - | 35.76 (GPT-2) |
| WikiText-103 (ppl) | **-** | **10.87** | 17.48 (GPT-2) |

**LAMBADA** : Amélioration massive de +18% sur le SOTA. La tâche teste la compréhension de dépendances longues - GPT-3 excelle grâce à sa grande fenêtre de contexte et sa capacité.

### Question Answering (Closed-book)

| Dataset | Zero-shot | One-shot | Few-shot | SOTA fine-tuné |
|---------|-----------|----------|----------|----------------|
| **TriviaQA** | 64.3% | 68.0% | **71.2%** | 68.0% (T5-11B) |
| **Natural Questions** | 14.6% | 23.0% | **29.9%** | 36.6% (T5-11B) |
| **WebQuestions** | 14.4% | 25.3% | **41.5%** | 44.1% (T5-11B) |

GPT-3 atteint le **SOTA closed-book sur TriviaQA** sans aucun fine-tuning, dépassant T5-11B fine-tuné.

### SuperGLUE

| Tâche | GPT-3 Few-shot | BERT-Large fine-tuné | SOTA |
|-------|----------------|----------------------|------|
| BoolQ | 76.4% | 77.4% | 91.0% |
| CB | 75.6% | 75.7% | 93.9% |
| COPA | **92.0%** | 70.6% | 94.8% |
| MultiRC | 72.9% | 70.0% | 88.1% |
| ReCoRD | **90.2%** | 72.0% | 94.1% |
| RTE | 69.0% | 70.4% | 92.5% |
| WiC | 49.4% | 69.6% | 76.1% |
| WSC | 80.1% | 64.3% | 93.8% |

- GPT-3 **égale ou dépasse BERT-Large fine-tuné sur 4/8 tâches**
- Excellent sur COPA (92%) et ReCoRD (90.2%)
- Faible sur WiC (49.4%) - proche du hasard

### Traduction (Few-shot)

| Direction | GPT-3 Few-shot | SOTA supervisé |
|-----------|----------------|----------------|
| Fr→En | **32.6 BLEU** | 40.0 |
| De→En | **40.6 BLEU** | 42.5 |
| Ro→En | **38.3 BLEU** | 40.0 |
| En→Fr | 19.2 BLEU | 45.0 |
| En→De | 24.3 BLEU | 42.0 |
| En→Ro | 21.0 BLEU | 38.0 |

- **Excellent en traduction vers l'anglais** (proche du SOTA supervisé)
- **Faible en traduction depuis l'anglais** (LM anglophone favorise la génération anglaise)

### Tâches de raisonnement / Capacités émergentes

**Arithmétique** (Few-shot, modèle 175B) :
| Opération | Accuracy |
|-----------|----------|
| Addition 2 chiffres | **100%** |
| Soustraction 2 chiffres | **98.9%** |
| Addition 3 chiffres | **80.2%** |
| Addition 4 chiffres | **25.5%** |
| Multiplication 2×1 chiffres | **29.2%** |

- Saut significatif de performance entre 13B et 175B
- Suggère des capacités de calcul émergentes aux grandes échelles

**Word scrambling** (Few-shot) :
| Tâche | GPT-3 175B |
|-------|------------|
| Cycle letters (words) | 74.2% |
| Random insert (chars) | 69.2% |
| Anagrams (1 move) | 70.6% |
| Anagrams (2 moves) | 38.5% |

**Utilisation de mots nouveaux** : GPT-3 peut utiliser un mot défini une seule fois dans une phrase cohérente (évaluation qualitative).

### Scaling des capacités

**Observation centrale** : Les performances few-shot augmentent plus rapidement avec la taille du modèle que les performances zero-shot.

- **Zero-shot** : Amélioration régulière et graduelle
- **Few-shot** : Amélioration rapide, surtout aux grandes échelles
- **Gap few-shot/zero-shot** : Augmente avec la taille du modèle

Cela suggère que les grands modèles sont meilleurs pour **l'in-context learning** - ils exploitent mieux les exemples fournis.

## Limites

### Limites architecturales et méthodologiques

1. **Architecture decoder-only unidirectionnelle** : Comme GPT-1/2, GPT-3 ne peut pas voir le contexte futur. BERT et les modèles bidirectionnels surpassent souvent sur les tâches de compréhension qui bénéficient du contexte bidirectionnel.

2. **Fenêtre de contexte limitée (2048 tokens)** : Limite le nombre d'exemples few-shot possibles, surtout pour les tâches avec de longs exemples. Impossible de traiter des documents longs en une fois.

3. **Pas de grounding / pas de mise à jour** : Le modèle ne peut pas accéder à des informations factuelles à jour (frozen au moment du pré-entraînement) ni corriger ses erreurs via feedback.

4. **Échec sur certaines tâches NLU** : Performances proches du hasard sur WiC (word-in-context), faibles sur ANLI, et sous-performant sur des benchmarks de lecture comme QuAC et RACE.

5. **Incohérence sur les longs textes** : GPT-3 peut perdre la cohérence sur de longues générations, répéter des informations, ou contredire des éléments antérieurs.

6. **Pas de raisonnement symbolique robuste** : L'arithmétique échoue au-delà de 3 chiffres. Pas de vrai raisonnement compositionnel.

### Biais et problèmes éthiques

7. **Biais de genre** : Le modèle associe les femmes à des termes d'apparence ("beautiful", "gorgeous") et les hommes à une plus grande diversité de descripteurs. Les professions sont genrées de manière stéréotypée.

8. **Biais raciaux** : Des sentiments différents sont associés à différentes races, avec une association négative plus forte pour certains groupes. Les détails des associations n'ont pas été publiés complètement.

9. **Biais religieux** : "Terrorism" associé à l'Islam, "ignorant" au Christianisme, stéréotypes reproduits pour d'autres religions. Le modèle reflète et amplifie les biais d'Internet.

10. **Génération de désinformation** : GPT-3 peut générer des articles de news que les humains ont du mal à distinguer d'articles réels. Risque de création de désinformation à grande échelle.

11. **Contenu toxique** : Peut générer du contenu offensant, haineux, ou inapproprié si prompté (ou parfois spontanément).

### Limites pratiques

12. **Coût prohibitif** : Entraînement estimé >4M$, inférence très coûteuse (175B paramètres). Inaccessible à la plupart des chercheurs et organisations.

13. **Pas de code ou poids publics** : Modèle accessible uniquement via API payante d'OpenAI. Reproductibilité limitée. Les datasets Books1/Books2 ne sont pas publics.

14. **Consommation énergétique** : L'entraînement et l'inférence de modèles de cette taille ont une empreinte carbone significative.

15. **Sensibilité au prompting** : Les performances peuvent varier significativement selon la formulation exacte du prompt, le choix des exemples, leur ordre, etc. Pas de méthode systématique pour optimiser les prompts.

16. **Scaling comme seule innovation** : L'architecture est essentiellement identique à GPT-2. Les gains viennent principalement du scaling, pas d'innovations architecturales.

## Liens utiles

- **PDF annoté**: [PDF annoté](../../papers/2026/gpt3.pdf)
- **Article**: [Language Models are Few-Shot Learners (arXiv)](https://arxiv.org/abs/2005.14165)
- **NeurIPS 2020**: [Paper](https://papers.nips.cc/paper/2020/hash/1457c0d6bfcb4967418bfb8ac142f64a-Abstract.html)
- **OpenAI Blog**: [GPT-3](https://openai.com/blog/gpt-3-apps/)

## Notes perso

### Évolution GPT-1 → GPT-2 → GPT-3

| Aspect | GPT-1 (2018) | GPT-2 (2019) | GPT-3 (2020) |
|--------|--------------|--------------|--------------|
| Paramètres | 117M | 1.5B | 175B |
| Scaling | 1x | 13x | 1500x |
| Contexte | 512 | 1024 | 2048 |
| Dataset | BooksCorpus | WebText | CC + Books + Wiki |
| Paradigme | Fine-tuning | Zero-shot | Few-shot |
| Contribution | Pre-train + FT | Multitask LM | In-context learning |

### Contributions conceptuelles majeures

1. **In-context learning comme capacité émergente** : L'idée que les modèles peuvent "apprendre" de nouvelles tâches à l'inférence, uniquement via le contexte, sans mise à jour des poids. C'est un changement de paradigme par rapport au fine-tuning.

2. **Démocratisation (relative) de l'adaptation** : Plus besoin de données annotées ni de compute d'entraînement pour adapter un modèle à une nouvelle tâche. Juste quelques exemples dans le prompt.

3. **Scaling comme recherche** : Le papier établit que simplement scaler les modèles peut débloquer des capacités qualitativement nouvelles. C'est le début de l'ère "scaling laws" et de la course aux gros modèles.

4. **Capacités émergentes** : Certaines capacités (arithmétique, word scrambling) n'apparaissent qu'aux très grandes échelles. Suggère que le scaling peut produire des discontinuités qualitatives.

### Points remarquables du papier

- Recommandé pour le **best paper award** par un reviewer NeurIPS
- Le papier documente honnêtement les **biais et limitations** (section entière dédiée)
- Premier modèle à générer du texte "indistinguishable" d'humains selon certaines métriques

### Impact et legacy

GPT-3 a lancé :
- L'industrie des "prompt engineering"
- L'API économie (accès payant aux LLMs)
- La course aux modèles >100B paramètres
- Le débat sur les risques de désinformation par IA
- Les applications grand public de LLMs (Copilot, etc.)

### Critiques importantes

- **"Scaling is not understanding"** : Les critiques argumentent que GPT-3 ne "comprend" pas vraiment le langage, il fait du pattern matching sophistiqué. Les échecs sur certaines tâches (ANLI, WiC) révèlent des failles de raisonnement.

- **Stochastic parrots** : Le papier de Bender et al. (2021) critique l'approche "bigger is better" et soulève les problèmes environnementaux, de biais, et d'illusion de compréhension.

- **Reproductibilité** : Sans accès aux poids et avec un coût de reproduction >4M$, la science reproductible est impossible. Seul OpenAI peut vraiment étudier GPT-3.

Ce papier marque un tournant : le passage de la recherche académique sur les LLMs à une ère industrielle où seuls les labs avec des ressources massives peuvent explorer la frontière.
