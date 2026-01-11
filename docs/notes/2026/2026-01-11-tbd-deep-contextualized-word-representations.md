---
title: "Deep contextualized word representations"
authors: ["Matthew E. Peters", "Mark Neumann", "Mohit Iyyer", "Matt Gardner", "Christopher Clark", "Kenton Lee", "Luke Zettlemoyer"]
venue: "NAACL"
year: 2018
citekey: peters2018deepcontextualizedwordrepresentations
links:
  pdf: "https://arxiv.org/abs/1802.05365"
  code: "http://allennlp.org/elmo"
  project: ""
pdf_annotated: "../../papers/2026/ELMo.pdf"
tags:
  areas: ["NLP", "Transfer Learning", "Language Modeling"]
  methods: ["ELMo", "BiLM", "Contextualized Embeddings", "Character CNN"]
  tasks: ["Question Answering", "Textual Entailment", "Semantic Role Labeling", "Coreference Resolution", "NER", "Sentiment Analysis"]
  status: "deep-read"
replication:
  repo: "http://allennlp.org/elmo"
  data: "1B Word Benchmark"
  env: ""
---

## TL;DR

ELMo (Embeddings from Language Models) introduit des représentations de mots **contextualisées profondes** apprises à partir d'un biLM (bidirectional Language Model) pré-entraîné. Contrairement aux embeddings traditionnels (Word2Vec, GloVe), chaque token reçoit une représentation qui est une **fonction de toute la phrase**, calculée comme une **combinaison linéaire apprise de toutes les couches internes du biLM**. ELMo améliore l'état de l'art sur 6 tâches NLP challengeantes avec des réductions d'erreur relative de 6-20%, en étant simplement ajouté aux modèles existants. L'analyse montre que les couches basses capturent la syntaxe (POS tagging) tandis que les couches hautes capturent la sémantique (word sense disambiguation).

## Contexte

En 2018, les word embeddings pré-entraînés (Word2Vec, GloVe) sont omniprésents en NLP, mais ont des limitations importantes :
- **Une seule représentation par mot** : Impossible de modéliser la polysémie ("play" comme verbe vs nom, sens sport vs théâtre)
- **Context-independent** : Le mot "bank" a la même représentation dans "river bank" et "bank account"
- **Pas de représentation profonde** : Les approches précédentes (CoVe de McCann et al. 2017) utilisent seulement la dernière couche d'un encodeur

Les tentatives précédentes pour apprendre des représentations contextuelles incluent context2vec (Melamud et al. 2016) et CoVe (encodeur MT supervisé), mais elles n'exploitent pas toute la profondeur du réseau. De plus, des travaux récents (Belinkov et al. 2017, Hashimoto et al. 2017) montrent que différentes couches de biRNNs profonds encodent différents types d'information (syntaxe vs sémantique).

## Idées clés

1. **Représentations contextualisées profondes** : Chaque token reçoit une représentation qui est fonction de toute la phrase d'entrée, pas seulement du mot isolé.

2. **Combinaison linéaire de toutes les couches du biLM** : Au lieu d'utiliser seulement la dernière couche, ELMo apprend une combinaison pondérée task-specific de **toutes les couches** (L=2 couches biLSTM + couche character) :
   ```
   ELMo_k^task = γ^task Σ(j=0 to L) s_j^task h_k,j^LM
   ```
   où s^task sont des poids softmax-normalized et γ^task un paramètre de scaling.

3. **Stratification de l'information linguistique** :
   - **Couches basses** : Encodent la syntaxe (97.3% acc POS tagging)
   - **Couches hautes** : Encodent la sémantique et la désambiguïsation du sens (69.0 F1 WSD)
   - Exposer toutes les couches permet au modèle downstream de sélectionner le type de supervision le plus utile

4. **Entraînement biLM à grande échelle** : Pré-entraînement sur ~30M phrases (1B Word Benchmark) avec un objectif de modèle de langage bidirectionnel conjoint (forward + backward).

5. **Représentation purement basée sur les caractères** : Utilise des CNNs sur caractères, permettant de générer des représentations pour **tous les tokens**, même hors vocabulaire.

6. **Facilité d'intégration** : ELMo peut être facilement ajouté à n'importe quel modèle supervisé existant en gelant les poids du biLM et en concaténant les vecteurs ELMo aux embeddings existants.

7. **Amélioration de l'efficacité des données** : Avec ELMo, un modèle entraîné sur 1% des données atteint les performances d'un modèle baseline sur 10% des données.

## Méthode

### Architecture du biLM

**Modèle de langage bidirectionnel** :
- **Forward LM** : p(t₁,...,tₙ) = ∏ p(tₖ | t₁,...,tₖ₋₁)
- **Backward LM** : p(t₁,...,tₙ) = ∏ p(tₖ | tₖ₊₁,...,tₙ)
- **Objectif joint** : Maximiser la log-vraisemblance forward + backward

**Architecture** :
- L = 2 couches biLSTM (4096 unités chacune avec projections de 512 dimensions)
- Connexion résiduelle entre les couches LSTM 1 et 2
- Couche token context-insensitive : 2048 filtres CNN sur caractères → 2 highway layers → projection à 512 dimensions
- Partage des paramètres pour la représentation token (Θₓ) et la couche softmax (Θₛ) entre directions forward/backward
- LSTMs séparés pour chaque direction

**Entraînement du biLM** :
- Corpus : 1B Word Benchmark (~30M phrases, Chelba et al. 2014)
- 10 epochs d'entraînement
- Perplexité moyenne forward/backward : 39.7
- Fine-tuning optionnel sur données domain-specific (réduit fortement la perplexité, e.g., SNLI : 72.1 → 16.8)

### Calcul des représentations ELMo

Pour chaque token tₖ, le biLM de L couches produit 2L+1 représentations :
- h_{k,0}^LM : couche caractères (context-independent)
- h_{k,j}^LM = [→h_{k,j}^LM ; ←h_{k,j}^LM] pour j=1,...,L : concaténation des états forward/backward

**ELMo task-specific** :
```
ELMo_k^task = γ^task Σ(j=0 to L) s_j^task h_{k,j}^LM
```
- s^task : poids softmax-normalized appris pour chaque tâche
- γ^task : paramètre de scaling (important pour l'optimisation)
- Layer normalization optionnelle avant pondération

### Intégration dans les modèles supervisés

**Procédure standard** :
1. Geler les poids du biLM pré-entraîné
2. Calculer les représentations ELMo pour chaque token
3. Concaténer ELMo avec les embeddings traditionnels : [xₖ ; ELMo_k^task]
4. Passer le tout dans le modèle task-specific (biRNN, CNN, etc.)

**Options avancées** :
- Ajouter ELMo aussi à la **sortie** du biRNN de la tâche : [hₖ ; ELMo_k^task] (utile pour SQuAD, SNLI avec couches d'attention)
- Dropout sur ELMo (typiquement 50%)
- Régularisation L2 : λ||w||₂² (encourage les poids à rester proches d'une moyenne)

## Résultats

### Performances sur 6 tâches NLP (Tableau 1)

| Tâche | Baseline | + ELMo | Amélioration absolue/relative |
|-------|----------|--------|-------------------------------|
| **SQuAD** (QA) | 81.1 F1 | **85.8 F1** | +4.7 / 24.9% |
| **SNLI** (Entailment) | 88.0 acc | **88.7 acc** | +0.7 / 5.8% |
| **SRL** | 81.4 F1 | **84.6 F1** | +3.2 / 17.2% |
| **Coref** | 67.2 F1 | **70.4 F1** | +3.2 / 9.8% |
| **NER** (CoNLL 2003) | 90.15 F1 | **92.22 F1** | +2.06 / 21% |
| **SST-5** (Sentiment) | 51.4 acc | **54.7 acc** | +3.3 / 6.8% |

**Nouveau state-of-the-art sur toutes les tâches** testées, y compris en battant les précédents meilleurs ensembles sur plusieurs tâches.

### Comparaison avec CoVe

ELMo surpasse systématiquement CoVe (représentations de l'encodeur MT) :
- **WSD** : ELMo 69.0 F1 vs CoVe 64.7 F1 (couche 2)
- **POS tagging** : ELMo 97.3% vs CoVe 93.3% (couche 1)
- Les représentations profondes (toutes couches) surpassent celles de la dernière couche seule

### Études d'ablation

**Importance d'utiliser toutes les couches** (Tableau 2) :
- SQuAD : Last only 84.7 → All layers (λ=0.001) **85.2** (+0.5)
- SNLI : Last only 89.1 → All layers (λ=0.001) **89.5** (+0.4)
- SRL : Last only 84.1 → All layers (λ=0.001) **84.8** (+0.7)

**Où ajouter ELMo** (Tableau 3) :
- SQuAD/SNLI : Meilleur quand ajouté à l'input ET output du biRNN
- SRL/Coref : Meilleur quand ajouté seulement à l'input

**Efficacité des données** (Figure 1) :
- Modèle ELMo avec 1% des données ≈ Baseline avec 10% des données (SRL)
- Les gains sont plus importants avec moins de données d'entraînement

### Évaluations intrinsèques

**Word Sense Disambiguation** (Tableau 5) :
- biLM couche 2 : **69.0 F1** (compétitif avec systèmes WSD supervisés state-of-the-art)
- biLM couche 1 : 67.4 F1
- Confirmation que les couches hautes capturent le sens des mots en contexte

**POS Tagging** (Tableau 6) :
- biLM couche 1 : **97.3% acc** (égal au state-of-the-art)
- biLM couche 2 : 96.8% acc
- Confirmation que les couches basses capturent mieux la syntaxe

**Visualisation des poids appris** (Figure 2) :
- Input layer : Préférence pour la première couche biLSTM (syntaxe)
- Output layer : Poids relativement balancés entre les couches

## Limites

1. **Coût computationnel** : Le biLM pré-entraîné ajoute des paramètres et du temps de calcul. Bien que les poids soient gelés, il faut calculer les représentations pour chaque token à chaque forward pass.

2. **Architecture fixe** : Le biLM a seulement 2 couches LSTM. Des architectures plus profondes pourraient capturer encore plus d'information, mais au coût de la complexité.

3. **Vocabulaire basé sur les caractères** : Bien que cela permette de gérer les mots OOV, les CNNs sur caractères peuvent ne pas capturer toutes les nuances morphologiques comparé à des approches explicites.

4. **Poids task-specific** : Les poids de combinaison linéaire doivent être appris pour chaque tâche. Une approche plus générale serait souhaitable.

5. **Fine-tuning requis pour certaines tâches** : Pour de petits datasets (NER) ou des domaines spécifiques, le fine-tuning du biLM est nécessaire pour de meilleures performances, ajoutant une étape supplémentaire.

6. **Pas d'explication complète** : Bien que les expériences montrent que différentes couches capturent différents types d'information, il n'y a pas de théorie complète expliquant pourquoi et comment.

7. **Importance du paramètre γ** : Le paramètre de scaling γ^task est crucial pour l'optimisation (surtout dans le cas "last-only"), mais cela ajoute un hyperparamètre à tuner.

8. **Modèle de langage bidirectionnel simple** : L'architecture joint simplement forward et backward LMs. Des approches plus sophistiquées (comme les masked language models de BERT par la suite) pourraient être plus efficaces.

9. **Comparaison limitée avec transfer learning** : Comparé seulement à CoVe. D'autres approches de transfer learning en NLP pourraient être explorées.

## Liens utiles

- **PDF annoté**: [PDF annoté](../../papers/2026/ELMo.pdf)
- **Article**: [Deep contextualized word representations (PDF)](https://arxiv.org/abs/1802.05365)
- **ArXiv**: [https://arxiv.org/pdf/1802.05365](https://arxiv.org/pdf/1802.05365)
- **Code et modèles pré-entraînés**: [http://allennlp.org/elmo](http://allennlp.org/elmo)

## Notes perso
