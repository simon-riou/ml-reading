---
title: "Efficient Estimation of Word Representations in Vector Space"
authors: ["Tomas Mikolov", "Kai Chen", "Greg Corrado", "Jeffrey Dean"]
venue: "ICLR Workshop"
year: 2013
citekey: mikolov2013efficientestimationwordrepresentations
links:
  pdf: "https://arxiv.org/abs/1301.3781"
  code: "https://code.google.com/p/word2vec/"
  project: ""
pdf_annotated: "../../papers/2026/word2vec.pdf"
tags:
  areas: ["NLP", "Representation Learning", "Language Modeling"]
  methods: ["CBOW", "Skip-gram", "Word Embeddings", "Neural Networks"]
  tasks: ["Word Similarity", "Semantic Relationships", "Syntactic Relationships"]
  status: "deep-read"
replication:
  repo: ""
  data: "Google News (6B tokens)"
  env: ""
---

## TL;DR

Word2Vec propose deux architectures simples et efficaces (CBOW et Skip-gram) pour apprendre des word embeddings de haute qualité à partir de très larges corpus. En supprimant la couche cachée non-linéaire des modèles de langage neuronaux, ces modèles peuvent s'entraîner en < 1 jour sur 1.6 milliard de mots et capturent des régularités sémantiques et syntaxiques permettant des opérations algébriques comme vector("King") - vector("Man") + vector("Woman") ≈ vector("Queen").

## Contexte

En 2013, les systèmes NLP traitent majoritairement les mots comme des unités atomiques (indices dans un vocabulaire), sans notion de similarité. Les modèles N-gram dominent le language modeling et peuvent s'entraîner sur des trillions de mots, mais atteignent leurs limites sur certaines tâches. Les modèles de langage neuronaux (NNLM, RNNLM) performent mieux mais sont computationnellement coûteux et n'avaient jamais été entraînés sur plus de quelques centaines de millions de mots avec des embeddings de dimension 50-100.

Les représentations distribuées de mots montraient déjà des promesses, avec la découverte que les word vectors capturent des régularités linéaires (ex: inflexions dans les langues flexionnelles). Cependant, aucune architecture n'avait réussi à scaler efficacement à des milliards de mots avec des vocabulaires de millions de termes.

## Idées clés

1. **Simplification architecturale** : Supprimer la couche cachée non-linéaire des NNLM réduit drastiquement la complexité computationnelle (terme dominant N×D×H éliminé) tout en préservant la capacité à apprendre des representations de qualité

2. **Deux architectures log-linéaires** :
   - **CBOW (Continuous Bag-of-Words)** : Prédit le mot central à partir du contexte (moyenne des vecteurs), ignore l'ordre des mots, Q = N×D + D×log₂(V)
   - **Skip-gram** : Prédit les mots du contexte à partir du mot central, Q = C×(D + D×log₂(V)) où C est la fenêtre de contexte

3. **Régularités linéaires sémantiques et syntaxiques** : Les word vectors préservent des relations algébriques permettant des analogies : vec("biggest") - vec("big") + vec("small") ≈ vec("smallest"), ou vec("Paris") - vec("France") + vec("Italy") ≈ vec("Rome")

4. **Optimisations d'efficacité** :
   - Hierarchical softmax avec Huffman binary tree : log₂(Unigram_perplexity(V)) au lieu de V évaluations
   - Partage de la matrice de projection pour tous les mots (CBOW)
   - Sampling des mots distants (Skip-gram) : mots proches ont plus de poids

5. **Trade-off dimensionnalité/données** : Augmenter simultanément la dimensionnalité des vecteurs ET la quantité de données est crucial ; rendements décroissants si on n'augmente qu'une dimension (Equation 4 : doubler les données ≈ doubler la dimensionnalité en coût)

6. **Complémentarité des architectures** : Skip-gram excelle sur les relations sémantiques, CBOW sur les syntaxiques ; leur combinaison avec RNNLM atteint SOTA sur Microsoft Sentence Completion (58.9%)

## Méthode

**Complexité computationnelle** : O = E × T × Q où E = epochs (3-50), T = tokens, Q dépend de l'architecture

**CBOW (Continuous Bag-of-Words)** :
- Architecture : Input → Projection (moyenne des vecteurs) → Output (hierarchical softmax)
- Contexte : 4 mots passés + 4 mots futurs pour prédire le mot central
- Pas de couche cachée non-linéaire
- Partage de la matrice de projection entre toutes les positions
- Complexité : Q = N×D + D×log₂(V)
- Meilleur sur questions syntaxiques

**Skip-gram** :
- Architecture : Input (mot central) → Projection → Output (prédiction contexte)
- Prédit C mots avant et après (C choisi aléatoirement dans [1; C_max])
- Pour chaque mot d'entraînement : sélectionne R ∈ [1; C], puis R×2 classifications
- Complexité : Q = C×(D + D×log₂(V))
- Meilleur sur questions sémantiques
- Sampling : mots distants ont moins de poids (échantillonnés moins souvent)

**Hierarchical Softmax avec Huffman** :
- Vocabulaire représenté comme arbre binaire de Huffman
- Mots fréquents → codes binaires courts
- Réduction de log₂(V) à log₂(Unigram_perplexity(V)) ≈ 2× speedup pour V=1M
- Crucial pour Skip-gram et CBOW (pas de couche cachée pour absorber le coût)

**Training settings** :
- Optimizer : SGD + backpropagation
- Learning rate : 0.025 initial, décroissance linéaire → 0 en fin d'epoch
- Epochs : 3 pour expériences Tables 2-4
- Parallel training : DistBelief framework, 50-100 replicas, mini-batch async SGD + Adagrad
- Corpus principal : Google News 6B tokens, vocabulaire 1M mots fréquents

**Test set - Semantic-Syntactic Word Relationship** :
- 8869 questions sémantiques (5 types) + 10675 syntaxiques (9 types)
- Types sémantiques : capital-city, currency, city-in-state, man-woman
- Types syntaxiques : comparative, superlative, past tense, plural, present participle, etc.
- Méthode : vec(b) - vec(a) + vec(c), trouver nearest neighbor (cosine), exact match requis
- Single-token words only, synonymes comptés comme erreurs

## Résultats

**Comparaison architectures (Table 3, 640-dim, 320M words, 82K vocab)** :
- RNNLM : 9% semantic / 36% syntactic (baseline, 8 semaines training 1 CPU)
- NNLM : 23% semantic / 53% syntactic (DistBelief, history=8)
- CBOW : 24% semantic / **64% syntactic**
- **Skip-gram : 55% semantic** / 59% syntactic

**Single-CPU vs Public vectors (Table 4, full vocab)** :
- Meilleurs modèles précédents : Huang NNLM 50-dim/990M : 13.3% sem / 11.6% syn
- Our NNLM 100-dim/6B : 34.2% sem / 64.5% syn / 50.8% total
- CBOW 300-dim/783M (1 jour) : 15.5% sem / 53.1% syn / 36.1% total
- **Skip-gram 300-dim/783M (3 jours) : 50.0% sem / 55.9% syn / 53.3% total**

**Impact dimensionnalité et données (Table 2, CBOW, 30k vocab)** :
- 50-dim, 24M words : 13.4% → 600-dim, 783M words : 50.4%
- Rendements décroissants si on n'augmente qu'une dimension
- Optimal : augmenter simultanément dimensionnalité ET données

**1 epoch vs 3 epochs (Table 5)** :
- 1 epoch CBOW 300-dim/783M (0.3 jour) : 33.6% total
- 1 epoch CBOW 300-dim/1.6B (0.6 jour) : 36.1% total (≈ 3 epochs/783M)
- 1 epoch Skip-gram 600-dim/783M (2.5 jours) : **55.5% total**
- 1 epoch Skip-gram 300-dim/1.6B (2 jours) : 53.8% total

**DistBelief large-scale (Table 6, 6B words, 50-100 replicas)** :
- NNLM 100-dim : 50.8% (14 jours × 180 cores)
- CBOW 1000-dim : 63.7% (2 jours × 140 cores) - 57.3% sem / 68.9% syn
- **Skip-gram 1000-dim : 65.6% (2.5 jours × 125 cores)** - 66.1% sem / 65.1% syn

**Microsoft Sentence Completion Challenge (Table 7)** :
- Baseline N-gram : 39%, LSA : 49%, Log-bilinear : 54.8%
- RNNLMs SOTA : 55.4%
- Skip-gram seul : 48.0%
- **Skip-gram + RNNLMs : 58.9%** (nouveau SOTA) - complémentarité des approches

**Analogies sémantiques (Table 8, Skip-gram 300-dim/783M)** :
- France:Paris :: Italy:Rome, Japan:Tokyo, Florida:Tallahassee ✓
- Einstein:scientist :: Messi:midfielder, Mozart:violinist, Picasso:painter ✓
- Copper:Cu :: zinc:Zn, gold:Au, uranium:plutonium ✓
- Microsoft:Windows :: Google:Android, IBM:Linux, Apple:iPhone ✓
- Accuracy ≈ 60% exact match, +10% si 10 exemples moyennés au lieu d'1

**Scaling** :
- Training time : < 1 jour pour embeddings de qualité sur 1.6B words (single CPU)
- Speed : milliards de mots/heure (implémentation C++ multi-thread)
- Perspective : training sur 1 trillion words possible avec DistBelief

## Limites

1. **Exact match trop strict** : Les synonymes sont comptés comme erreurs dans l'évaluation, sous-estimant potentiellement la qualité réelle des embeddings

2. **Pas d'information morphologique** : Les modèles n'ont aucune connaissance de la structure interne des mots, limitant les performances sur questions syntaxiques (atteindre 100% accuracy impossible)

3. **Single-token words uniquement** : Les entités multi-mots (New York) ne sont pas gérées nativement dans le test set

4. **Trade-off dimensionnalité/données manuel** : Pas de méthode automatique pour choisir le ratio optimal entre augmentation de dimensionnalité et de données

5. **Fenêtre de contexte fixe** : La taille de contexte C est un hyperparamètre à tuner manuellement ; augmenter C améliore la qualité mais augmente la complexité

6. **Sampling strategy simple** : Le downweighting des mots distants (Skip-gram) est heuristique, pourrait être optimisé

7. **Évaluation limitée** : Le test Semantic-Syntactic est compréhensif mais reste une proxy ; impact sur tâches NLP downstream pas complètement évalué dans ce paper

8. **Vocabulaire fréquence-based** : Restriction au 1M mots les plus fréquents ignore les mots rares et entités nommées spécifiques

9. **Hierarchical softmax avec Huffman** : Bien qu'efficace, cette approche privilégie les mots fréquents ; negative sampling (introduit dans travaux suivants) pourrait être meilleur

10. **Pas de multi-sense** : Un mot = un vecteur, pas de gestion de la polysémie (bank = banque vs rive)

## Liens utiles

- **PDF annoté**: [Word2Vec - PDF annoté](../../papers/2026/word2vec.pdf)
- **Article**: [Efficient Estimation of Word Representations in Vector Space (PDF)](https://arxiv.org/pdf/1301.3781)
- **ArXiv**: [https://arxiv.org/abs/1301.3781](https://arxiv.org/abs/1301.3781)
- **Code**: [Google Code - word2vec (C++)](https://code.google.com/p/word2vec/)
- **Test set**: [Semantic-Syntactic Word Relationship test](http://www.fit.vutbr.cz/~imikolov/rnnlm/word-test.v1.txt)

## Notes perso

En gros, avant on utilisait des modeles type NNLM, RNNML qui créait les embeddings puis faisait leur tache de NLP (traduction, sentiment, ...). Le papier change ca, en proposant 2 models qui crée les embeddings a part (ce qui est mieux et moins couteux).

CBOW: un vecteur one hot des mots du contexte -> on recupere les indices correspondants de la matrice d'embeddings, puis on moyenne ca et on prédit le mot cible. Ce qui nous interesse c'est pas le model c'est la grosse matrice d'embeddings au milieu.

Skip-gram: Pareil qu'avant sauf qu'on predit les mots du contexte via le mot cible, en entrainant le model plusieurs fois sur le meme mot mais en prédisant un mot du contexte différent a chaque fois.