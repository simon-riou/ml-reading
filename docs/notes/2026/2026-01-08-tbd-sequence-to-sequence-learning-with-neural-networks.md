---
title: "Sequence to Sequence Learning with Neural Networks"
authors: ["Ilya Sutskever", "Oriol Vinyals", "Quoc V. Le"]
venue: "NIPS"
year: 2014
citekey: sutskever2014sequencesequencelearningneural
links:
  pdf: "https://arxiv.org/abs/1409.3215"
  code: ""
  project: ""
pdf_annotated: "../../papers/2026/seq2seq.pdf"
tags:
  areas: ["NLP", "Machine Translation", "Sequence Modeling"]
  methods: ["LSTM", "Seq2Seq", "Encoder-Decoder", "Beam Search"]
  tasks: ["Neural Machine Translation", "Sequence-to-Sequence"]
  status: "deep-read"
replication:
  repo: ""
  data: "WMT'14 English-French"
  env: ""
---

## TL;DR

Ce papier présente une approche end-to-end pour la traduction automatique neuronale en utilisant deux LSTMs profonds : un encodeur qui lit la phrase source et la compresse en un vecteur de taille fixe, et un décodeur qui génère la traduction à partir de ce vecteur. La méthode atteint un BLEU de 34.8 sur WMT'14 anglais-français, surpassant les systèmes SMT basés sur les phrases. L'astuce clé est d'**inverser l'ordre des mots dans la phrase source**, ce qui introduit des dépendances à court terme et facilite grandement l'apprentissage.

## Contexte

En 2014, les DNNs excellent sur la reconnaissance vocale et d'images, mais sont limités aux problèmes où les entrées et sorties ont une dimensionnalité fixe. De nombreux problèmes importants (traduction, question-réponse) sont des problèmes de séquence-à-séquence où les longueurs varient et ne sont pas connues a priori. Les approches existantes utilisent principalement les réseaux neuronaux pour re-scorer les listes n-best des systèmes SMT traditionnels, plutôt que pour la traduction directe.

Bien que les RNNs soient des généralisations naturelles pour les séquences, ils ont du mal avec les dépendances à long terme. Les LSTMs ont démontré leur capacité à apprendre de telles dépendances, ce qui les rend appropriés pour mapper des séquences source à des séquences cible avec un décalage temporel considérable.

## Idées clés

1. **Architecture Encoder-Decoder avec LSTM** : Utiliser un LSTM pour lire la séquence d'entrée et obtenir une représentation vectorielle de dimension fixe, puis un autre LSTM pour décoder la séquence de sortie à partir de cette représentation.

2. **Inversion de la phrase source** : Au lieu de mapper "a, b, c" → "α, β, γ", le modèle mappe "c, b, a" → "α, β, γ". Cela place "a" en proximité de "α", "b" proche de "β", etc., réduisant le "minimal time lag" et facilitant la communication entre source et cible via SGD.

3. **LSTMs profonds** : Utiliser 4 couches de LSTM améliore significativement les performances (chaque couche réduit la perplexité de ~10%).

4. **LSTMs séparés pour encoder et décoder** : Augmente le nombre de paramètres à coût computationnel négligeable et permet l'entraînement sur plusieurs paires de langues simultanément.

5. **Performance sur phrases longues** : Contrairement aux attentes, le LSTM gère bien les phrases longues (>35 mots) grâce à l'inversion des phrases source qui améliore l'utilisation de la mémoire.

## Méthode

**Architecture** :
- 2 LSTMs distincts : encodeur et décodeur
- 4 couches par LSTM avec 1000 cellules par couche
- Embeddings de 1000 dimensions
- Vocabulaire : 160k mots (source), 80k mots (cible)
- 384M paramètres totaux (64M connexions récurrentes)
- Représentation d'une phrase : vecteur de 8000 nombres réels

**Objectif d'entraînement** :
- Maximiser p(T|S) = ∏ p(y_t|v, y_1, ..., y_{t-1})
- v = représentation de la phrase source (dernier état caché de l'encodeur)
- Chaque p(y_t|...) est un softmax sur les 80k mots

**Détails d'entraînement** :
- Initialisation : uniforme [-0.08, 0.08]
- SGD sans momentum, learning rate 0.7
- Après 5 epochs : réduction de moitié du LR tous les demi-epochs
- 7.5 epochs au total
- Batch size : 128 séquences de longueur similaire (speedup 2x)
- Gradient clipping : si ||g||_2 > 5, alors g = 5g/||g||_2

**Parallélisation** :
- 8 GPUs : 4 pour les couches LSTM (1 par couche), 4 pour le softmax
- Débit : 6300 mots/seconde (vs 1700 sur 1 GPU)
- Entraînement : ~10 jours

**Décodage** :
- Beam search gauche-à-droite simple
- Beam size B (même B=1 fonctionne bien, B=2 donne la plupart des bénéfices)
- Arrêt quand `<EOS>` est généré

## Résultats

**Traduction directe (WMT'14 EN→FR)** :
- Ensemble de 5 LSTMs, beam=12 : **34.81 BLEU**
- Baseline SMT [Schwenk'14] : 33.30 BLEU
- Premier système de traduction neuronale pure à battre un système SMT à grande échelle
- Vocabulaire limité (80k) pénalise les mots OOV

**Rescoring des 1000-best du baseline** :
- Ensemble de 5 LSTMs : **36.5 BLEU** (+3.2 vs baseline)
- Proche du meilleur résultat WMT'14 (37.0 de Durrani et al.)
- Single LSTM reversed : 35.85 BLEU

**Impact de l'inversion** :
- Non-inversé : perplexité 5.8, BLEU 25.9
- Inversé : perplexité 4.7, BLEU 30.6
- Amélioration drastique sur les phrases longues

**Performance selon la longueur** :
- Pas de dégradation jusqu'à 35 mots
- Dégradation mineure seulement sur les phrases les plus longues
- Contraste avec les difficultés rapportées par d'autres (Cho et al., Bahdanau et al.)

**Représentations apprises** :
- Les représentations vectorielles sont sensibles à l'ordre des mots
- Relativement invariantes à la voix active/passive
- Phrases de sens similaire → vecteurs proches (visualisation PCA)

## Limites

1. **Vocabulaire fixe** : Le modèle ne peut pas générer de mots hors vocabulaire (80k pour le français). Les mots OOV sont remplacés par un token "UNK", ce qui pénalise le score BLEU.

2. **Explication incomplète de l'inversion** : Bien que l'inversion des phrases source améliore drastiquement les performances, les auteurs n'ont pas d'explication complète du phénomène. L'hypothèse est que cela réduit le "minimal time lag" en introduisant des dépendances à court terme.

3. **Pas d'attention mechanism** : Contrairement à Bahdanau et al. (2014), ce modèle compresse toute la phrase source en un seul vecteur fixe. L'attention pourrait améliorer encore les performances.

4. **Décodeur simple** : Le beam search utilisé est basique. Des stratégies de décodage plus sophistiquées pourraient améliorer les résultats.

5. **Coût computationnel** : L'entraînement nécessite 8 GPUs pendant ~10 jours, ce qui est coûteux.

6. **Architecture non-optimisée** : Les auteurs mentionnent que leur approche est "relatively unoptimized", suggérant qu'il y a beaucoup de marge d'amélioration.

7. **Traduction mono-directionnelle** : Le modèle lit la source de droite à gauche (après inversion) mais génère la cible de gauche à droite. Une architecture bidirectionnelle pourrait capturer plus d'informations.

## Liens utiles

- **PDF annoté**: [PDF annoté](../../papers/2026/seq2seq.pdf)
- **Article**: [Sequence to Sequence Learning with Neural Networks (PDF)](https://arxiv.org/abs/1409.3215)
- **ArXiv**: [https://arxiv.org/pdf/1409.3215](https://arxiv.org/pdf/1409.3215)

## Notes perso
