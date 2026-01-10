---
title: "Neural Machine Translation by Jointly Learning to Align and Translate"
authors: ["Dzmitry Bahdanau", "Kyunghyun Cho", "Yoshua Bengio"]
venue: "ICLR"
year: 2015
citekey: bahdanau2015neural
links:
  pdf: "https://arxiv.org/pdf/1409.0473"
  code: ""
  project: ""
pdf_annotated: ""
tags:
  areas: ["neural-machine-translation", "deep-learning"]
  methods: ["attention-mechanism", "encoder-decoder", "rnn", "seq2seq"]
  tasks: ["machine-translation"]
  status: "deep-read"
replication:
  repo: ""
  data: ""
  env: ""
---

## TL;DR

Ce papier introduit le **mécanisme d'attention** pour la traduction automatique neuronale en proposant une architecture qui apprend conjointement à aligner et traduire. Contrairement aux encodeurs-décodeurs classiques qui compressent toute la phrase source en un vecteur de taille fixe (bottleneck), le modèle proposé (RNNsearch) encode la phrase source en une séquence de vecteurs (annotations) et sélectionne dynamiquement les parties pertinentes lors du décodage via un soft-alignment différentiable. L'encodeur utilise un **BiRNN** pour capturer le contexte avant et après chaque mot source. Le décodeur calcule un **vecteur de contexte dynamique** pour chaque mot cible via des poids d'attention, permettant au modèle de "chercher" automatiquement les informations pertinentes dans la source. Sur WMT'14 EN→FR, RNNsearch-50 atteint **26.75 BLEU** (34.16 sans UNK), comparable à Moses (33.30 BLEU), et montre une robustesse remarquable aux phrases longues (50+ mots), là où les modèles classiques s'effondrent. L'attention permet d'inspecter visuellement les alignements appris, qui correspondent aux intuitions linguistiques.

## Contexte

En 2014, la traduction automatique neuronale (NMT) émerge comme alternative aux systèmes SMT traditionnels (phrase-based), qui combinent de nombreux sous-composants entraînés séparément. Les approches NMT proposées (Kalchbrenner & Blunsom 2013, Sutskever et al. 2014, Cho et al. 2014) utilisent une architecture encoder-decoder : un RNN encode la phrase source en un **vecteur de taille fixe**, puis un décodeur génère la traduction à partir de ce vecteur unique.

**Problème majeur** : Cho et al. (2014b) montrent empiriquement que les performances de l'encoder-decoder basique se dégradent **rapidement** avec la longueur des phrases. Compresser toute l'information d'une phrase (quelle que soit sa longueur) en un seul vecteur fixe constitue un **bottleneck** limitant les capacités du modèle. Les phrases longues (>30 mots) posent des difficultés particulières, l'information critique étant perdue ou diluée dans le vecteur de contexte fixe.

Les systèmes SMT traditionnels utilisent des alignements explicites entre mots source et cible, mais dans les approches neuronales classiques, l'alignement n'est pas modélisé explicitement. Les RNNs souffrent encore des problèmes de dépendances à long terme malgré les LSTMs. Les travaux existants utilisent principalement les réseaux neuronaux comme features additionnelles dans les systèmes SMT ou pour re-ranker les candidats, plutôt que comme système de traduction autonome end-to-end.

## Idées clés

1. **Encoder en séquence d'annotations, pas en vecteur fixe** : Au lieu d'encoder toute la phrase source en un seul vecteur c, encoder en une séquence d'annotations (h₁, ..., h_Tx) où chaque hⱼ contient des informations sur toute la phrase source mais **se concentre sur les mots autour de la position j**. Cela libère le modèle de devoir compresser toute l'information en un vecteur unique.

2. **Contexte dynamique via soft-alignment** : Pour chaque mot cible yᵢ, calculer un vecteur de contexte **distinct** cᵢ = Σⱼ αᵢⱼhⱼ, où les poids d'attention αᵢⱼ indiquent l'importance de l'annotation source hⱼ pour générer le mot cible yᵢ. Le décodeur "cherche" automatiquement les parties pertinentes de la phrase source à chaque étape.

3. **Alignement différentiable entraîné end-to-end** : Contrairement aux systèmes SMT où l'alignement est une variable latente, le modèle d'alignement a(sᵢ₋₁, hⱼ) calcule directement un **soft-alignment** via un petit feedforward network. Le gradient de la fonction de coût peut être backpropagé à travers le mécanisme d'alignement, permettant l'entraînement conjoint de l'alignement et de la traduction.

4. **Encodeur bidirectionnel (BiRNN)** : Utiliser un BiRNN où chaque annotation hⱼ = [→hⱼ; ←hⱼ] concatène les états cachés forward et backward. Cela permet à chaque annotation de résumer **à la fois les mots précédents et suivants**, donnant un contexte riche centré sur le mot xⱼ. Les RNNs ont tendance à mieux représenter les inputs récents, donc hⱼ se concentre naturellement sur les mots autour de xⱼ.

5. **Attention comme expected annotation** : On peut interpréter αᵢⱼ comme la probabilité que le mot cible yᵢ soit aligné au mot source xⱼ. Le contexte cᵢ est alors l'**espérance** de l'annotation sur tous les alignements possibles. Cette formulation probabiliste élégante unifie alignement et traduction.

6. **Soulagement du burden de l'encodeur** : En permettant au décodeur d'avoir un mécanisme d'attention, on libère l'encodeur de devoir encoder **toute** l'information dans un vecteur fixe. L'information peut être **distribuée** dans la séquence d'annotations et récupérée sélectivement par le décodeur selon les besoins.

7. **Robustesse aux phrases longues** : Le modèle n'a besoin d'encoder **que les parties autour d'un mot particulier** avec précision, pas la phrase entière parfaitement. Cela permet de gérer naturellement les phrases longues sans dégradation de performance, contrairement aux encoders classiques qui s'effondrent sur les phrases >30 mots.

## Méthode

**Architecture globale** :

L'architecture RNNsearch se compose d'un encodeur bidirectionnel et d'un décodeur avec attention.

**Encodeur bidirectionnel** :

- Forward RNN : →f lit la phrase source x₁ → x_Tx et calcule les états cachés forward (→h₁, ..., →h_Tx)
- Backward RNN : ←f lit la phrase source x_Tx → x₁ et calcule les états cachés backward (←h₁, ..., ←h_Tx)
- Annotation pour chaque mot : hⱼ = [→hⱼ; ←hⱼ] (concaténation forward + backward)
- Chaque annotation hⱼ ∈ ℝ^(2n) contient des informations sur toute la phrase source, mais se concentre sur les mots autour de xⱼ

**Décodeur avec attention** :

Probabilité conditionnelle : p(yᵢ | y₁, ..., yᵢ₋₁, x) = g(yᵢ₋₁, sᵢ, cᵢ)

État caché du décodeur : sᵢ = f(sᵢ₋₁, yᵢ₋₁, cᵢ)

**Mécanisme d'attention** (calcul du contexte cᵢ) :

```
cᵢ = Σⱼ αᵢⱼ hⱼ                                    # Somme pondérée des annotations

αᵢⱼ = exp(eᵢⱼ) / Σₖ exp(eᵢₖ)                      # Softmax des scores (normalisation)

eᵢⱼ = a(sᵢ₋₁, hⱼ) = vₐᵀ tanh(Wₐsᵢ₋₁ + Uₐhⱼ)      # Modèle d'alignement (feedforward NN)
```

- αᵢⱼ : poids d'attention (importance de l'annotation source hⱼ pour générer yᵢ)
- eᵢⱼ : score d'alignement (similarité entre état décodeur sᵢ₋₁ et annotation hⱼ)
- Le modèle d'alignement a est un MLP à 1 couche cachée avec tanh

**Unité récurrente gated (GRU-like)** :

Pour l'encodeur et le décodeur, utilisation de gated hidden units (similaires aux LSTMs) :

```
sᵢ = (1 - zᵢ) ⊙ sᵢ₋₁ + zᵢ ⊙ s̃ᵢ                  # Interpolation entre état précédent et candidat

s̃ᵢ = tanh(W e(yᵢ₋₁) + U[rᵢ ⊙ sᵢ₋₁] + C cᵢ)        # État candidat

zᵢ = σ(Wz e(yᵢ₋₁) + Uz sᵢ₋₁ + Cz cᵢ)              # Update gate

rᵢ = σ(Wr e(yᵢ₋₁) + Ur sᵢ₋₁ + Cr cᵢ)              # Reset gate
```

- zᵢ : update gates (contrôlent combien de l'état précédent conserver)
- rᵢ : reset gates (contrôlent combien de l'état précédent réinitialiser)
- e(yᵢ₋₁) : embedding du mot précédent

**Output layer (deep output avec maxout)** :

```
p(yᵢ | sᵢ, yᵢ₋₁, cᵢ) ∝ exp(yᵢᵀ Wo tᵢ)

tᵢ = [max{t̃ᵢ,₂ⱼ₋₁, t̃ᵢ,₂ⱼ}]ⱼ₌₁,...,ₗ               # Maxout layer

t̃ᵢ = Uo sᵢ₋₁ + Vo e(yᵢ₋₁) + Co cᵢ
```

- Maxout avec l=500 hidden units
- Softmax final sur vocabulaire Ky pour probabilités des mots

**Détails d'implémentation** :

- **Architecture** : Encodeur BiRNN (1000 forward + 1000 backward hidden units), Décodeur (1000 hidden units), Embeddings (620 dim), Maxout output layer (500 units), Alignment model (1000 hidden units)
- **Dataset** : WMT'14 EN-FR (348M mots après data selection), vocabulaire 30k mots (source et cible), mots OOV → [UNK]
- **Optimisation** : SGD + Adadelta (ε=10⁻⁶, ρ=0.95), minibatch 80 sentences, gradient clipping (||g||₂ ≤ 1)
- **Initialisation** : Matrices récurrentes (orthogonales aléatoires), Wₐ et Uₐ (Gaussienne μ=0, σ²=0.001²), autres poids (Gaussienne μ=0, σ²=0.01²), biais (zéro)
- **Entraînement** : ~5 jours, deux variants (sentences ≤30 mots : RNNsearch-30, sentences ≤50 mots : RNNsearch-50)
- **Décodage** : Beam search pour approximer arg max p(y|x)
- **Optimisation batch** : Tri des phrases par longueur toutes les 20 updates (minimiser le padding)

**Modèle baseline (RNNencdec)** : Encodeur et décodeur avec 1000 hidden units chacun, même setup d'entraînement, mais contexte c fixe = dernier état caché de l'encodeur.

## Résultats

**Performances quantitatives (WMT'14 EN→FR, Table 1)** :

| Modèle | All sentences BLEU | No UNK BLEU |
|--------|-------------------|-------------|
| RNNencdec-30 | 13.93 | 24.19 |
| **RNNsearch-30** | **21.50** | **31.44** |
| RNNencdec-50 | 17.82 | 26.71 |
| **RNNsearch-50** | **26.75** | **34.16** |
| RNNsearch-50* | **28.45** | **36.15** |
| Moses (phrase-based SMT) | 33.30 | 35.63 |

*RNNsearch-50* = entraîné plus longtemps jusqu'à convergence complète sur dev set

**Observations clés** :

1. **RNNsearch >> RNNencdec** : Sur toutes les configurations, RNNsearch surpasse largement RNNencdec (+7.57 BLEU pour -30, +8.93 BLEU pour -50). L'attention apporte un gain massif.

2. **Performance comparable à Moses** : RNNsearch-50* atteint 28.45 BLEU (36.15 sans UNK), **proche de Moses 33.30 BLEU**. C'est remarquable car Moses utilise un corpus monolingue additionnel de 418M mots que RNNsearch n'a pas. Sur les phrases sans mots inconnus, RNNsearch-50* (36.15) est **presque au niveau de Moses (35.63)**.

3. **RNNsearch-30 bat RNNencdec-50** : Même entraîné sur des phrases plus courtes (≤30 mots), RNNsearch-30 (21.50 BLEU) surpasse RNNencdec-50 (17.82 BLEU) entraîné sur phrases ≤50 mots. L'attention est plus importante que la capacité à voir des phrases longues en entraînement.

**Robustesse aux phrases longues (Figure 2)** :

- **RNNencdec** : Performance **s'effondre dramatiquement** au-delà de 20 mots. À 50 mots, BLEU ~5 (quasi inutilisable).
- **RNNsearch-50** : Performance **stable** même pour phrases 50+ mots, BLEU ~25. **Pas de dégradation** avec la longueur.
- **RNNsearch-30** : Même entraîné sur phrases courtes, reste robuste jusqu'à 40 mots (BLEU ~15-20), bien mieux que RNNencdec.

Cela confirme l'hypothèse : le vecteur de contexte fixe est un bottleneck pour les phrases longues. L'attention résout ce problème en encodant l'information de manière distribuée.

**Analyse qualitative des alignements (Figure 3)** :

Les visualisations des poids d'attention αᵢⱼ révèlent des propriétés remarquables :

1. **Alignements majoritairement monotones** : Poids forts le long de la diagonale (ordre similaire EN-FR), ce qui est attendu pour cette paire de langues.

2. **Réordonnancement non-trivial** : Le modèle gère correctement les différences d'ordre adjectif-nom (EN: "European Economic Area" → FR: "zone économique européenne"). L'attention saute intelligemment pour aligner "zone"-"Area", puis revient en arrière pour "économique"-"Economic" et "européenne"-"European".

3. **Soft-alignment vs hard-alignment** : Exemple "the man" → "l' homme". Un hard-alignment mappe "the"→"l'" et "man"→"homme", mais pour générer "l'" correctement, il faut regarder "man" (le mot suivant) pour déterminer le genre. Le soft-alignment résout cela naturellement : pour générer "l'", le modèle regarde **à la fois** "the" et "man".

4. **Gestion des phrases de longueurs différentes** : Le soft-alignment gère naturellement les cas où source et cible ont des longueurs différentes, sans mappings artificiels vers NULL.

5. **Alignements linguistiquement plausibles** : Les alignements découverts correspondent à l'intuition linguistique humaine, validant que le modèle apprend des correspondances sémantiques réelles.

**Exemples de traductions longues (Table 3, Annexe C)** :

- **Phrase 1** (31 mots) : "An admitting privilege is the right of a doctor to admit a patient to a hospital or a medical centre to carry out a diagnosis or a procedure, based on his status as a health care worker at a hospital."
  - **RNNencdec-50** : S'effondre à mi-phrase (après ~15 mots), remplace la fin par du texte incohérent ("en fonction de son état de santé" au lieu de "based on his status as a health care worker")
  - **RNNsearch-50** : Traduction **complète et correcte**, préserve tout le sens sans omettre de détails

- **Phrase 2** (34 mots) : Même pattern - RNNencdec commence à dévier après ~30 mots, manque des guillemets fermants. RNNsearch traduit correctement jusqu'au bout.

Ces exemples qualitatifs confirment que RNNsearch permet une **traduction fiable des phrases longues**, là où RNNencdec échoue systématiquement.

**Statistiques d'entraînement (Table 2)** :

- RNNsearch-50 : 2.88×10⁵ updates (2.2 epochs, 111h sur Quadro K-6000)
- RNNencdec-50 : 6.00×10⁵ updates (4.5 epochs, 108h)
- Training NLL : RNNsearch-50 (40.7) vs RNNencdec-50 (44.0) - l'attention améliore la vraisemblance
- Dev NLL : RNNsearch-50 (38.1) vs RNNencdec-50 (43.6) - meilleure généralisation avec attention

## Limites

1. **Vocabulaire fixe limité (30k mots)** : Les mots hors vocabulaire sont mappés à [UNK], ce qui pénalise fortement le score BLEU. Sur le test set complet, RNNsearch-50 obtient 26.75 BLEU, mais 34.16 BLEU sur les phrases sans UNK (+7.41 points). La gestion des mots rares/inconnus reste un défi majeur pour atteindre les performances des systèmes SMT sur toutes les phrases. Les auteurs identifient explicitement cela comme un challenge futur.

2. **Coût computationnel O(Tx × Ty)** : Le modèle doit calculer les poids d'attention αᵢⱼ pour **chaque paire** (mot cible i, mot source j), soit Tx × Ty évaluations du modèle d'alignement a(sᵢ₋₁, hⱼ) par phrase. Pour des phrases de 50 mots source/cible, cela fait 2500 évaluations. Ce coût n'est pas prohibitif pour la traduction (phrases typiquement 15-40 mots), mais pourrait limiter l'applicabilité à d'autres tâches avec des séquences très longues (documents entiers, vidéos).

3. **Pas de corpus monolingue utilisé** : Moses utilise 418M mots additionnels en corpus monolingue français pour améliorer son modèle de langue cible. RNNsearch n'utilise que les 348M mots du corpus parallèle. Avec un corpus monolingue pour pré-entraîner l'encodeur, les performances pourraient s'améliorer significativement, mais cela n'a pas été exploré.

4. **Alignement contraint par l'attention monotone implicite** : Bien que le modèle puisse en théorie apprendre des alignements non-monotones arbitraires, l'architecture BiRNN favorise implicitement les alignements proches de la monotonie. Pour des paires de langues avec réordonnancement massif (EN-DE, EN-JA), le modèle pourrait avoir plus de difficultés. Graves (2013) utilise des kernels Gaussiens avec contrainte de localisation croissante monotone ; Bahdanau n'a pas cette contrainte mais ne teste que EN-FR (réordonnancement modéré).

5. **Pas de mécanisme de couverture (coverage)** : L'attention peut "oublier" de traduire certains mots source ou traduire le même mot plusieurs fois. Il n'y a pas de mécanisme forçant le modèle à couvrir tous les mots source exactement une fois. Les travaux ultérieurs (Tu et al. 2016, Mi et al. 2016) ajoutent des coverage vectors pour résoudre ce problème.

6. **Beam search simple sans optimisations** : Le décodage utilise un beam search basique sans length normalization, coverage penalty, ou autres heuristiques utilisées dans les systèmes SMT. Des stratégies de décodage plus sophistiquées pourraient améliorer les résultats, mais n'ont pas été explorées.

7. **Architecture non-optimisée** : Les auteurs mentionnent explicitement que leur approche est "relatively unoptimized", suggérant qu'il existe beaucoup de marge d'amélioration (architecture, hyperparamètres, régularisation, etc.). C'était une preuve de concept montrant la viabilité de l'attention, pas une tentative d'optimisation maximale.

8. **Pas d'analyse théorique** : Le papier ne fournit pas d'explication théorique rigoureuse de **pourquoi** l'attention fonctionne si bien. L'intuition (éviter le bottleneck du vecteur fixe, distribuer l'information) est claire, mais il n'y a pas de garanties formelles de convergence ou d'analyse de la capacité du modèle.

9. **Entraînement long et coûteux** : RNNsearch-50 nécessite ~111h d'entraînement sur GPU Quadro K-6000 (5 jours). Pour l'époque (2014), c'était un investissement computationnel significatif, limitant l'accessibilité pour la recherche.

10. **Évaluation limitée à une seule paire de langues** : Seul EN→FR est testé. Les résultats pourraient ne pas se généraliser à des paires de langues morphologiquement riches, à faibles ressources, ou avec des structures très différentes (EN-JA, EN-AR, etc.).

11. **Pas de comparaison avec autres mécanismes d'attention** : Le modèle d'alignement a(sᵢ₋₁, hⱼ) est un MLP simple avec tanh. D'autres fonctions (dot-product, bilinear, etc.) n'ont pas été explorées. Les travaux ultérieurs (Luong et al. 2015) compareront différentes attention functions.

12. **Soft-alignment uniquement** : Contrairement aux modèles avec hard-attention (Xu et al. 2015 pour image captioning), qui échantillonnent stochastiquement des positions à attendre via REINFORCE, Bahdanau utilise uniquement soft-attention (moyenne pondérée différentiable). Le hard-attention pourrait être plus interprétable et économe en calcul, mais n'est pas exploré ici.

## Liens utiles

- **PDF annoté**: [PDF annoté](../../papers/2026/attention_by_alignement.pdf)
- **Article**: [Neural Machine Translation by Jointly Learning to Align and Translate (PDF)](https://arxiv.org/pdf/1409.0473)
- **ArXiv**: [arXiv:1409.0473](https://arxiv.org/abs/1409.0473)

## Notes perso

Premiere fois qu on ajoute de l'attention (premiere fois pas sur mais bon c'est le plus gros) aux RNN. C'est basique mais ca marche.