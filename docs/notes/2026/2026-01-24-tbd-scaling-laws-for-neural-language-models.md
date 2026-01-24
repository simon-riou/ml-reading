---
title: "Scaling Laws for Neural Language Models"
authors: ["Jared Kaplan", "Sam McCandlish", "Tom Henighan", "Tom B. Brown", "Benjamin Chess", "Rewon Child", "Scott Gray", "Alec Radford", "Jeffrey Wu", "Dario Amodei"]
venue: "arXiv"
year: 2020
citekey: kaplan2020scalinglawsneurallanguage
links:
  pdf: "https://arxiv.org/abs/2001.08361"
  code: ""
  project: ""
pdf_annotated: "../../papers/2026/scaling_laws.pdf"
tags:
  areas: ["NLP", "Language Modeling", "Deep Learning Theory", "Scaling"]
  methods: ["Transformer", "Power Laws", "Compute-Optimal Training"]
  tasks: ["Language Modeling", "Cross-Entropy Loss Prediction"]
  status: "skimmed"
replication:
  repo: ""
  data: "WebText2"
  env: ""
---

## TL;DR

Ce papier fondateur d'OpenAI établit des **lois de scaling empiriques en puissance** pour les modèles de langage Transformer. Les performances (mesurées par la cross-entropy loss) suivent des **power laws** prévisibles en fonction de trois facteurs : le nombre de paramètres N, la taille du dataset D, et le compute C. Les résultats clés : (1) la loss scale comme L(N) ∝ N^(-0.076), L(D) ∝ D^(-0.095), et L(C) ∝ C^(-0.050) ; (2) les hyperparamètres architecturaux (profondeur, largeur, nombre de têtes) ont un **impact minimal** tant que N reste fixe ; (3) les grands modèles sont **plus sample-efficient** ; (4) pour un budget compute fixe, il est optimal d'entraîner des **modèles très grands sur peu de données** en s'arrêtant bien avant convergence (N ∝ C^0.73, D ∝ C^0.27). Ces scaling laws permettent de prédire les performances et d'allouer optimalement les ressources, posant les bases de l'ère des LLMs massifs.

## Contexte

En 2020, le deep learning pour le NLP progresse rapidement avec BERT, GPT-2, XLNet, mais plusieurs questions fondamentales restent ouvertes :

**Manque de compréhension quantitative** : On sait intuitivement que "plus gros = meilleur", mais il n'existe pas de relations quantitatives précises reliant taille du modèle, données, compute, et performances. Impossible de prédire les performances avant d'entraîner.

**Allocation de ressources ad hoc** : Les chercheurs choisissent taille de modèle, durée d'entraînement, et taille de dataset de manière empirique. Pas de théorie pour optimiser l'allocation d'un budget compute fixe.

**Questions non résolues** :
- La profondeur est-elle plus importante que la largeur ?
- Combien de données faut-il pour un modèle de taille N ?
- Faut-il entraîner jusqu'à convergence ou s'arrêter tôt ?
- Comment les performances extrapolent-elles à des échelles jamais testées ?

**Travaux préliminaires** : Quelques travaux ont observé des relations entre taille de modèle et performances, mais sans établir de lois quantitatives générales ni explorer systématiquement l'espace des hyperparamètres.

**Motivation** : OpenAI dispose des ressources pour mener une étude systématique à grande échelle, variant taille de modèle (768 à 1.5B paramètres), taille de dataset (22M à 23B tokens), forme du réseau, batch size, et compute sur plus de **7 ordres de grandeur**.

## Idées clés

1. **Performance = power law en N, D, C** : La loss de test suit des lois de puissance remarquablement lisses avec chacun des trois facteurs de scale (quand les deux autres ne sont pas limitants). Ces trends s'étendent sur 6+ ordres de grandeur sans déviation, suggérant une universalité profonde.

2. **Faible dépendance à la forme du réseau** : À nombre de paramètres N fixe, les performances varient de seulement quelques pourcents quand on change radicalement le ratio profondeur/largeur (facteur 40x), le nombre de têtes d'attention, ou la dimension du feed-forward. Seul le **scale total** compte.

3. **Équation unifiée L(N, D) pour l'overfitting** : Une seule équation capture la dépendance simultanée en N et D, avec l'overfitting prévisible selon le ratio N^0.74/D. Quand on multiplie N par 8x, il suffit de multiplier D par ~5x pour éviter l'overfitting.

4. **Grands modèles = meilleure sample efficiency** : Les grands modèles atteignent une loss donnée avec **moins de tokens** et **moins de steps** que les petits modèles. Un modèle 10^9 paramètres nécessite ~100x moins d'exemples qu'un modèle 10^5 paramètres pour la même performance.

5. **Convergence inefficiente pour le compute** : Pour un budget compute fixe, il est optimal d'entraîner un **très grand modèle** et de s'arrêter **bien avant convergence** (~10% au-dessus de la loss convergée). La taille optimale croît comme N ∝ C^0.73, tandis que les données croissent lentement D ∝ C^0.27.

6. **Critical batch size** : Il existe un batch size critique B_crit(L) qui optimise le trade-off temps/compute. B_crit suit aussi une power law en la loss et est **indépendant de la taille du modèle**.

7. **Transformers >> LSTMs sur les longs contextes** : Les LSTMs plafonnent après ~100 tokens dans le contexte, alors que les Transformers continuent à s'améliorer sur les 1024 tokens. L'écart se creuse avec la taille du modèle.

8. **Généralisation prévisible** : Les performances sur d'autres distributions (Wikipedia, Books, Common Crawl) s'améliorent en parallèle avec la distribution d'entraînement, avec un offset constant. La généralisation dépend uniquement de la loss d'entraînement, pas de la profondeur ou de la durée d'entraînement.

## Méthode

### Architecture et paramétrage

**Transformer decoder-only** basé sur GPT-2 :
- Vocabulaire BPE de 50,257 tokens
- Contexte de 1024 tokens
- Optimiseur Adam (Adafactor pour >1B paramètres)
- Learning rate avec warmup linéaire + cosine decay

**Définition de N** (paramètres non-embedding) :
```
N ≈ 12 × n_layer × d_model²
```
Les embeddings (n_vocab × d_model + n_ctx × d_model) sont **exclus** car leur inclusion obscurcit les trends.

**Estimation du compute** :
```
C ≈ 6 × N × B × S  (en FLOPs)
```
où B = batch size, S = steps. Le facteur 6 compte forward (2N) + backward (4N).

### Dataset : WebText2

Extension de WebText (GPT-2) avec :
- Liens Reddit outbound (2017-2018) avec ≥3 karma
- 20.3M documents, 96GB de texte
- 22.9B tokens (après tokenization BPE)
- Test set de 660M tokens

### Expériences systématiques

**Variations explorées** :
- Taille de modèle : 768 à 1.5B paramètres
- Taille de dataset : 22M à 23B tokens
- Forme : profondeur, largeur, têtes, d_ff
- Batch size : 2^19 tokens (varié pour mesurer B_crit)
- Contexte : principalement 1024, aussi testé plus court

**Protocole** :
- Entraînement avec early stopping sur test loss
- 10% dropout pour régularisation
- 250k steps avec batch size 512 séquences

## Résultats

### Scaling laws fondamentales

**Loss vs Paramètres (données infinies)** :
```
L(N) = (N_c / N)^αN    où αN ≈ 0.076, N_c ≈ 8.8×10¹³
```
Doubler N réduit la loss d'un facteur 2^(-0.076) ≈ 0.95 (gain de 5%).

**Loss vs Dataset (modèle infini)** :
```
L(D) = (D_c / D)^αD    où αD ≈ 0.095, D_c ≈ 5.4×10¹³
```

**Loss vs Compute optimal** :
```
L(C_min) = (C_c^min / C_min)^α_C^min    où α_C^min ≈ 0.050
```

### Équation unifiée L(N, D)

```
L(N, D) = [(N_c/N)^(αN/αD) + D_c/D]^αD
```

Cette équation prédit précisément l'overfitting pour toutes les combinaisons (N, D) testées. Pour éviter l'overfitting :
```
D ≳ 5×10³ × N^0.74
```

### Allocation optimale du compute

Pour un budget C fixe :
- **Taille optimale** : N_opt ∝ C^0.73 (croissance rapide)
- **Données optimales** : D_opt ∝ C^0.27 (croissance lente)
- **Steps optimaux** : S_opt ∝ C^0.03 (quasi-constant)
- **Batch size optimal** : B_opt ∝ C^0.24

**Implication majeure** : Pour 1000x plus de compute, utiliser ~200x plus de paramètres mais seulement ~5x plus de données, avec un nombre de steps quasi-identique.

### Indépendance de la forme

| Variation | Range testé | Impact sur loss |
|-----------|-------------|-----------------|
| Aspect ratio (d_model/n_layer) | 40x | <3% |
| Feed-forward ratio (d_ff/d_model) | variable | <2% |
| Dimension des têtes | variable | <2% |

Les performances dépendent de N, **pas** de comment N est distribué entre profondeur et largeur.

### Transformers vs LSTMs

- LSTMs : plafonnent après ~100 tokens de contexte
- Transformers : s'améliorent continûment sur les 1024 tokens
- À N égal, Transformers surpassent asymptotiquement les LSTMs
- La différence augmente avec la taille du modèle

### Critical batch size

```
B_crit(L) = B* / L^(1/αB)    où B* ≈ 2×10⁸, αB ≈ 0.21
```

B_crit est **indépendant de N** et ne dépend que de la loss cible. Pour les plus grands modèles à convergence : B_crit ≈ 1-2M tokens.

## Limites

### Limites méthodologiques

1. **Pas de compréhension théorique** : Les scaling laws sont purement empiriques. On ne sait pas pourquoi ces exposants spécifiques émergent, ni si/quand ils changeront.

2. **Extrapolation incertaine** : Les trends sont testés sur ~7 ordres de grandeur, mais les prédictions à des échelles bien supérieures (10^12 paramètres) sont spéculatives.

3. **Contradiction interne à grande échelle** : Les trends L(C) et L(D) se contredisent autour de C* ~ 10^4 PF-days, N* ~ 10^12 paramètres, suggérant que les lois doivent changer avant ce point.

4. **Dataset unique** : Toutes les expériences utilisent WebText2. Les exposants pourraient différer pour d'autres domaines (code, math, multilingue).

5. **Architecture unique** : Seuls les Transformers decoder-only sont étudiés en détail. Encoder-decoder et autres architectures pourraient avoir des scaling laws différentes.

6. **Pas de régularisation optimisée** : Le dropout est fixé à 10% sans optimisation. De meilleures techniques de régularisation pourraient modifier les relations.

### Limites pratiques

7. **Compute massif requis** : Même l'étude des scaling laws nécessite des ressources considérables (des centaines de runs). Inaccessible aux laboratoires académiques.

8. **Pas d'évaluation downstream** : Seule la cross-entropy loss est étudiée. La relation entre loss et performances sur des tâches réelles n'est pas établie.

9. **Batch size fixe dans la plupart des expériences** : Les résultats principaux utilisent un batch size fixe, rendant les comparaisons de compute imparfaites.

10. **Contexte court (1024 tokens)** : Les trends pourraient changer pour des contextes beaucoup plus longs.

### Remise en question ultérieure

11. **Chinchilla (2022) contredira certaines conclusions** : Hoffmann et al. montreront que N et D devraient croître au même rythme (N ∝ D ∝ C^0.5), pas N >> D comme suggéré ici. La différence vient probablement du régime d'entraînement (single epoch vs multi-epoch).

## Liens utiles

- **PDF annoté**: [PDF annoté](../../papers/2026/scaling_laws.pdf)
- **Article**: [Scaling Laws for Neural Language Models (arXiv)](https://arxiv.org/abs/2001.08361)
- **ArXiv**: [https://arxiv.org/abs/2001.08361](https://arxiv.org/abs/2001.08361)

## Notes perso

### Importance historique

Ce papier est **fondamental** pour comprendre l'ère des LLMs massifs. Il établit que :
1. Le scaling est prévisible et suit des lois simples
2. Les gros modèles ne sont pas juste "meilleurs" mais plus **efficients**
3. L'architecture compte moins que le scale

Ces insights ont directement motivé GPT-3 (même équipe, même année) et la course aux modèles massifs.

### Équations clés à retenir

**Les trois scaling laws de base** :
```
L(N) ∝ N^(-0.076)    (doubler N → -5% loss)
L(D) ∝ D^(-0.095)    (doubler D → -6% loss)
L(C) ∝ C^(-0.050)    (doubler C → -3% loss)
```

**Allocation compute-optimale** :
```
N_opt ∝ C^0.73    (privilégier la taille)
D_opt ∝ C^0.27    (économiser les données)
```

**Relation overfitting** :
```
D ≳ 5000 × N^0.74    (pour éviter l'overfitting)
```

### Lien avec GPT-3

Ce papier et GPT-3 partagent plusieurs auteurs (Kaplan, Brown, Child, Radford, Amodei). Les scaling laws ont directement informé la conception de GPT-3 :
- Modèle massif (175B) entraîné sur relativement peu de données (~300B tokens, soit ~1.7 epochs)
- Arrêt avant convergence
- Focus sur le in-context learning plutôt que le fine-tuning

### Chinchilla vs Kaplan

La "controverse" Chinchilla (2022) vs ce papier :

| Aspect | Kaplan 2020 | Chinchilla 2022 |
|--------|-------------|-----------------|
| N optimal | N ∝ C^0.73 | N ∝ C^0.50 |
| D optimal | D ∝ C^0.27 | D ∝ C^0.50 |
| Implication | Gros modèle, peu de données | Équilibré N ≈ D |

La différence vient probablement du régime d'entraînement :
- Kaplan : single epoch ou moins (pas de réutilisation des données)
- Chinchilla : multi-epoch possible

En pratique, Chinchilla semble plus accurate pour l'entraînement moderne, mais les deux perspectives sont valides dans leurs régimes respectifs.

### "More is different"

Citation importante du papier :
> "Smooth quantitative change can mask major qualitative improvements: 'more is different'."

Les auteurs anticipent que les améliorations continues de la loss pourraient cacher des **changements qualitatifs de capacité** - ce qui s'est avéré vrai avec les capacités émergentes observées dans GPT-3 et les modèles ultérieurs.
