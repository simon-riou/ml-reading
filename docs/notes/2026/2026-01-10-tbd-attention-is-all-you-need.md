---
title: "Attention Is All You Need"
authors: ["Ashish Vaswani", "Noam Shazeer", "Niki Parmar", "Jakob Uszkoreit", "Llion Jones", "Aidan N. Gomez", "Łukasz Kaiser", "Illia Polosukhin"]
venue: "NIPS"
year: 2017
citekey: vaswani2017attention
links:
  pdf: "https://arxiv.org/pdf/1706.03762"
  code: "https://github.com/tensorflow/tensor2tensor"
  project: ""
pdf_annotated: "../../papers/2026/transformers.pdf"
tags:
  areas: ["deep-learning", "nlp", "machine-translation"]
  methods: ["transformer", "self-attention", "multi-head-attention", "positional-encoding"]
  tasks: ["machine-translation", "sequence-modeling", "parsing"]
  status: "deep-read"
replication:
  repo: "https://github.com/tensorflow/tensor2tensor"
  data: "WMT 2014"
  env: ""
---

## TL;DR

Le **Transformer** est la première architecture de transduction de séquences reposant **entièrement sur l'attention** (self-attention), éliminant complètement la récurrence et les convolutions. L'architecture utilise un **encodeur et décodeur empilés** (N=6 layers chacun) avec **multi-head self-attention** et des feed-forward networks. Le mécanisme clé est le **scaled dot-product attention** : Attention(Q,K,V) = softmax(QK^T/√dk)V, utilisé en **8 têtes parallèles** pour capturer différents sous-espaces de représentation. Pour encoder l'ordre séquentiel sans récurrence, des **positional encodings sinusoïdaux** sont ajoutés aux embeddings. Sur WMT'14 EN→DE, le Transformer atteint **28.4 BLEU** (SOTA), et **41.8 BLEU** sur EN→FR, entraîné en seulement **3.5 jours sur 8 GPUs** (fraction du coût des modèles récurrents). L'architecture est **hautement parallélisable** (O(1) opérations séquentielles vs O(n) pour RNNs), apprend les dépendances longue-distance en **O(1) path length** vs O(n) pour RNNs, et généralise à d'autres tâches (parsing : 92.7 F1 sur WSJ). Les visualisations d'attention montrent que le modèle apprend des structures syntaxiques et sémantiques interprétables.

## Contexte

En 2017, les RNNs (LSTMs/GRUs) dominent le sequence modeling et la traduction automatique. Les modèles encoder-decoder avec attention (Bahdanau 2015, Luong 2015) sont state-of-the-art, mais conservent des **couches récurrentes** qui posent des problèmes majeurs :

**Problème de séquentialité** : Les RNNs factorisent le calcul le long des positions temporelles (h_t = f(h_{t-1}, x_t)), rendant la parallelisation **impossible** au sein d'un exemple. Pour les longues séquences, les contraintes mémoire limitent le batching, ralentissant drastiquement l'entraînement.

**Path length pour dépendances longues** : Dans un RNN, le signal entre deux positions distantes doit traverser O(n) opérations séquentielles, rendant difficile l'apprentissage de dépendances à longue distance. Les modèles convolutionnels (ByteNet, ConvS2S) améliorent cela (O(log_k(n)) pour convolutions dilatées), mais restent moins efficaces que O(1).

**Alternatives existantes** : ByteNet et ConvS2S utilisent des CNNs comme building blocks pour paralléliser le calcul, mais le nombre d'opérations pour relier deux positions croît avec leur distance (linéairement pour ConvS2S, logarithmiquement pour ByteNet). Self-attention a été utilisée pour lecture, résumé et entailment, mais **toujours en combinaison avec RNNs**.

Le Transformer est le **premier modèle** reposant **entièrement sur self-attention**, sans RNNs ni convolutions alignées sur la séquence, permettant une parallélisation massive et des chemins de dépendances O(1).

## Idées clés

1. **Scaled Dot-Product Attention** : Calcul efficace d'attention via produit matriciel : Attention(Q,K,V) = softmax(QK^T/√dk)V. Le **scaling par 1/√dk** empêche les grands produits scalaires qui poussent le softmax dans des régions à gradients quasi-nuls. Dot-product attention est bien plus rapide que l'additive attention (Bahdanau) grâce aux optimisations de multiplication matricielle.

2. **Multi-Head Attention** : Au lieu d'une seule attention avec d_model dimensions, projeter Q, K, V **h fois** (h=8 heads) avec des matrices apprises différentes vers dk, dk, dv dimensions (dk=dv=d_model/h=64). Exécuter l'attention en **parallèle** sur chaque tête, concaténer les sorties, et projeter. Permet au modèle d'attendre conjointement à des informations de **différents sous-espaces de représentation** à différentes positions, là où une seule tête moyenne et inhibe cela.

3. **Self-Attention** : Relier différentes positions d'une **même séquence** pour en calculer une représentation. Dans l'encodeur : chaque position peut attendre à toutes les positions de la layer précédente. Dans le décodeur : masking pour auto-régressivité (position i ne peut attendre qu'aux positions ≤i). Complexité O(n²·d) mais **O(1) opérations séquentielles** et **O(1) path length** pour toute paire de positions.

4. **Encoder-Decoder Attention** : Dans le décodeur, une 3ème sub-layer où **Q vient du décodeur, K et V viennent de l'encodeur**. Chaque position du décodeur peut attendre à toutes les positions de la séquence source, mimant l'attention classique dans seq2seq mais sans RNN.

5. **Positional Encoding** : Sans récurrence ni convolution, le modèle n'a **aucune notion de l'ordre**. Injection d'informations de position via des fonctions sinusoïdales : PE(pos, 2i) = sin(pos/10000^(2i/d_model)), PE(pos, 2i+1) = cos(...). Les wavelengths forment une progression géométrique, permettant au modèle d'apprendre à attendre par positions relatives (PE_{pos+k} est une fonction linéaire de PE_pos). Alternative: learned embeddings (performances quasi-identiques).

6. **Residual Connections + Layer Normalization** : Autour de chaque sub-layer : LayerNorm(x + Sublayer(x)). Facilite le gradient flow dans les 6 layers empilées. Toutes les sub-layers et embeddings ont dimension d_model=512 pour faciliter les connexions résiduelles.

7. **Position-wise Feed-Forward Networks** : FFN(x) = max(0, xW1 + b1)W2 + b2, appliqué indépendamment à chaque position. Deux transformations linéaires avec ReLU entre, dimensions d_model=512 → d_ff=2048 → d_model=512. Équivalent à deux convolutions kernel size 1.

8. **Parallélisation massive** : Self-attention connecte toutes les positions avec O(1) opérations séquentielles (vs O(n) pour RNNs). La complexité per-layer O(n²·d) est plus rapide que RNN O(n·d²) quand n<d (généralement vrai pour word-piece representations). Permet un entraînement **beaucoup plus rapide** sur GPU.

## Méthode

**Architecture globale** : Encoder-decoder, chacun avec N=6 layers identiques empilées.

**Encoder** (par layer) :
1. Multi-head self-attention (8 heads)
2. Position-wise feed-forward network (FFN)
3. Residual connection + LayerNorm autour de chaque sub-layer

**Décodeur** (par layer) :
1. **Masked** multi-head self-attention (masking des positions futures)
2. Multi-head encoder-decoder attention (Q du décodeur, K/V de l'encodeur)
3. Position-wise FFN
4. Residual + LayerNorm autour de chaque sub-layer

**Scaled Dot-Product Attention** :
```
Attention(Q, K, V) = softmax(QK^T / √dk) V
```
Q, K ∈ ℝ^(n×dk), V ∈ ℝ^(n×dv). Le scaling 1/√dk évite que les dot products deviennent trop grands (variance dk sans scaling).

**Multi-Head Attention** :
```
MultiHead(Q, K, V) = Concat(head₁, ..., head_h) W^O
où head_i = Attention(QW^Q_i, KW^K_i, VW^V_i)
```
Projections : W^Q_i, W^K_i ∈ ℝ^(d_model×dk), W^V_i ∈ ℝ^(d_model×dv), W^O ∈ ℝ^(h·dv×d_model)
h=8, dk=dv=d_model/h=64. Coût total similaire à single-head full dimensionality.

**Position-wise FFN** :
```
FFN(x) = max(0, xW₁ + b₁)W₂ + b₂
```
d_model=512, d_ff=2048. Même transformation pour toutes les positions, paramètres différents par layer.

**Positional Encoding** :
```
PE(pos, 2i) = sin(pos / 10000^(2i/d_model))
PE(pos, 2i+1) = cos(pos / 10000^(2i/d_model))
```
Ajouté aux input/output embeddings. Permet d'extrapoler à des longueurs > training.

**Détails d'implémentation** :
- **Base model** : d_model=512, d_ff=2048, h=8, dk=dv=64, N=6, Pdrop=0.1
- **Big model** : d_model=1024, d_ff=4096, h=16, N=6, Pdrop=0.3 (EN-FR : 0.1)
- **Dataset** : WMT'14 EN-DE (4.5M sentence pairs, 37k BPE vocab), EN-FR (36M pairs, 32k wordpiece vocab)
- **Batching** : ~25k source + 25k target tokens par batch
- **Hardware** : 8× NVIDIA P100 GPUs
- **Optimiseur** : Adam (β₁=0.9, β₂=0.98, ε=10⁻⁹)
- **Learning rate** : lrate = d_model^(-0.5) · min(step^(-0.5), step · warmup_steps^(-1.5)), warmup=4000 steps
- **Régularisation** : Residual dropout (Pdrop=0.1), Label smoothing (ε_ls=0.1)
- **Entraînement** : Base model 100k steps (12h), Big model 300k steps (3.5 jours)
- **Décodage** : Beam search (beam=4, length penalty α=0.6), checkpoint averaging (5 derniers pour base, 20 pour big)
- **Weight sharing** : Même matrice d'embeddings pour input/output embeddings et pre-softmax linear, multipliée par √d_model dans les embeddings

## Résultats

**Traduction automatique (WMT'14)** :

| Modèle | EN-DE BLEU | EN-FR BLEU | Training Cost (FLOPs) |
|--------|-----------|-----------|---------------------|
| ByteNet | 23.75 | - | - |
| GNMT + RL | 24.6 | 39.92 | 2.3×10¹⁹ / 1.4×10²⁰ |
| ConvS2S | 25.16 | 40.46 | 9.6×10¹⁸ / 1.5×10²⁰ |
| MoE | 26.03 | 40.56 | 2.0×10¹⁹ / 1.2×10²⁰ |
| ConvS2S Ensemble | 26.36 | 41.29 | 7.7×10¹⁹ / 1.2×10²¹ |
| **Transformer (base)** | **27.3** | **38.1** | **3.3×10¹⁸** |
| **Transformer (big)** | **28.4** | **41.8** | **2.3×10¹⁹** |

**Observations clés** :

1. **SOTA EN→DE** : Transformer big atteint **28.4 BLEU**, surpassant tous les modèles précédents (y compris ensembles) de **>2 BLEU**. Entraîné en 3.5 jours sur 8 P100 GPUs.

2. **SOTA EN→FR** : **41.8 BLEU** (single model), surpassant tous les modèles publiés à **<1/4 du coût d'entraînement** du précédent SOTA.

3. **Efficacité computationnelle** : Base model (27.3 BLEU EN-DE) entraîné en **12h** avec un coût de **3.3×10¹⁸ FLOPs**, fraction du coût des modèles compétitifs (ConvS2S : 9.6×10¹⁸, GNMT : 2.3×10¹⁹).

4. **Même le base model bat tous les ensembles précédents** à une fraction du coût.

**Ablations (Table 3)** :

- **Nombre de heads** : Single-head 0.9 BLEU pire que h=8 (best). Trop de heads (h=32) dégrade aussi.
- **Dimension dk** : Réduire dk nuit à la qualité (suggère que déterminer la compatibilité n'est pas facile).
- **Model size** : Bigger is better (d_model=1024, d_ff=4096 : 26.0 BLEU vs 25.8 base).
- **Dropout crucial** : Sans dropout, -1.2 BLEU. Avec P=0.2, -0.3 BLEU.
- **Positional encoding** : Learned vs sinusoidal quasi-identiques (25.7 vs 25.8 BLEU).

**English Constituency Parsing (WSJ)** :

| Parser | Training | WSJ F1 |
|--------|----------|---------|
| Vinyals & Kaiser (RNN seq2seq) | WSJ only | 88.3 |
| Transformer (4 layers, d_model=1024) | WSJ only | **91.3** |
| Transformer (4 layers) | semi-supervised (17M sentences) | **92.7** |
| Dyer et al. (RNN Grammar) | generative | 93.3 |

Le Transformer **généralise remarquablement** à d'autres tâches sans tuning task-specific, surpassant BerkeleyParser (90.4) même avec seulement 40K phrases WSJ.

**Visualisations d'attention** :

- Les différentes têtes **apprennent des tâches distinctes** (dépendances longues, résolution anaphorique, structure syntaxique).
- Layer 5 montre des têtes capturant "making...more difficult" (dépendance longue), "its" → "Law" (anaphore).
- Les visualisations révèlent que le modèle apprend des structures **linguistiquement plausibles**, rendant le modèle plus **interprétable** que les RNNs.

## Limites

1. **Complexité quadratique en longueur de séquence** : Self-attention a une complexité O(n²·d) où n est la longueur. Pour des séquences très longues (documents, images, audio, vidéo), cela devient **prohibitif** en mémoire et calcul. Les auteurs suggèrent restricted self-attention (neighborhood de taille r) pour réduire à O(r·n·d), mais cela augmente le path length à O(n/r). Non exploré dans le paper.

2. **Pas de traitement natif des séquences très longues** : Le modèle est testé sur des phrases 15-40 mots. Pour des documents entiers (milliers de tokens), la complexité O(n²) rend l'approche impraticable sans modifications architecturales (sparse attention, local attention, etc.).

3. **Dépendance aux positional encodings** : Le modèle n'a **aucune notion intrinsèque de l'ordre**. Si les positional encodings échouent à capturer l'information positionnelle correctement, les performances s'effondrent. Bien que les sinusoids fonctionnent, le choix optimal de positional encoding reste un sujet ouvert.

4. **Architecture non-optimisée pour d'autres modalités** : Le paper se concentre sur le texte. Extension à la vision, audio, vidéo nécessite des adaptations non triviales (comment tokeniser? comment gérer la structure 2D/3D?). Mentionné comme travail futur mais non exploré.

5. **Beam search standard** : Le décodage utilise beam search basique (beam=4). Des stratégies plus sophistiquées (diverse beam search, nucleus sampling pour génération) pourraient améliorer la qualité, mais ne sont pas explorées.

6. **Pas d'analyse théorique** : Le paper ne fournit pas d'explication théorique formelle de **pourquoi** self-attention fonctionne si bien. Intuition (parallélisation, O(1) path length) est claire, mais manque de garanties de convergence ou d'analyse de capacité du modèle.

7. **Génération séquentielle reste auto-régressive** : Bien que l'encodeur soit parallèle, le décodeur génère **un token à la fois** (O(n) steps séquentiels à l'inference). "Making generation less sequential" est mentionné comme objectif futur mais non résolu ici.

8. **Interprétabilité limitée malgré les visualisations** : Bien que les visualisations d'attention montrent des patterns syntaxiques, il est difficile de garantir que le modèle **s'appuie** réellement sur ces patterns vs corrélations spurieuses. Les attention maps peuvent être trompeuses.

9. **Pas de comparaison avec Bahdanau attention dans l'architecture** : Le paper compare avec additive attention (Bahdanau) pour justifier dot-product, mais ne teste pas une architecture Transformer utilisant additive attention à la place de scaled dot-product pour valider le choix empiriquement.

10. **Résultats parsing limités** : Sur parsing, le Transformer (92.7 F1 semi-supervised) reste **inférieur** au RNN Grammar génératif (93.3 F1). Suggère que pour certaines tâches structurées, les inductive biases des RNNs pourraient encore être bénéfiques.

11. **Coût mémoire des multi-heads** : Bien que le coût computationnel total soit similaire à single-head, stocker h=8 matrices d'attention augmente les besoins mémoire, limitant potentiellement la taille de batch ou la longueur de séquence.

12. **Aucune garantie de généralisation hors-domaine** : Le modèle est testé sur WMT'14 (news) et WSJ (finance). Performances sur domaines très différents (médical, légal, low-resource languages) inconnues.

## Liens utiles

- **PDF annoté**: [PDF annoté](../../papers/2026/transformers.pdf)
- **Article**: [Attention Is All You Need (PDF)](https://arxiv.org/pdf/1706.03762)
- **ArXiv**: [arXiv:1706.03762](https://arxiv.org/abs/1706.03762)
- **Code**: [tensor2tensor](https://github.com/tensorflow/tensor2tensor)

## Notes perso

Toutes mes notes interessantes sont sur le pdf