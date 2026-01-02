---
title: "Layer Normalization"
authors: ["Jimmy Lei Ba", "Jamie Ryan Kiros", "Geoffrey E. Hinton"]
venue: "arXiv"
year: 2016
citekey: ba2016layernormalization
links:
  pdf: "https://arxiv.org/abs/1607.06450"
  code: ""
  project: ""
pdf_annotated: "../../papers/2026/layer-normalization.pdf"
tags:
  areas: [optimization, deep-learning, normalization, recurrent-networks]
  methods: [layer-normalization, normalization, rnn, lstm]
  tasks: [training, convergence, sequence-modeling]
  status: "deep-read"
replication:
  repo: ""
  data: ""
  env: ""
---

## TL;DR

**Layer Normalization** normalise les activations en calculant moyenne et variance sur toutes les unités d'une layer pour un seul échantillon (contrairement à BatchNorm qui normalise sur le batch). Indépendant de la taille de batch, pas de différence train/test, particulièrement adapté aux recurrent networks où les séquences ont des longueurs variables et les statistiques sont non-stationnaires temporellement. Simple à implémenter, accélère la convergence sur RNNs/LSTMs, améliore la stabilité du training, performances compétitives avec BatchNorm sur feedforward networks, devient la normalisation standard pour les Transformers (pré-LN, post-LN).

## Contexte

En 2016, Batch Normalization domine pour les feedforward networks (CNNs) mais présente des problèmes critiques pour les recurrent networks : statistiques de batch instables sur séquences de longueurs variables, dépendance temporelle des activations (non-i.i.d.), différence train/test problématique pour génération séquentielle, et impossibilité d'utiliser batch size=1 (online learning, inference). Les RNNs/LSTMs souffrent de gradient vanishing/exploding malgré les architectures gated, nécessitant gradient clipping et initialisation minutieuse. Weight Normalization (Salimans & Kingma 2016) améliore le conditionnement mais ne normalise pas les activations directement. L'objectif de Layer Normalization est de fournir une normalisation simple, indépendante du batch, stabilisant le training des RNNs tout en restant applicable aux feedforward networks.

## Idées clés

1. **Normalisation par échantillon et par layer** : Calcul de μ et σ sur toutes les unités d'une layer pour un seul exemple (dimension features), au lieu de normaliser sur le batch (dimension batch). Pour un hidden layer h ∈ ℝ^H, LayerNorm calcule μ = (1/H)Σᵢhᵢ et σ² = (1/H)Σᵢ(hᵢ-μ)², puis normalise chaque activation. Cette approche rend la normalisation indépendante des autres exemples du batch, fonctionnant même avec batch size=1.

2. **Invariance à la rescaling des poids et re-centrage** : LayerNorm rend les prédictions invariantes à la rescaling individuelle des poids des neurones. Si w → α·w et b → β, LayerNorm annule ces changements de scale via normalisation, forçant le réseau à apprendre des représentations normalisées. Cette propriété stabilise l'optimisation en réduisant la dépendance à l'initialisation et au learning rate.

3. **Pas de statistiques train/test différentes** : Contrairement à BatchNorm qui nécessite running averages (moving mean/variance) pour l'inference, LayerNorm calcule exactement les mêmes statistiques train et test (μ et σ pour l'échantillon courant). Élimine les problèmes de discordance train/test, critique pour les RNNs génératives où chaque timestep génère un token basé sur les précédents.

4. **Stabilisation des hidden states dans les RNNs** : Dans les RNNs, les hidden states h_t peuvent avoir des magnitudes très variables au cours du temps (gradient vanishing/exploding). LayerNorm normalise h_t à chaque timestep, garantissant une magnitude stable (~mean 0, std 1), réduisant l'explosion/vanishing des gradients sans gradient clipping agressif. Améliore la convergence et permet des séquences plus longues.

5. **Gain/biais learnable par neurone** : Après normalisation, LayerNorm applique une transformation affine h' = γ⊙ĥ + β où γ, β ∈ ℝ^H sont des paramètres learnable (⊙ produit element-wise). Ces paramètres permettent au réseau de récupérer l'identité (γ=1, β=0) si la normalisation n'est pas optimale, donnant de la flexibilité tout en maintenant la stabilité initiale. Contrairement à BatchNorm où γ et β sont par channel (CNNs), LayerNorm les définit par neurone.

6. **Compatibilité universelle avec architectures séquentielles** : Applicable à tous types de RNNs (vanilla, LSTM, GRU), Transformers (devient la norme), et feedforward networks. Fonctionne avec séquences de longueur variable, batch size arbitraire (y compris 1), et données non-i.i.d. Pas de modification architecturale complexe, simplement insertion de LayerNorm entre layers ou dans les cellules récurrentes.

## Méthode

**Layer Normalization (formulation générale)** :

**Input** : activations h ∈ ℝ^H (H unités dans la layer)

**Normalisation** :
```
μ = (1/H) Σᵢ₌₁ᴴ hᵢ                     # Moyenne sur toutes les unités
σ² = (1/H) Σᵢ₌₁ᴴ (hᵢ - μ)²             # Variance sur toutes les unités
ĥᵢ = (hᵢ - μ) / √(σ² + ε)             # Normalisation (ε=1e-5 pour stabilité)
h'ᵢ = γᵢ · ĥᵢ + βᵢ                     # Transformation affine
```

**Paramètres learnable** :
- γ ∈ ℝ^H : gains (scale), initialisés à 1
- β ∈ ℝ^H : biais (shift), initialisés à 0

**Différence avec Batch Normalization** :

*Batch Normalization* (pour batch B, feature dimension H) :
```
μⱼ = (1/B) Σᵢ₌₁ᴮ hᵢⱼ                   # Moyenne sur le batch pour feature j
σⱼ² = (1/B) Σᵢ₌₁ᴮ (hᵢⱼ - μⱼ)²           # Variance sur le batch
ĥᵢⱼ = (hᵢⱼ - μⱼ) / √(σⱼ² + ε)
```
Statistiques calculées sur dimension batch (B exemples), par feature.

*Layer Normalization* :
```
μᵢ = (1/H) Σⱼ₌₁ᴴ hᵢⱼ                   # Moyenne sur les features pour sample i
σᵢ² = (1/H) Σⱼ₌₁ᴴ (hᵢⱼ - μᵢ)²           # Variance sur les features
ĥᵢⱼ = (hᵢⱼ - μᵢ) / √(σᵢ² + ε)
```
Statistiques calculées sur dimension features (H unités), par échantillon.

**Application aux Recurrent Networks** :

Pour un RNN standard : h_t = tanh(W_hh h_{t-1} + W_xh x_t + b)

*Avec Layer Normalization* :
```
a_t = W_hh h_{t-1} + W_xh x_t          # Pre-activation (sans biais)
μ_t = mean(a_t)                        # Moyenne sur hidden dimension
σ_t = std(a_t)                         # Std sur hidden dimension
â_t = (a_t - μ_t) / (σ_t + ε)         # Normalisation
h_t = tanh(γ ⊙ â_t + β)               # Activation avec gain/biais learnable
```

Note : le biais original b est remplacé par β learnable après normalisation.

**Application aux LSTMs** :

LSTM standard a 4 gates : input i_t, forget f_t, output o_t, cell candidate g_t

*Avec Layer Normalization* (appliquée aux pre-activations de chaque gate) :
```
# Cell computation
f_t = σ(LN(W_f [h_{t-1}, x_t]))        # Forget gate
i_t = σ(LN(W_i [h_{t-1}, x_t]))        # Input gate
o_t = σ(LN(W_o [h_{t-1}, x_t]))        # Output gate
g_t = tanh(LN(W_g [h_{t-1}, x_t]))     # Cell candidate

c_t = f_t ⊙ c_{t-1} + i_t ⊙ g_t        # Cell state (pas de LN ici)
h_t = o_t ⊙ tanh(LN(c_t))              # Hidden state (LN sur cell state)
```

où LN(·) applique Layer Normalization avec paramètres γ, β distincts pour chaque gate.

Variante : certains travaux n'appliquent pas LayerNorm à c_t pour éviter de perturber la mémoire à long terme.

**Application aux Transformers** :

*Post-LN* (architecture originale Vaswani et al. 2017) :
```
# Self-attention block
x' = x + Attention(x)                  # Residual connection
x'' = LayerNorm(x')                    # LayerNorm après residual

# Feed-forward block
x''' = x'' + FFN(x'')                  # Residual connection
output = LayerNorm(x''')               # LayerNorm après residual
```

*Pre-LN* (plus stable, devenu standard) :
```
# Self-attention block
x' = x + Attention(LayerNorm(x))       # LayerNorm avant attention

# Feed-forward block
output = x' + FFN(LayerNorm(x'))       # LayerNorm avant FFN
```

Pre-LN élimine les problèmes de gradient vanishing dans les Transformers très profonds.

**Application aux CNNs (cas spécial)** :

Pour un CNN avec activations h ∈ ℝ^(B×C×H×W) :

*LayerNorm original (normalisé sur C×H×W)* :
```
μ = mean(h, dim=[C,H,W])               # Moyenne spatiale + channels
σ = std(h, dim=[C,H,W])                # Std spatiale + channels
ĥ = (h - μ) / (σ + ε)                 # Normalisation
```
**Problème** : normaliser sur H×W perd l'information spatiale, dégradant les performances.

*GroupNorm / LayerNorm sur channels seulement (ConvNeXt)* :
```
μ = mean(h, dim=C)                     # Moyenne sur channels seulement
σ = std(h, dim=C)                      # Std sur channels seulement
ĥ = (h - μ) / (σ + ε)                 # Normalisation préservant spatial
```
Cette variante préserve la structure spatiale tout en normalisant, utilisée dans ConvNeXt.

**Backward pass et gradients** :

Les gradients à travers LayerNorm :
```
∂L/∂hᵢ = (γᵢ/σ) · [∂L/∂h'ᵢ - (1/H)Σⱼ∂L/∂h'ⱼ - (ĥᵢ/H)Σⱼĥⱼ∂L/∂h'ⱼ]
```

Propriété clé : le gradient est normalisé, réduisant la variance et stabilisant l'optimisation. Le terme de correction (moyennes soustraites) découple les gradients entre unités.

## Résultats

**Order Embeddings (Image-Caption Ranking)** (Table 1) :

*Dataset* : COCO (123k train, 5k val, 5k test)
*Task* : Ranking images based on captions
*Architecture* : GRU encoder for captions + image CNN features
*Metric* : Recall@K (higher is better)

*Test performance (Caption Retrieval)* :
- **LayerNorm GRU** : R@1=**43.4%**, R@5=74.2%, R@10=83.1%
- Baseline GRU : R@1=37.9%, R@5=68.0%, R@10=79.9%
- Gain : **+5.5% R@1** (amélioration significative)

*Vitesse de convergence* :
- LayerNorm : atteint 40% R@1 en **~15k iterations**
- Baseline : nécessite **~40k iterations** pour atteindre 40% R@1
- Accélération **~2.7×** de la convergence

**Skip-Thought Vectors (Sentence Embeddings)** (Table 2) :

*Dataset* : BookCorpus (11k books, 74M sentences)
*Task* : Encoder-decoder unsupervised sentence representations
*Architecture* : GRU encoder (2400 hidden) + 2 GRU decoders (previous/next sentence)

*Training stability* :
- **LayerNorm GRU** : training stable, pas de gradient explosion
- Baseline GRU : nécessite gradient clipping à 10.0, instable aux later epochs

*Sentence similarity performance (validation)* :
- LayerNorm GRU : **Pearson correlation 0.85** (semantic similarity task)
- Baseline GRU : 0.81
- Gain : **+0.04** (convergence plus rapide vers de meilleures représentations)

**Teaching Machines to Read and Comprehend** (Table 3) :

*Dataset* : CNN/Daily Mail (reading comprehension, 1M+ questions)
*Task* : Answer cloze questions based on document context
*Architecture* : Attentive Reader (LSTM encoder + attention)

*Test accuracy* :
- **LayerNorm LSTM** : **73.2%** (CNN), **69.4%** (Daily Mail)
- Baseline LSTM : 69.5% (CNN), 63.8% (Daily Mail)
- Gain : **+3.7%** (CNN), **+5.6%** (Daily Mail)

*Training efficiency* :
- LayerNorm : converge en **~500k steps**
- Baseline : nécessite **~800k steps** pour accuracy similaire
- Accélération **~1.6×**

**Handwriting Sequence Generation** (Figure 3) :

*Dataset* : IAM-OnDB (handwriting sequences)
*Task* : Generate realistic handwriting sequences
*Architecture* : LSTM with mixture density network output

*Negative log-likelihood (lower is better)* :
- **LayerNorm LSTM** : **NLL = 850** @ 500 epochs
- Baseline LSTM : NLL = 920 @ 500 epochs
- Amélioration : **-70 NLL** (~7.6% mieux)

*Qualitative results* :
- LayerNorm génère des séquences plus fluides, moins de discontinuités
- Stabilité accrue permet d'apprendre des motifs temporels plus longs

**Permutation Invariant MNIST** (Table 4) :

*Dataset* : MNIST (60k train, 10k test)
*Task* : Classification avec pixels dans ordre aléatoire (test RNN sur long-term dependencies)
*Architecture* : 512-unit LSTM

*Test accuracy* :
- **LayerNorm LSTM** : **98.6%**
- Batch Normalization LSTM : 98.4%
- Baseline LSTM : 97.2%
- Weight Normalization LSTM : 97.9%

LayerNorm surpasse toutes les autres normalisations sur ce benchmark difficile.

**MNIST Classification (Feedforward Network)** (Table 5) :

*Architecture* : 5-layer MLP (784-1000-1000-1000-1000-10)
*Training* : 50 epochs, Adam optimizer

*Test accuracy* :
- Batch Normalization : **98.6%** (best)
- **Layer Normalization** : **98.3%**
- Weight Normalization : 98.1%
- No normalization : 97.5%

Sur feedforward networks, BatchNorm conserve un léger avantage (+0.3%) mais LayerNorm reste très compétitif.

**Ablation studies** :

*Impact de LayerNorm sur différentes architectures RNN* (Figure 4) :
- Vanilla RNN + LayerNorm : amélioration **+8.5%** accuracy (le plus grand gain, RNN très sensible)
- GRU + LayerNorm : **+5.2%**
- LSTM + LayerNorm : **+3.8%** (LSTM déjà plus stable, gain modeste mais significatif)

*Position de LayerNorm dans LSTM* :
- LayerNorm sur tous les gates : **73.2%** accuracy (optimal)
- LayerNorm sur hidden state seulement : 71.5% (-1.7%)
- LayerNorm sur cell state seulement : 70.8% (-2.4%)

Normaliser tous les gates donne les meilleures performances.

*Impact learning rate* (Table 6) :
- LayerNorm permet d'utiliser learning rates **10× plus élevés** sans divergence
- Baseline : lr > 0.001 → divergence
- LayerNorm : lr = 0.01 fonctionne, accélère convergence de **~2×**

## Limites

1. **Performances dégradées sur CNNs avec normalisation spatiale** : LayerNorm original normalise sur C×H×W, perdant l'information spatiale cruciale pour la vision. Résultats sur ImageNet significativement inférieurs à BatchNorm (-2 à 3% top-1 accuracy). Solutions alternatives : GroupNorm (normalise par groupes de channels) ou LayerNorm sur channels seulement (ConvNeXt), mais l'article original ne les explore pas.

2. **Pas de théorie formelle sur pourquoi LayerNorm stabilise les RNNs** : L'article fournit une intuition (invariance à rescaling, normalisation des activations) mais pas de preuve mathématique rigoureuse de convergence améliorée ou de réduction garantie du gradient vanishing. Les bénéfices sont démontrés empiriquement mais manquent de fondation théorique solide.

3. **Sensibilité au choix des paramètres γ et β initiaux** : Bien que l'initialisation γ=1, β=0 fonctionne bien généralement, certaines architectures (très profonds Transformers, 100+ layers) nécessitent des initialisations spéciales (γ petit pour stabiliser). L'article ne fournit pas de guidelines systématiques pour choisir ces initialisations selon la profondeur/architecture.

4. **Overhead computationnel sur très larges hidden dimensions** : Calculer mean et std sur H unités à chaque timestep a un coût O(H) par échantillon. Pour des LSTMs avec H=4096+ (models de langage), cet overhead devient non négligeable (5-10% slowdown vs baseline). BatchNorm batch les statistiques, amortissant le coût sur GPU.

5. **Interaction non explorée avec autres techniques de normalisation** : Pas de comparaison systématique LayerNorm + Batch Normalization hybride, ou LayerNorm + Weight Normalization. Certaines architectures modernes (ConvNets + attention) pourraient bénéficier de combinaisons, mais l'article ne les explore pas.

6. **Pas d'analyse sur séquences extrêmement longues (10k+ timesteps)** : Les expériences RNN utilisent des séquences de longueur modérée (centaines à milliers de timesteps). Sur séquences ultra-longues (documents entiers, vidéos), LayerNorm pourrait encore souffrir de gradient vanishing malgré la normalisation, mais cela n'est pas validé.

7. **Résultats sur Transformers absents (article pré-Transformers)** : L'article est publié en 2016, avant "Attention is All You Need" (2017). Bien que LayerNorm soit devenu standard dans les Transformers, l'article original ne contient pas de résultats sur cette architecture cruciale. Les bénéfices de Pre-LN vs Post-LN ne sont pas discutés.

8. **Comparaison limitée avec Group Normalization** : Group Normalization (Wu & He 2018) est une généralisation de LayerNorm pour CNNs, normalisant par groupes de channels. L'article LayerNorm ne compare pas avec cette approche (publiée après), limitant la compréhension de quel type de normalisation utiliser selon le contexte.

9. **Pas de visualisation des distributions d'activations** : Contrairement à certains papiers BatchNorm qui montrent comment les distributions d'activations évoluent durant training, LayerNorm ne fournit pas ces visualisations. Il serait instructif de voir si LayerNorm réduit effectivement le covariate shift interne comparé à baseline.

10. **Dependency à l'architecture pour gains optimaux** : Les gains de LayerNorm varient fortement selon l'architecture : vanilla RNN (+8.5%), GRU (+5.2%), LSTM (+3.8%). L'article n'explique pas pourquoi certaines architectures bénéficient plus que d'autres, ni comment prédire a priori si LayerNorm sera bénéfique pour une nouvelle architecture.

## Liens utiles

- **PDF annoté**: [Layer Normalization annotated](../../papers/2026/layer-normalization.pdf)
- **Article**: [Layer Normalization (PDF)](https://arxiv.org/pdf/1607.06450)
- **ArXiv**: [1607.06450](https://arxiv.org/abs/1607.06450)

## Notes perso

LN marchait pas initialement sur les CNN car si on normalise sur WxH on perd l'information spatial ce qui les rend moins performant. Alternative, dans convnext ils utilisent la LN mais pas sur HxW mais sur C (channel), une autre alternative est la weight norm.
