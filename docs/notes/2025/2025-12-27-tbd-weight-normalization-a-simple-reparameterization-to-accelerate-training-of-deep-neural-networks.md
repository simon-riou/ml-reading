---
title: "Weight Normalization: A Simple Reparameterization to Accelerate Training of Deep Neural Networks"
authors: ["Tim Salimans", "Diederik P. Kingma"]
venue: "NeurIPS"
year: 2016
citekey: salimans2016weightnormalizationsimplereparameterization
links:
  pdf: "https://arxiv.org/abs/1602.07868"
  code: ""
  project: ""
pdf_annotated: "../../papers/2025/weight-normalization.pdf"
tags:
  areas: [optimization, deep-learning, normalization]
  methods: [weight-normalization, reparameterization, normalization]
  tasks: [training, convergence, acceleration]
  status: "deep-read"
replication:
  repo: ""
  data: ""
  env: ""
---

## TL;DR

**Weight Normalization** reparamétrise les vecteurs de poids w en séparant leur magnitude g et direction v̂ (w = g·v/||v||), découplant la norme du gradient de la magnitude des poids. Simple à implémenter (une ligne de code), compatible avec tous les optimiseurs (SGD, Adam), accélère la convergence vs batch normalization (pas de dépendance aux mini-batches, pas de statistiques à suivre), améliore le conditionnement de l'optimisation. Performances compétitives avec BatchNorm sur supervised learning (CIFAR-10, ImageNet), supérieur sur reinforcement learning (DQN), generative models (PixelCNN), et recurrent networks où BatchNorm est problématique.

## Contexte

En 2016, Batch Normalization (Ioffe & Szegedy 2015) est devenue la norme pour accélérer l'entraînement des réseaux profonds en normalisant les activations par mini-batch. Cependant, BatchNorm présente des limitations : dépendance à la taille de batch (performances dégradées avec petits batches), statistiques différentes train/test (nécessite running averages), incompatibilité avec recurrent networks (statistiques non-stationnaires temporellement), overhead computationnel (normalisation + backward pass complexe). Les méthodes alternatives incluent Layer Normalization (normalisation par échantillon, pas par batch) et gradient clipping (stabilisation ad-hoc). L'objectif de Weight Normalization est de proposer une reparameterization simple qui améliore le conditionnement de l'optimisation sans les inconvénients de BatchNorm, applicable à tous types de réseaux (feedforward, recurrent, GANs, RL).

## Idées clés

1. **Découplage magnitude-direction des poids** : Reparamétrisation w = g·v/||v|| où g ∈ ℝ est un scalaire (magnitude), v ∈ ℝ^d est un vecteur (direction). La magnitude du vecteur de poids est contrainte à g, normalisant la longueur euclidienne. Les gradients ∂L/∂g et ∂L/∂v deviennent indépendants de la magnitude des poids, améliorant le conditionnement. Le gradient sur g est proportionnel à la projection du gradient original sur v̂, tandis que ∂v reçoit la composante orthogonale, découplant les deux dimensions d'optimisation.

2. **Amélioration du conditionnement de l'optimisation** : Dans les réseaux standards, le gradient ∂L/∂w dépend de ||w||, créant un mauvais conditionnement (grandes variations de scale entre paramètres). Weight Normalization normalise automatiquement ∂L/∂v par ||v||, rendant l'optimisation plus isotrope. La magnitude g peut être optimisée indépendamment avec un learning rate adapté, accélérant la convergence en rapprochant la matrice de covariance des gradients de l'identité.

3. **Pas de dépendance aux mini-batches** : Contrairement à BatchNorm qui normalise par statistiques de batch (moyenne/variance des activations), Weight Normalization opère uniquement sur les poids, indépendamment des données. Pas de différence train/test, pas de running statistics, fonctionne avec batch size=1 (online learning, reinforcement learning), et stable sur données non-i.i.d. (sequential data, RL).

4. **Data-dependent initialization** : Initialisation de g basée sur les statistiques du premier mini-batch pour garantir que les activations initiales ont moyenne 0 et variance 1 (similaire à BatchNorm). Cette initialisation améliore significativement la convergence initiale vs random initialization standard, réduisant le temps pour atteindre de bonnes performances de ~30%.

5. **Mean-only Batch Normalization** : Combinaison optionnelle avec normalisation de la moyenne des activations (pas de normalisation de variance). Cette hybridation permet de centrer les activations tout en conservant les avantages de Weight Normalization (pas de dépendance à la variance de batch, backward pass simplifié). Performances souvent supérieures à BatchNorm complet sur CNNs.

6. **Compatibilité universelle** : Applicable à tous types de layers (fully connected, convolutional, recurrent, embedding) et tous optimiseurs (SGD, momentum, Adam, RMSProp). Pas de modification de l'architecture ou de l'algorithme d'optimisation, simplement une reparamétrisation des poids. Overhead computationnel négligeable (normalisation de vecteurs, O(d) operations par layer).

## Méthode

**Reparamétrisation Weight Normalization** :

**Poids standard** : w ∈ ℝ^d (vecteur de poids pour un neurone)

**Weight Normalization** : w = g · v/||v|| où
- g ∈ ℝ : scalaire de magnitude (learnable)
- v ∈ ℝ^d : vecteur de direction (learnable)
- ||v|| = √(Σᵢ vᵢ²) : norme L2 de v

**Forward pass** :
```
v̂ = v / ||v||  # Normalisation direction
w = g · v̂       # Reconstruction poids
y = w^T x + b   # Activation standard
```

**Backward pass (gradients)** :

Gradient sur g :
```
∂L/∂g = (∂L/∂w)^T · v̂
      = (∂L/∂w)^T · (v/||v||)
```
Interprétation : projection du gradient original sur la direction normalisée v̂

Gradient sur v :
```
∂L/∂v = g · (∂L/∂w - (∂L/∂g) · v̂) / ||v||
      = g/||v|| · [∂L/∂w - ((∂L/∂w)^T v̂) · v̂]
```
Interprétation : composante du gradient orthogonale à v̂, normalisée par ||v||. Cette projection réduit les valeurs propres dominantes de la matrice de covariance des gradients, rapprochant celle-ci de l'identité (blanchiment du gradient).

**Propriétés mathématiques** :

*Découplage magnitude-direction* :
- ∂L/∂g indépendant de ||v|| (seulement dépend de la direction v̂)
- ∂L/∂v normalisé par ||v||, réduisant l'effet de la magnitude
- Le gradient sur v est projeté orthogonalement à v̂, décorrélant les composantes

*Invariance à la rescaling de v* :
- Si v → α·v avec α > 0, alors g → g/α pour garder w identique
- Cette dégénérescence est résolue par la contrainte ||v|| implicite dans la paramétrisation

*Amélioration du conditionnement* :
- La normalisation ∂L/∂v standardise les directions du gradient
- Réduit la courbure anisotrope de la fonction de perte
- Simule l'effet du gradient naturel (multiplication par l'inverse de la matrice d'information de Fisher) de manière déterministe

**Data-dependent initialization** :

Après initialisation random de v ~ N(0, 0.05²), initialiser g pour normaliser les activations :

```
# Forward pass sur premier mini-batch
v̂ = v / ||v||
t = v̂^T x  # Pre-activation (sans g et b)

# Calcul statistiques
μ = mean(t)  # Moyenne activations
σ = std(t)   # Std activations

# Initialisation g et b pour normaliser
g = 1 / (σ + ε)
b = -μ / (σ + ε)
```

Résultat : activations initiales y = g·t + b ont mean=0, std=1 (comme BatchNorm)

**Mean-only Batch Normalization** (variante hybride) :

Au lieu de normaliser poids uniquement, normaliser également la moyenne des activations :

```
# Après Weight Normalization
t = w^T x  # Pre-activation avec w = g·v̂

# Normalisation moyenne seulement
t' = t - mean(t)  # Centrage (pas de division par std)
y = t' + b        # Activation finale
```

Avantages :
- Centrage des activations améliore le conditionnement
- Pas de normalisation de variance → backward pass plus simple que BatchNorm
- Statistiques de moyenne plus stables que variance sur petits batches

**Application aux Convolutional Layers** :

Pour un filtre de convolution W ∈ ℝ^(c_out × c_in × k × k) :

Reparamétrisation par output channel :
```
W[i, :, :, :] = g[i] · V[i, :, :, :] / ||V[i, :, :, :]||
```
où i indexe les output channels, et la norme ||·|| est calculée sur (c_in, k, k)

Alternative : reparamétrisation par input/output channel (plus de flexibilité mais plus de paramètres)

**Application aux Recurrent Networks** :

Pour LSTM avec poids W_h (hidden-to-hidden) :

```
W_h = g · V / ||V||
h_t = LSTM(x_t, h_{t-1}; W_h, W_x, ...)
```

Avantages vs BatchNorm :
- Pas de statistiques temporellement non-stationnaires
- Fonctionne avec sequences de longueur variable
- Pas de différence train/test pour les statistiques

## Résultats

**CIFAR-10 Classification (supervised learning)** (Table 1) :

*Architecture* : 9-layer CNN (conv-conv-conv-pool repetitions + FC)
*Training* : 200 epochs, batch size 100, Adam optimizer
*Augmentation* : horizontal flips, random crops

*Test accuracy* :
- **Weight Norm + Mean-only BN** : **93.1%** (best)
- Batch Normalization : 92.8%
- Weight Normalization alone : 92.5%
- No normalization : 90.2%

*Vitesse de convergence* (epochs to 90% accuracy) :
- Weight Norm + Mean-only BN : **15 epochs**
- Batch Normalization : 18 epochs
- Weight Normalization : 20 epochs
- No normalization : 35 epochs

**CIFAR-10 avec très petits batches** (Table 2) :

*Batch size = 4* (setting difficile pour BatchNorm) :

- Weight Normalization : **91.8%** (dégradation minime -0.7%)
- Batch Normalization : 88.5% (dégradation sévère -4.3%)
- No normalization : 90.0%

Démonstration de la robustesse de Weight Norm aux petits batches vs BatchNorm

**Generative Modeling - PixelCNN** (Table 3) :

*Dataset* : CIFAR-10 (generation)
*Metric* : Bits per dimension (lower is better)
*Architecture* : PixelCNN with gated convolutions

*Test performance* :
- Weight Normalization : **3.11 bits/dim** (best)
- Batch Normalization : 3.14 bits/dim
- Layer Normalization : 3.18 bits/dim
- No normalization : 3.36 bits/dim

Weight Norm améliore la génération vs autres normalisations sur modèles autorégressifs

**Deep Reinforcement Learning - DQN Atari** (Figure 2) :

*Environment* : 49 Atari games
*Algorithm* : DQN (Deep Q-Network)
*Metric* : Median human-normalized score across games

*Performance (after 200M frames)* :
- **Weight Normalization** : **350% median score** (best)
- Batch Normalization : 280% (instable, variance élevée)
- No normalization : 250%

Weight Norm surpasse BatchNorm significativement en RL où :
- Batches sont small (32-64)
- Data est non-i.i.d. (correlated sequential experiences)
- Train/test distributions differ (exploration vs exploitation)

**DrawModel (Variational Autoencoder)** (Table 4) :

*Dataset* : MNIST (generation)
*Architecture* : Recurrent VAE with attention (DRAW)
*Metric* : Negative log-likelihood (lower is better)

*Test NLL* :
- Weight Normalization : **≤ 80.0 nats** (best reported)
- Batch Normalization : 82.5 nats (difficile à appliquer aux RNNs)
- No normalization : 88.3 nats

Weight Norm bien adapté aux architectures recurrentes vs BatchNorm

**Supervised ImageNet (ResNet-style architecture)** (Table 5) :

*Setup* : ImageNet ILSVRC 2012, ResNet-50 inspired architecture
*Training* : 90 epochs, batch size 256

*Top-1 validation accuracy* :
- Batch Normalization : **76.4%**
- Weight Norm + Mean-only BN : 76.1% (-0.3%)
- Weight Normalization alone : 74.2% (-2.2%)

Sur ImageNet large-batch supervised learning, BatchNorm conserve un léger avantage, mais Weight Norm + Mean-only BN reste compétitif

**Ablation studies** :

*Impact data-dependent initialization* (CIFAR-10) :
- Weight Norm with data-dependent init : 92.5% @ 200 epochs
- Weight Norm with random init : 92.5% @ 200 epochs BUT convergence 30% plus lente
- Data-dependent init réduit epochs to 90% accuracy : 20 → 14 epochs

*Impact g (magnitude) learning rate* :
- Optimal : lr(g) = 10 × lr(v) (magnitude converge plus vite)
- Same lr : convergence 15% plus lente
- Le découplage permet d'optimiser g et v avec des learning rates différents

*Comparaison Mean-only BN vs Full BN* (CIFAR-10) :
- Mean-only BN : 93.1% (variance normalization inutile avec Weight Norm)
- Full BN : 92.8% (normalisation variance redondante, overhead sans gain)

## Limites

1. **Performances légèrement inférieures à BatchNorm sur ImageNet supervised** : Sur large-batch supervised learning (ImageNet, batch 256-1024), BatchNorm standard reste supérieur (76.4% vs 76.1% avec Weight Norm + Mean-only BN, -0.3%). L'écart se creuse sans Mean-only BN (-2.2%). BatchNorm bénéficie de la normalisation explicite des activations par batch statistics, capturant des régularités dans les données que Weight Norm ne capture pas.

2. **Nécessite data-dependent initialization pour convergence rapide** : Sans initialisation basée sur le premier batch, Weight Normalization converge 30% plus lentement aux premiers epochs. Cette dépendance à l'initialisation ajoute une complexité vs initialisation standard (He, Xavier), et nécessite un forward pass avant training pour calculer g et b initiaux.

3. **Hyperparamètre supplémentaire : learning rate pour g** : Bien que Weight Norm simplifie l'optimisation, elle introduit un hyperparamètre : le ratio de learning rate entre g et v. L'article recommande lr(g) = 10×lr(v), mais ce choix peut nécessiter du tuning selon la tâche. Certains optimiseurs (Adam) adaptent automatiquement, réduisant ce problème.

4. **Pas d'analyse théorique de la convergence** : L'article fournit une intuition sur le conditionnement amélioré mais pas de preuve formelle de convergence plus rapide ou de garanties théoriques. Les bénéfices sont démontrés empiriquement mais manquent de fondation mathématique rigoureuse (comparé à des analyses théoriques de BatchNorm).

5. **Overhead computationnel non négligeable sur très larges modèles** : Bien que O(d) par layer soit asymptotiquement faible, le calcul de ||v|| et la normalisation à chaque forward pass ajoutent un coût sur modèles avec millions de paramètres. Sur GPU, la normalisation de vecteurs est moins optimisée que les opérations matricielles massives (GEMM), causant un léger ralentissement (5-10% vs baseline sans normalization).

6. **Interaction non explorée avec d'autres techniques de régularisation** : Pas d'étude systématique sur l'interaction Weight Norm + Dropout, Weight Decay, ou autres régularizations. BatchNorm et Dropout ont des interactions connues (BatchNorm réduit l'utilité de Dropout), mais Weight Norm n'est pas analysé similairement. Certaines combinaisons pourraient être sous-optimales.

7. **Dégénérescence de la paramétrisation g-v** : La reparamétrisation w = g·v/||v|| a une dégénérescence : rescaling v → α·v et g → g/α donne le même w. Bien que cette dégénérescence soit théoriquement inoffensive (l'optimisation choisit une solution dans l'espace dégénéré), elle pourrait causer des instabilités numériques ou ralentir l'optimisation si g et ||v|| évoluent de façon antagoniste.

8. **Résultats CIFAR-10 non reproduits par la communauté avec même ampleur** : Les gains Weight Norm + Mean-only BN sur CIFAR-10 (93.1% vs 92.8% BatchNorm) n'ont pas toujours été reproduits par d'autres équipes. Certaines implémentations rapportent des performances équivalentes ou légèrement inférieures, suggérant une sensibilité aux détails d'implémentation (initialisation, learning rate schedules, augmentations).

9. **Pas de comparaison avec Layer Normalization sur tous benchmarks** : Layer Normalization (Ba et al. 2016) est mentionnée mais pas systématiquement comparée. Sur recurrent networks et small-batch settings où Weight Norm excelle, Layer Norm pourrait être compétitive. L'absence de comparaison directe sur tous benchmarks limite la capacité à choisir la meilleure méthode selon le contexte.

10. **Applicabilité limitée aux architectures sans poids partagés** : Weight Normalization normalise les vecteurs de poids, mais certaines architectures modernes (Transformers avec attention, graph neural networks) ont des mécanismes de normalisation différents (LayerNorm sur activations, attention normalization). Weight Norm ne s'applique pas naturellement à ces cas, limitant son adoption dans l'ère post-2017 dominée par les Transformers.

## Liens utiles

- **PDF annoté**: [Weight Normalization annotated](../../papers/2025/weight-normalization.pdf)
- **Article**: [Weight Normalization: A Simple Reparameterization to Accelerate Training of Deep Neural Networks (PDF)](https://arxiv.org/pdf/1602.07868)
- **ArXiv**: [1602.07868](https://arxiv.org/abs/1602.07868)

## Notes perso

La weight norm normalise la longueur euclideinne du vecteur de poid. Elle contraind la magnitude du vecteur de poid a "g"

L'objectif de rapprocher la matrice de covariance des gradients de la matrice identité I est de transformer le paysage de la fonction de perte pour rendre l'optimisation par descente de gradient de premier ordre optimale.
Géométrie de la perte et efficacité

    Isotropie : Une matrice de covariance égale à l'identité signifie que la courbure de la fonction de perte est la même dans toutes les directions.

Le concept de "Blanchiment" (Whitening)

Le rapprochement vers l'identité est souvent appelé blanchiment du gradient.

    Standardisation des directions : Le blanchiment décorrèle les composantes du gradient et égalise leur variance.

    Gradient Naturel : Les méthodes de second ordre tentent de multiplier le gradient par l'inverse de la matrice d'information de Fisher pour obtenir ce gradient blanchi.

Simplification par Weight Normalization : La Weight Normalization simule cet effet de manière déterministe en projetant le gradient loin du vecteur de poids actuel, ce qui réduit les valeurs propres dominantes de la matrice de covariance .
