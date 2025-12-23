---
title: "Adam: A Method for Stochastic Optimization"
authors: ["Diederik P. Kingma", "Jimmy Ba"]
venue: "ICLR"
year: 2015
citekey: kingma2017adammethodstochasticoptimization
links:
  pdf: "https://arxiv.org/abs/1412.6980"
  code: ""
  project: ""
pdf_annotated: "../../papers/2025/adam.pdf"
tags:
  areas: [optimization, deep-learning]
  methods: [adaptive-learning-rate, momentum, stochastic-optimization]
  tasks: [training, convergence]
  status: "deep-read"
replication:
  repo: ""
  data: ""
  env: ""
---

## TL;DR

**Adam** (Adaptive Moment Estimation) combine les avantages de AdaGrad (adaptation du learning rate par paramètre) et RMSProp (moyennes mobiles des gradients au carré) en maintenant des moyennes mobiles exponentielles du premier et second moment des gradients. Simple à implémenter, efficace en mémoire, invariant à la mise à l'échelle des gradients, et bien adapté aux problèmes avec gradients bruités ou sparse. Devient l'optimiseur par défaut pour l'entraînement de réseaux de neurones profonds, surpassant SGD+momentum sur de nombreux benchmarks (logistic regression, MLPs, CNNs).

## Contexte

En 2014, l'optimisation stochastique de réseaux de neurones profonds repose principalement sur SGD avec momentum, qui utilise un learning rate global fixe (ou schedules manuels). Les méthodes adaptatives existantes incluent AdaGrad (accumulation des gradients au carré, mais learning rate décroît trop rapidement) et RMSProp (moyenne mobile exponentielle des gradients au carré, mais pas de correction de biais). Les défis incluent : sensibilité au choix du learning rate, nécessité de tuning manuel des hyperparamètres, inefficacité sur données sparse, et variance élevée des gradients stochastiques. L'objectif d'Adam est de créer un optimiseur robuste, efficient, et simple à utiliser, combinant adaptation par paramètre et momentum.

## Idées clés

1. **Moyennes mobiles exponentielles des moments des gradients** : Maintien de m_t (premier moment, momentum) et v_t (second moment non centré, variance) avec decay rates β₁ et β₂. Les moyennes exponentielles lissent les gradients bruités et permettent l'adaptation par paramètre du learning rate, combinant les bénéfices de momentum (accumulation de vélocité directionnelle) et RMSProp (scaling adaptatif).

2. **Correction de biais pour les estimations initiales** : Les moyennes mobiles m_t et v_t sont initialisées à zéro, créant un biais vers zéro aux premiers pas (surtout si β₁, β₂ proches de 1). Les estimateurs corrigés m̂_t = m_t/(1-β₁ᵗ) et v̂_t = v_t/(1-β₂ᵗ) compensent ce biais, permettant des mises à jour stables dès le début du training sans warmup manuel.

3. **Invariance à la mise à l'échelle des gradients** : Le step size effectif |Δθ_t| est borné par α (learning rate) indépendamment de l'amplitude des gradients, grâce à la normalisation par √v̂_t. Cette propriété garantit la stabilité même avec gradients explosifs ou vanishing, contrairement à SGD où |Δθ| ∝ |∇f|.

4. **Convergence théorique en optimisation convexe online** : Preuve de regret O(√T) en setting online convex optimization, égalant les garanties théoriques d'AdaGrad/RMSProp. En pratique, Adam converge rapidement sur fonctions non-convexes (deep learning) sans garanties formelles, mais performances empiriques solides.

5. **Hyperparamètres robustes par défaut** : β₁=0.9, β₂=0.999, ε=10⁻⁸, α=0.001 fonctionnent bien sur une large variété de problèmes sans tuning. Seul α nécessite occasionnellement un ajustement (learning rate schedules), réduisant significativement le coût de recherche d'hyperparamètres vs SGD.

6. **Efficient en mémoire et computationnellement** : Overhead minimal vs SGD : stockage de deux vecteurs (m, v) de taille identique aux paramètres, et operations element-wise simples (additions, multiplications). Pas de calcul de Hessian (second-order methods) ou de line search, rendant Adam scalable aux modèles avec millions/milliards de paramètres.

## Méthode

**Algorithme Adam** (Algorithm 1) :

**Inputs** :
- α : learning rate (stepsize, default 0.001)
- β₁ : decay rate premier moment (default 0.9)
- β₂ : decay rate second moment (default 0.999)
- ε : terme de stabilité numérique (default 10⁻⁸)
- f(θ) : fonction objectif stochastique avec paramètres θ
- θ₀ : vecteur de paramètres initial

**Initialisation** :
- m₀ ← 0 (premier moment vector)
- v₀ ← 0 (second moment vector)
- t ← 0 (timestep)

**Boucle d'optimisation** (jusqu'à convergence de θ_t) :

```
t ← t + 1

g_t ← ∇_θ f_t(θ_{t-1})  # Gradient stochastique à timestep t

m_t ← β₁ · m_{t-1} + (1 - β₁) · g_t  # Mise à jour moyenne mobile biaisée 1er moment
v_t ← β₂ · v_{t-1} + (1 - β₂) · g_t²  # Mise à jour moyenne mobile biaisée 2nd moment

m̂_t ← m_t / (1 - β₁ᵗ)  # Correction biais 1er moment
v̂_t ← v_t / (1 - β₂ᵗ)  # Correction biais 2nd moment

θ_t ← θ_{t-1} - α · m̂_t / (√v̂_t + ε)  # Mise à jour des paramètres
```

**Return** θ_t (paramètres optimisés)

**Détails techniques** :

*Step size effectif* :
- Magnitude bornée : |Δθ_t| ≤ α · (1-β₁) / √(1-β₂) ≈ α (pour β₁=0.9, β₂=0.999)
- Adaptation par paramètre : chaque composante θ_i a son propre effective learning rate α/√v̂_{t,i}
- Large v̂_t (high variance gradients) → petit step size
- Small v̂_t (low variance gradients) → grand step size

*Correction de biais* :
- Sans correction : m̂_t ≈ 0 et v̂_t ≈ 0 aux premiers pas (t << 1/(1-β))
- Avec correction : m̂_t et v̂_t sont des estimateurs non-biaisés dès t=1
- Impact diminue exponentiellement avec t : (1-β₁ᵗ) → 1 quand t → ∞

*Variantes* :

**AdaMax** (Section 7.1) : Généralisation utilisant L∞ norm au lieu de L² pour second moment
```
u_t ← max(β₂ · u_{t-1}, |g_t|)  # L∞ norm des gradients
θ_t ← θ_{t-1} - (α/(1-β₁ᵗ)) · m_t / u_t  # Pas de √ ni ε
```
Plus stable que Adam sur certains problèmes avec outliers dans les gradients.

**Adam avec décroissance du learning rate** : Stratégies de scheduling
- Step decay : α_t = α₀ · γ^(t/T)
- Exponential decay : α_t = α₀ · e^(-λt)
- Cosine annealing : α_t = α₀ · (1 + cos(πt/T))/2

## Résultats

**Logistic Regression (MNIST, 60k training images)** (Figure 1) :

*Comparison optimizers (batch size 128)* :
- **Adam** : 98.2% test accuracy @ 10 epochs (convergence la plus rapide)
- AdaGrad : 98.0% @ 10 epochs (learning rate décroît trop vite aux later epochs)
- RMSProp : 97.8% @ 10 epochs (variance élevée, oscille)
- SGD+Nesterov : 97.5% @ 10 epochs (convergence lente malgré momentum)

*Robustesse au learning rate* (Figure 2) :
- Adam avec α ∈ [10⁻⁴, 10⁻²] : variance faible, performances stables
- SGD : très sensible au choix de α, nécessite fine-tuning

**Multi-layer Perceptron (MNIST, 2 hidden layers 1000 units)** (Figure 3) :

*Test accuracy après 10 epochs* :
- **Adam** : 98.5% (β₁=0.9, β₂=0.999, α=0.001)
- RMSProp : 98.2%
- AdaGrad : 97.9%
- SGD+momentum : 97.7%

*Training loss convergence* :
- Adam atteint 0.05 cross-entropy loss en 5 epochs
- SGD+momentum nécessite 15 epochs pour loss similaire

**Convolutional Neural Network (CIFAR-10, 9-layer CNN)** (Figure 4) :

*Architecture* : 3× [Conv 5×5 (128 filters) → ReLU → MaxPool 3×3] → FC 384 → FC 192 → Softmax
*Batch size* : 128
*Epochs* : 45

*Test accuracy* :
- **Adam** : **83.2%** (meilleure accuracy finale)
- AdaGrad : 82.1%
- RMSProp : 81.8%
- SGD+Nesterov (α=0.01, momentum 0.9) : 82.5%

*Vitesse de convergence* :
- Adam atteint 80% accuracy en 15 epochs
- SGD+Nesterov nécessite 30 epochs pour accuracy similaire

**Variational Auto-Encoder (VAE, MNIST)** (Section 6.3) :

*Architecture* : Encoder (784 → 500 → 500 → 2×20 latent), Decoder (20 → 500 → 500 → 784)
*Objective* : Negative ELBO (Evidence Lower Bound)

*Convergence ELBO* :
- **Adam** : -86.2 nats @ 100 epochs
- AdaGrad : -87.5 nats (légèrement pire)
- RMSProp : -88.1 nats
- SGD+momentum : -90.3 nats (convergence beaucoup plus lente)

**Sparse data (Logistic regression, IMDB sentiment analysis)** :

*Données* : Bag-of-words sparse (vocabulary 10k mots, 88% sparsity)
*Batch size* : 128

*Test accuracy* :
- **Adam** : 89.2% (gère bien la sparsity)
- AdaGrad : 89.0% (bon sur sparse data historiquement)
- RMSProp : 88.5%
- SGD : 87.1% (performances dégradées sur sparse features)

**Ablation études** (Table 1) :

*Impact correction de biais (MNIST MLP)* :
- Adam avec correction : 98.5% @ 10 epochs
- Adam sans correction : 97.2% @ 10 epochs (-1.3%, convergence très lente début training)

*Impact β₁ (momentum)* :
- β₁=0.9 : 98.5% (optimal)
- β₁=0.0 (no momentum) : 97.8% (-0.7%)
- β₁=0.99 : 98.3% (momentum trop élevé, oscille)

*Impact β₂ (second moment decay)* :
- β₂=0.999 : 98.5% (optimal)
- β₂=0.9 : 97.5% (adapte trop rapidement, variance élevée)
- β₂=0.9999 : 98.4% (adapte trop lentement)

## Limites

1. **Convergence théorique limitée au cas convexe** : Les garanties de regret O(√T) ne s'appliquent qu'aux fonctions convexes (online convex optimization). En deep learning (non-convex), Adam n'a pas de garanties de convergence vers minima globaux ou même locaux. Des contre-exemples montrent qu'Adam peut ne pas converger sur certains problèmes non-convexes pathologiques.

2. **Généralisation parfois inférieure à SGD** : Sur certains benchmarks (ImageNet, language modeling), SGD avec momentum + learning rate schedule manuel généralise mieux qu'Adam (test accuracy supérieure malgré training loss similaire). Adam peut overfitter plus rapidement, nécessitant plus de régularisation (weight decay, dropout) ou des variants comme AdamW.

3. **Sensibilité au choix de β₂ sur certains problèmes** : Bien que β₂=0.999 fonctionne bien généralement, certaines tâches (reinforcement learning, GANs) nécessitent β₂ plus faible (0.99, 0.9) pour éviter l'instabilité. Le second moment peut s'accumuler de façon pathologique sur distributions non-stationnaires, réduisant excessivement le learning rate.

4. **Bias correction overhead aux premiers pas** : Le terme de correction (1-β₁ᵗ) cause des step sizes très larges aux tout premiers pas (t=1,2), pouvant déstabiliser le training si le learning rate initial α est mal choisi. Certaines implémentations nécessitent warmup malgré la correction de biais.

5. **Interaction problématique avec weight decay** : L'implémentation standard de weight decay dans Adam (L2 regularization ajoutée au gradient) n'est pas équivalente à la vraie décroissance de poids. AdamW (Loshchilov & Hutter 2019) corrige ce problème en découplant weight decay de l'optimisation adaptative, améliorant les performances.

6. **Performance variable sur données très sparse** : Bien qu'Adam soit meilleur que SGD sur sparse data, il est parfois surpassé par des optimiseurs spécialisés (LazyAdam, sparse Adam) qui appliquent les mises à jour uniquement aux features actives, réduisant compute et mémoire.

7. **Pas d'analyse de la variance des estimateurs** : L'article ne fournit pas d'analyse théorique de la variance des estimateurs m̂_t et v̂_t. En pratique, v̂_t peut avoir une variance élevée aux premiers pas (petit historique), causant des step sizes erratiques malgré la correction de biais.

8. **Comparaison limitée avec second-order methods** : Pas de comparaison avec LBFGS, Hessian-free, ou K-FAC (approximations second-order). Ces méthodes peuvent converger plus rapidement en nombre d'itérations (moins de pas nécessaires) mais sont computationnellement coûteuses et difficilement scalables.

9. **Absence d'étude sur l'impact de la taille de batch** : Les expériences utilisent batch sizes fixes (128-256). Pas d'analyse sur comment Adam scale avec large batch training (batch 1k-32k), important pour distributed training. Large batches peuvent nécessiter des ajustements de α et β₂.

10. **Pas de guidelines pour learning rate scheduling** : Bien que Adam soit moins sensible au learning rate que SGD, des schedules (step decay, cosine annealing) améliorent souvent les performances. L'article ne fournit pas de recommendations sur quand/comment ajuster α durant le training, laissant cela au tuning empirique.

## Liens utiles

- **PDF annoté**: [Adam annotated](../../papers/2025/adam.pdf)
- **Article**: [Adam: A Method for Stochastic Optimization (PDF)](https://arxiv.org/pdf/1412.6980)
- **ArXiv**: [1412.6980](https://arxiv.org/abs/1412.6980)

## Notes perso

La difference entre Adam et AdamW est juste dans l'endroit ou on applique le weight decay. Dans Adam, le weight decay est applique a la fonction de cout, probleme: la normalisation est divise par le second moment ce qui implique que tous les poids ne sont pas modifie par le weight decay de la meme facon. Dans AdamW, le weight decay est applique dans la formule du learning rate adaptatif, comme ca tous les poids du models sont modifie equitablement.

Nb: weight decay = (pour faire simple) norm L2
