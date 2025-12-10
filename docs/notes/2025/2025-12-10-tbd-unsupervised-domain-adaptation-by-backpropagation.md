---
title: "Unsupervised Domain Adaptation by Backpropagation"
authors: ["Yaroslav Ganin", "Victor Lempitsky"]
venue: "arXiv"
year: 2015
citekey: ganin2015unsuperviseddomainadaptationbackpropagation
links:
  pdf: "https://arxiv.org/abs/1409.7495"
  code: ""
  project: ""
pdf_annotated: "../../papers/gradient_reversal_layer.pdf"
tags:
  areas: [domain-adaptation, transfer-learning, deep-learning]
  methods: [gradient-reversal-layer, adversarial-training, domain-invariant-features]
  tasks: [classification, domain-adaptation]
  status: "deep-read"
replication:
  repo: ""
  data: ""
  env: ""
---

## TL;DR

Introduction d'une couche de **Gradient Reversal Layer (GRL)** permettant l'adaptation de domaine non supervisée dans les réseaux profonds. La méthode entraîne simultanément un extracteur de features discriminatives et invariantes au domaine via un mécanisme adversarial, implémenté simplement par rétropropagation standard. Approche générique applicable à toute architecture feed-forward, avec des résultats state-of-the-art sur OFFICE datasets et amélioration substantielle sur des shifts de domaine importants (MNIST→MNIST-M: 57.9% du gap comblé, SYN NUMBERS→SVHN: 66.1%).

## Contexte

En 2015, les architectures profondes nécessitent d'énormes quantités de données labellisées. L'adaptation de domaine (DA) offre une solution attractive quand on dispose de données labellisées dans un domaine source (ex: images synthétiques) et de données non labellisées dans le domaine cible (ex: images réelles). Les méthodes DA précédentes travaillaient avec des représentations fixes ("shallow DA"), alors que l'objectif est d'intégrer l'adaptation directement dans le processus d'apprentissage des features ("deep DA"). Le défi: apprendre des features à la fois (i) discriminatives pour la tâche principale et (ii) invariantes au shift entre domaines.

## Idées clés

1. **Apprentissage de features domain-invariant par optimisation adversariale** : Optimisation conjointe de trois composants : (i) extracteur de features, (ii) prédicteur de labels (classifieur principal), (iii) classifieur de domaine. Les paramètres de l'extracteur sont optimisés pour **minimiser** la perte du prédicteur de labels ET **maximiser** la perte du classifieur de domaine, encourageant l'émergence de features indistinguables entre domaines.

2. **Gradient Reversal Layer (GRL)** : Couche sans paramètres qui agit comme l'identité en forward propagation mais **multiplie le gradient par -λ** en backpropagation. Permet d'implémenter l'optimisation adversariale via SGD standard dans n'importe quel framework deep learning, sans modification de l'algorithme d'entraînement.

3. **Recherche de point-selle (saddle point)** : La méthode cherche un point-selle de la fonctionnelle E(θf, θy, θd) où θf maximise la perte du classifieur de domaine (domain confusion) tout en minimisant la perte de prédiction, tandis que θd minimise sa propre perte (discrimination optimale des domaines).

4. **Lien avec H∆H-distance** : Le classifieur de domaine optimal donne une borne supérieure sur dH∆H(S,T), la distance de discrepancy entre distributions. La rétropropagation du gradient inversé réduit cette distance, améliorant la borne théorique sur l'erreur du domaine cible: εT(h) ≤ εS(h) + ½dH∆H(S,T) + C.

5. **Simplicité d'implémentation** : Approche générique applicable à n'importe quelle architecture feed-forward entraînable par backpropagation. Nécessite uniquement d'ajouter une branche de classification de domaine avec la GRL, sans modification du processus d'entraînement standard.

6. **Adaptation progressive via scheduling de λ** : Le facteur d'adaptation λ croît progressivement de 0 à 1 selon λp = 2/(1+exp(-γ·p)) - 1, supprimant le signal bruité du classifieur de domaine en début d'entraînement quand les features ne sont pas encore bien formées.

## Méthode

**Architecture (Figure 1)** :
- **Feature extractor Gf(x; θf)** : x → f ∈ R^D (CNN avec 2-3 couches convolutionnelles selon le dataset)
- **Label predictor Gy(f; θy)** : f → y ∈ Y (réseau feed-forward, perte Ly = logistic regression)
- **Domain classifier Gd(Rλ(f); θd)** : f → d ∈ {0,1} via GRL (3 FC layers : x→1024→1024→2, perte Ld = binomial cross-entropy)

**Gradient Reversal Layer** (Equations 7-8) :
- Forward: Rλ(x) = x (identité)
- Backward: dRλ/dx = -λI (multiplication par -λ)

**Optimisation par point-selle** (Equations 1-3) :
Fonctionnelle à optimiser:
```
E(θf, θy, θd) = Σ(i: di=0) Ly(Gy(Gf(xi; θf); θy), yi)
                - λ Σ(i=1..N) Ld(Gd(Gf(xi; θf); θd), di)
```

Conditions du point-selle:
- (θ̂f, θ̂y) = argmin E(θf, θy, θ̂d) : minimise perte labels, maximise perte domaine
- θ̂d = argmax E(θ̂f, θ̂y, θd) : discrimine optimalement les domaines

**Mises à jour SGD** (Equations 4-6) :
```
θf ← θf - µ(∂Li_y/∂θf - λ∂Li_d/∂θf)  [via GRL]
θy ← θy - µ∂Li_y/∂θy
θd ← θd - µ∂Li_d/∂θd
```

**Protocole d'entraînement** :
- Batches de 128 : 50% domaine source (avec labels), 50% domaine cible (sans labels)
- SGD avec momentum 0.9, learning rate annealing : µp = µ0/(1+α·p)^β avec µ0=0.01, α=10, β=0.75
- Scheduling λ : λp = 2/(1+exp(-γ·p)) - 1 avec γ=10
- Prétraitement : mean subtraction sur images
- Dropout et ℓ2-norm restriction pour SVHN

**Architectures testées** (Appendix B, Figure 5) :
- MNIST : 2 conv (5×5, 32/48 maps) + 2 FC (100 units) → inspiré LeNet-5
- SVHN : 3 conv (5×5/5×5/5×5, 64/64/128 maps) + 2 FC (3072/2048 units)
- GTSRB : 3 conv (5×5/3×3/5×5, 96/144/256 maps) + 1 FC (512 units)
- OFFICE : AlexNet pré-entraîné ImageNet, adaptation sur fc7 (256-dim)

## Résultats

**Digit Classification** (Table 1) :

*MNIST → MNIST-M* (digits avec background coloré complexe) :
- Source only : 57.49% | **Proposed : 81.49% (+57.9% du gap)** | Train on target : 98.91%
- SA baseline : 60.78% (+7.9%) → gains modestes, adaptation difficile
- Visualizations t-SNE : forte séparation sans adaptation, distributions alignées avec GRL

*SYN NUMBERS → SVHN* (synthétique → Street View) :
- Source only : 86.65% | **Proposed : 90.48% (+66.1% du gap)** | Train on target : 92.44%
- SA baseline : 86.72% (+1.3%) → pas d'amélioration significative, shift très important
- Couverture de 2/3 du gap malgré structured clutter dans SVHN backgrounds

*SVHN → MNIST* :
- Source only : 59.19% | **Proposed : 71.07% (+29.3% du gap)** | Train on target : 99.51%
- Adaptation réussie grâce à la diversité de SVHN (modèle plus générique)

*MNIST → SVHN* :
- Échec de l'adaptation non supervisée (domaines trop différents, MNIST pas assez diverse)
- SA échoue également → limitation des méthodes DA unsupervised pour shifts extrêmes

*SYN SIGNS → GTSRB* (43 classes traffic signs) :
- Source only : 74.00% | **Proposed : 88.66% (+56.7% du gap)** | Train on target : 99.87%
- SA : 76.35% (+9.1%) → amélioration modeste
- Complexité accrue (43 classes vs 10) mais adaptation réussie

*Semi-supervised GTSRB* (Figure 4, 1280 labeled target samples) :
- Validation error significativement plus bas que source-only ou target-only
- Démontre l'extensibilité de la méthode au cas semi-supervisé

**OFFICE Dataset** (Table 2, state-of-the-art comparison) :

*Amazon → Webcam* :
- **Proposed : 67.3%±1.7%** vs DDC : 59.4%±0.8% → **+7.9% absolus**
- Meilleure performance sur le shift le plus difficile (plus grande différence entre domaines)
- Gains substantiels vs compétiteurs : SA (45.0%), DANN (53.6%), DLID (51.9%)

*DSLR → Webcam* :
- **Proposed : 94.0%±0.8%** vs DDC : 92.5%±0.3% → **+1.5% absolus**
- Dépasse DLID (78.2%) et atteint quasi-saturation sur cette tâche plus facile

*Webcam → DSLR* :
- **Proposed : 93.7%±1.0%** vs DDC : 91.7%±0.8% → **+2.0% absolus**
- Performances similaires à DSLR→Webcam, confirme la robustesse

**Détails protocole OFFICE** :
- Fine-tuning AlexNet pré-entraîné ImageNet (fc7 features 256-dim)
- 5 random splits, même nombre de samples source/catégorie que baselines
- Setting tranductif : utilise tout le domaine cible non-labelisé (vs inductive dans certaines baselines)
- Architecture domain classifier identique à DDC (Tzeng et al. 2014) pour comparaison équitable

**Visualisations t-SNE** (Figure 3) :
- Séparation claire domaines source (bleu) / target (rouge) sans adaptation
- Forte superposition des distributions après adaptation avec GRL
- Corrélation entre succès de l'adaptation et overlap visuel des features

## Limites

1. **Échecs sur shifts extrêmes** : MNIST→SVHN échoue car MNIST n'est pas assez diverse pour généraliser à SVHN. La méthode unsupervised ne peut pas combler des gaps de complexité trop importants entre domaines (nécessiterait données labeled target).

2. **Choix d'hyperparamètres** : Le scheduling de λ (γ=10) et le learning rate annealing (α=10, β=0.75) sont fixés empiriquement. Les auteurs suggèrent d'utiliser l'erreur sur le domaine source + erreur du classifieur de domaine comme proxy pour validation unsupervised, mais pas de protocole systématique.

3. **Architecture du classifieur de domaine** : Configuration "somewhat arbitrary" (3 FC layers 1024→1024→2). Un tuning plus fin pourrait améliorer les performances, notamment l'attachement à différentes couches du feature extractor.

4. **Comparaison OFFICE** : Setting tranductif (utilise tout le target domain) vs certaines baselines inductives. Bien que justifié par la premise "abundance of unlabeled data", rend la comparaison moins directe avec méthodes comme GFK ou SA.

5. **Analyse théorique limitée** : Le lien avec H∆H-distance (Section 3.3) nécessite l'hypothèse forte que Hd contient Hp∆Hp (symmetric difference hypothesis set). Validation empirique de cette hypothèse non fournie.

6. **Dépendance à la qualité du feature extractor initial** : Performances probablement sensibles à l'initialisation. Les auteurs suggèrent d'utiliser autoencoders pré-entraînés (Glorot et al. 2011) mais n'explorent pas cette direction expérimentalement.

7. **Covariate shift assumption** : La borne théorique εT(h) ≤ εS(h) + ½dH∆H(S,T) + C suppose que P(y|x) est identique entre domaines (seul P(x) diffère). Cette hypothèse peut être violée dans certains scénarios pratiques.

8. **Évaluation semi-supervisée incomplète** : Figure 4 montre des résultats prometteurs pour SYN SIGNS→GTSRB semi-supervisé, mais pas d'évaluation systématique sur autres datasets ou comparaison avec méthodes semi-supervisées dédiées.

## Liens utiles

- **PDF annoté**: [Gradient Reversal Layer annotated](../../papers/gradient_reversal_layer.pdf)
- **Article**: [Unsupervised Domain Adaptation by Backpropagation (PDF)](https://arxiv.org/pdf/1409.7495)
- **ArXiv**: [1409.7495](https://arxiv.org/abs/1409.7495)

## Notes perso
