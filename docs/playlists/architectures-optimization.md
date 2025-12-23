# ⚙️ 🔧 Architectures & Optimization

Techniques d'optimisation et composants architecturaux qui ont transformé l'entraînement des réseaux de neurones

!!! tip "Parcours de lecture"
    Cette playlist contient **9 papiers** sélectionnés pour leur impact sur le domaine.

---

## 1. Dropout (2014)

**Auteurs**: Nitish Srivastava, Geoffrey Hinton, Alex Krizhevsky, Ilya Sutskever, Ruslan Salakhutdinov

**READ**: ✅

!!! quote "Pourquoi ce papier ?"
    A Simple Way to Prevent Neural Networks from Overfitting - désactivation aléatoire de neurones empêchant la co-adaptation.

**Liens**:

- 📝 [Mes notes](../notes/2025/2025-10-25-tbd-dropout-a-simple-way-to-prevent-neural-networks-from-overfitting.md)
- 🔗 [Article](http://jmlr.org/papers/v15/srivastava14a.html)

---

## 2. Adam Optimizer (2014)

**Auteurs**: Diederik P. Kingma, Jimmy Ba

**READ**: ✅

!!! quote "Pourquoi ce papier ?"
    A Method for Stochastic Optimization - optimiseur adaptatif combinant momentum et RMSProp, devenu l'optimiseur par défaut.

**Liens**:

- 📝 [Mes notes](../notes/2025/2025-12-23-tbd-adam-a-method-for-stochastic-optimization.md)
- 🔗 [Article](https://arxiv.org/abs/1412.6980)

---

## 3. Batch Normalization (2015)

**Auteurs**: Sergey Ioffe, Christian Szegedy

**READ**: ✅

!!! quote "Pourquoi ce papier ?"
    Normalisation par mini-batch accélérant l'entraînement et permettant des learning rates plus élevés.

**Liens**:

- 📝 [Mes notes](../notes/2025/2025-11-02-tbd-batch-normalization-accelerating-deep-network-training-by-reducing-internal-covariate-shift.md)
- 🔗 [Article](https://arxiv.org/abs/1502.03167)

---

## 4. Weight Normalization (2016)

**Auteurs**: Tim Salimans, Diederik P. Kingma

**READ**: ⬜

!!! quote "Pourquoi ce papier ?"
    Reparamétrisation des poids en magnitude et direction pour accélérer la convergence sans dépendance au batch.

**Liens**:

- 🔗 [Article](https://arxiv.org/abs/1602.07868)

---

## 5. Layer Normalization (2016)

**Auteurs**: Jimmy Lei Ba, Jamie Ryan Kiros, Geoffrey E. Hinton

**READ**: ⬜

!!! quote "Pourquoi ce papier ?"
    Normalisation par couche indépendante du batch, cruciale pour les RNNs et Transformers.

**Liens**:

- 🔗 [Article](https://arxiv.org/abs/1607.06450)

---

## 6. Mixed Precision Training (2017)

**Auteurs**: Paulius Micikevicius, Sharan Narang, Jonah Alben, Gregory Diamos, Erich Elsen, David Garcia, Boris Ginsburg, Michael Houston, Oleksii Kuchaiev, Ganesh Venkatesh, et al.

**READ**: ⬜

!!! quote "Pourquoi ce papier ?"
    Entraînement en précision mixte (FP16/FP32) pour accélération 2-3x avec efficacité mémoire sans perte de précision.

**Liens**:

- 🔗 [Article](https://arxiv.org/abs/1710.03740)

---

## 7. Group Normalization (2018)

**Auteurs**: Yuxin Wu, Kaiming He

**READ**: ⬜

!!! quote "Pourquoi ce papier ?"
    Normalisation par groupes de canaux, alternative à Batch Norm stable avec petits batchs.

**Liens**:

- 🔗 [Article](https://arxiv.org/abs/1803.08494)

---

## 8. Spectral Normalization for GANs (2018)

**Auteurs**: Takeru Miyato, Toshiki Kataoka, Masanori Koyama, Yuichi Yoshida

**READ**: ⬜

!!! quote "Pourquoi ce papier ?"
    Normalisation spectrale stabilisant l'entraînement des GANs en contraignant les constantes de Lipschitz.

**Liens**:

- 🔗 [Article](https://arxiv.org/abs/1802.05957)

---

## 9. LoRA (2021)

**Auteurs**: Edward J. Hu, Yelong Shen, Phillip Wallis, Zeyuan Allen-Zhu, Yuanzhi Li, Shean Wang, Lu Wang, Weizhu Chen

**READ**: ⬜

!!! quote "Pourquoi ce papier ?"
    Low-Rank Adaptation - fine-tuning efficace des LLMs en apprenant des matrices low-rank, réduisant drastiquement les paramètres entraînables.

**Liens**:

- 🔗 [Article](https://arxiv.org/abs/2106.09685)

---

