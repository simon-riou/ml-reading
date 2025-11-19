# ✨ 🎨 Modèles Génératifs - Images

GANs, VAEs, Diffusion Models - l'art de créer du contenu visuel

!!! tip "Parcours de lecture"
    Cette playlist contient **19 papiers** sélectionnés pour leur impact sur le domaine.

---

## 1. Variational Autoencoders (VAE) (2013)

**Auteurs**: Diederik P. Kingma, Max Welling

**READ**: ⬜

!!! quote "Pourquoi ce papier ?"
    Auto-Encoding Variational Bayes - framework probabiliste pour l'apprentissage de représentations latentes et la génération.

**Liens**:

- 🔗 [Article](https://arxiv.org/abs/1312.6114)

---

## 2. GANs (2014)

**Auteurs**: Ian J. Goodfellow, Jean Pouget-Abadie, Mehdi Mirza, Bing Xu, David Warde-Farley, Sherjil Ozair, Aaron Courville, Yoshua Bengio

**READ**: ⬜

!!! quote "Pourquoi ce papier ?"
    Generative Adversarial Networks - jeu minimax entre générateur et discriminateur. Révolution dans la génération d'images.

**Liens**:

- 🔗 [Article](https://arxiv.org/abs/1406.2661)

---

## 3. Conditional GANs (2014)

**Auteurs**: Mehdi Mirza, Simon Osindero

**READ**: ⬜

!!! quote "Pourquoi ce papier ?"
    Extension des GANs permettant de conditionner la génération sur des labels ou d'autres informations.

**Liens**:

- 🔗 [Article](https://arxiv.org/abs/1411.1784)

---

## 4. DCGAN (2015)

**Auteurs**: Alec Radford, Luke Metz, Soumith Chintala

**READ**: ⬜

!!! quote "Pourquoi ce papier ?"
    Deep Convolutional GANs - architecture stable utilisant des convolutions, batch norm et des guidelines d'entraînement.

**Liens**:

- 🔗 [Article](https://arxiv.org/abs/1511.06434)

---

## 5. Pix2Pix (2016)

**Auteurs**: Phillip Isola, Jun-Yan Zhu, Tinghui Zhou, Alexei A. Efros

**READ**: ⬜

!!! quote "Pourquoi ce papier ?"
    Image-to-Image Translation avec GANs conditionnels - traduction supervisée entre domaines d'images appariées.

**Liens**:

- 🔗 [Article](https://arxiv.org/abs/1611.07004)

---

## 6. CycleGAN (2017)

**Auteurs**: Jun-Yan Zhu, Taesung Park, Phillip Isola, Alexei A. Efros

**READ**: ⬜

!!! quote "Pourquoi ce papier ?"
    Traduction d'images non-appariées utilisant la cohérence cyclique - pas besoin de données paired.

**Liens**:

- 🔗 [Article](https://arxiv.org/abs/1703.10593)

---

## 7. Progressive GAN (2017)

**Auteurs**: Tero Karras, Timo Aila, Samuli Laine, Jaakko Lehtinen

**READ**: ⬜

!!! quote "Pourquoi ce papier ?"
    Croissance progressive des GANs permettant de générer des images haute résolution avec stabilité améliorée.

**Liens**:

- 🔗 [Article](https://arxiv.org/abs/1710.10196)

---

## 8. StyleGAN (2018)

**Auteurs**: Tero Karras, Samuli Laine, Timo Aila

**READ**: ⬜

!!! quote "Pourquoi ce papier ?"
    Architecture basée sur le style permettant un contrôle granulaire des attributs générés via AdaIN.

**Liens**:

- 🔗 [Article](https://arxiv.org/abs/1812.04948)

---

## 9. BigGAN (2018)

**Auteurs**: Andrew Brock, Jeff Donahue, Karen Simonyan

**READ**: ⬜

!!! quote "Pourquoi ce papier ?"
    Large Scale GAN Training - scaling des GANs pour une synthèse d'images haute fidélité sur ImageNet.

**Liens**:

- 🔗 [Article](https://arxiv.org/abs/1809.11096)

---

## 10. StyleGAN2 (2019)

**Auteurs**: Tero Karras, Samuli Laine, Miika Aittala, Janne Hellsten, Jaakko Lehtinen, Timo Aila

**READ**: ⬜

!!! quote "Pourquoi ce papier ?"
    Analyse et amélioration de StyleGAN - suppression des artefacts et meilleure qualité d'image.

**Liens**:

- 🔗 [Article](https://arxiv.org/abs/1912.04958)

---

## 11. DDPM (2020)

**Auteurs**: Jonathan Ho, Ajay Jain, Pieter Abbeel

**READ**: ✅

!!! quote "Pourquoi ce papier ?"
    Denoising Diffusion Probabilistic Models - génération via débruitage progressif, surpassant les GANs en qualité.

**Liens**:

- 📝 [Mes notes](../notes/2025/2025-11-12-tbd-denoising-diffusion-probabilistic-models.md)
- 🔗 [Article](https://arxiv.org/abs/2006.11239)
- 💻 [Code](https://github.com/hojonathanho/diffusion)

---

## 12. DDIM (2020)

**Auteurs**: Jiaming Song, Chenlin Meng, Stefano Ermon

**READ**: ⬜

!!! quote "Pourquoi ce papier ?"
    Denoising Diffusion Implicit Models - échantillonnage accéléré et déterministe pour les modèles de diffusion.

**Liens**:

- 🔗 [Article](https://arxiv.org/abs/2010.02502)

---

## 13. DALL-E (2021)

**Auteurs**: Aditya Ramesh, Mikhail Pavlov, Gabriel Goh, Scott Gray, Chelsea Voss, Alec Radford, Mark Chen, Ilya Sutskever

**READ**: ⬜

!!! quote "Pourquoi ce papier ?"
    Zero-Shot Text-to-Image Generation - transformer autorégressif pour générer des images depuis du texte.

**Liens**:

- 🔗 [Article](https://arxiv.org/abs/2102.12092)

---

## 14. Improved DDPM (2021)

**Auteurs**: Alex Nichol, Prafulla Dhariwal

**READ**: ⬜

!!! quote "Pourquoi ce papier ?"
    Améliorations des DDPM - log-likelihood et qualité d'échantillonnage améliorées.

**Liens**:

- 🔗 [Article](https://arxiv.org/abs/2102.09672)

---

## 15. Classifier-Free Guidance (2021)

**Auteurs**: Jonathan Ho, Tim Salimans

**READ**: ⬜

!!! quote "Pourquoi ce papier ?"
    Guidance sans classifier pour les modèles de diffusion - meilleur compromis qualité/diversité.

**Liens**:

- 🔗 [Article](https://arxiv.org/abs/2207.12598)

---

## 16. Latent Diffusion / Stable Diffusion (2021)

**Auteurs**: Robin Rombach, Andreas Blattmann, Dominik Lorenz, Patrick Esser, Björn Ommer

**READ**: ⬜

!!! quote "Pourquoi ce papier ?"
    High-Resolution Image Synthesis with Latent Diffusion Models - diffusion dans l'espace latent pour efficacité computationnelle.

**Liens**:

- 🔗 [Article](https://arxiv.org/abs/2112.10752)

---

## 17. DALL-E 2 (2022)

**Auteurs**: Aditya Ramesh, Prafulla Dhariwal, Alex Nichol, Casey Chu, Mark Chen

**READ**: ⬜

!!! quote "Pourquoi ce papier ?"
    Génération hiérarchique text-to-image avec CLIP latents - qualité et résolution améliorées.

**Liens**:

- 🔗 [Article](https://arxiv.org/abs/2204.06125)

---

## 18. Imagen (2022)

**Auteurs**: Chitwan Saharia, William Chan, Saurabh Saxena, Lala Li, Jay Whang, Emily Denton, et al.

**READ**: ⬜

!!! quote "Pourquoi ce papier ?"
    Modèles de diffusion text-to-image photoréalistes avec compréhension linguistique profonde via T5.

**Liens**:

- 🔗 [Article](https://arxiv.org/abs/2205.11487)

---

## 19. Stable Diffusion v2 (2022)

**Auteurs**: Stability AI

**READ**: ⬜

!!! quote "Pourquoi ce papier ?"
    Version améliorée et open-source de Latent Diffusion - démocratisation de la génération text-to-image.

**Liens**:

- 🔗 [Blog](https://stability.ai/news/stable-diffusion-v2-release)

---

