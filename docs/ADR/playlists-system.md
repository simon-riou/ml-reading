# ADR: Système de Playlists

## Contexte

Le projet contient de nombreux papiers de recherche organisés par tags. Cependant, il manquait une façon de créer des parcours de lecture chronologiques qui racontent l'histoire d'un domaine spécifique du deep learning.

## Décision

Implémentation d'un système de playlists qui permet de :
- Créer des collections thématiques de papiers
- Organiser chronologiquement les papiers importants d'un domaine
- Expliquer pourquoi chaque papier est important dans son contexte
- Générer automatiquement des pages navigables

## Architecture

### Structure des fichiers

```
docs/
├── playlists.yaml           # Configuration des playlists
├── playlists/               # Pages générées
│   ├── index.md             # Index de toutes les playlists
│   ├── computer-vision-foundations.md
│   ├── regularization-techniques.md
│   └── ...
docs_build/
└── gen_playlists.py         # Script de génération
```

### Format YAML

```yaml
playlists:
  - id: playlist-identifier
    title: "Titre de la Playlist"
    description: "Description du domaine"
    icon: "🎯"
    papers:
      - citekey: "key-from-citation-bib"
        title: "Titre court"
        year: 2021
        why: "Pourquoi ce papier est important"
```

### Génération automatique

Le script `gen_playlists.py` :
1. Lit `playlists.yaml`
2. Récupère les métadonnées depuis `CITATION.bib`
3. Trouve les notes associées dans `docs/notes/`
4. Génère les pages markdown avec liens vers les notes

### Intégration MkDocs

Le plugin `gen-files` exécute automatiquement le script lors du build :
```yaml
plugins:
  - gen-files:
      scripts:
        - docs_build/gen_playlists.py
```

## Avantages

✅ **Chronologique** : Raconte l'évolution d'un domaine
✅ **Contextuel** : Explique pourquoi chaque papier est important
✅ **Automatique** : Génération automatique depuis YAML
✅ **Lié** : Connexion avec notes existantes et CITATION.bib
✅ **Extensible** : Facile d'ajouter de nouvelles playlists

## Utilisation

### Créer une nouvelle playlist

1. Éditer `docs/playlists.yaml`
2. Ajouter une nouvelle entrée dans `playlists:`
3. Lancer `python docs_build/gen_playlists.py` (ou `mkdocs build`)

### Ajouter un papier à une playlist

1. S'assurer que le papier existe dans `CITATION.bib`
2. Ajouter l'entrée dans la playlist :
```yaml
- citekey: "key-from-bib"
  title: "Nom court"
  year: 2021
  why: "Impact du papier"
```

### Organisation recommandée

- **Fondations** : Papiers qui ont créé un domaine (AlexNet, ResNet, ViT...)
- **Techniques** : Méthodes spécifiques (Dropout, Label Smoothing...)
- **Recettes** : Guides pratiques d'implémentation
- **Évolution** : Timeline d'un sous-domaine (GANs, Diffusion...)

## Exemples de playlists

### Computer Vision Foundations
Raconte l'évolution de la vision : CNNs → ResNets → Transformers

### Regularization Techniques
Les techniques qui ont amélioré la généralisation

### Training Recipes
Guides pratiques pour l'entraînement SOTA

## Maintenance

Le système est **auto-généré** :
- Les changements dans `playlists.yaml` sont automatiquement reflétés
- Les liens vers les notes sont maintenus automatiquement
- Les métadonnées viennent de `CITATION.bib`

## Futur

Améliorations possibles :
- [ ] Graphe de dépendances entre papiers
- [ ] Timeline visuelle interactive
- [ ] Filtrage par auteur/institution
- [ ] Statistiques de lecture par playlist
