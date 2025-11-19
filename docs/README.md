# 📚 ML/DL Reading Log

Suivi des papiers de recherche que j'ai lu

## 🌐 Site Web

Addresse : **[simon-riou.github.io/ml-reading](https://simon-riou.github.io/ml-reading)**

## 📖 Organisation

### 📝 Notes
- **Une note par article** dans `docs/notes/YYYY/`
- Notes perso sur l'article
- Filtre par tags [tags dispo](tags/index.md)

### 🎵 Playlists
- **Collections thématiques** de papiers organisés chronologiquement
- Racontent l'histoire d'un domaine du deep learning
- Exemples : Computer Vision, Régularisation, Training Recipes...
- Voir toutes les [playlists disponibles](playlists/index.md)

### 🏷️ Système de Tags
Les tags sont contrôlés via `docs/tags.yaml` :
- **Areas** : domaines de recherche (generalization, robustness, nlp, vision...)
- **Methods** : méthodes utilisées (transformers, diffusion, gnn...)
- **Tasks** : tâches abordées (classification, summarization, retrieval...)
- **Status** : état de lecture (to-read, skimmed, deep-read, replicated)

### 📚 Références
- Toutes les references sont ici : `CITATION.bib`

### 📄 PDFs
- Si PDF disponible ils sont ici : `papers/YYYY/`

## 🛠️ Outils

### Scripts Utilitaires
- `scripts/new_note.py` : Création rapide d'une nouvelle note
- `scripts/sync_bib.py` : Synchronisation des références bibliographiques
- `docs_build/gen_playlists.py` : Génération des pages de playlists

## 🚀 Utilisation

### Créer une nouvelle note
```bash
python scripts/new_note.py "Nom de l'article"
```

### Synchroniser la bibliographie
```bash
python scripts/sync_bib.py
```

### Créer ou mettre à jour une playlist

1. Éditer `docs/playlists.yaml`
2. Ajouter ou modifier une playlist
3. Générer les pages :
```bash
python docs_build/gen_playlists.py
```

Les playlists seront automatiquement générées lors du build MkDocs.

### Développement local
```bash
pip install -r requirements.txt

# Lancer serveur local :
mkdocs serve
```
