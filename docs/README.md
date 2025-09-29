# 📚 ML/DL Reading Log

Suivi des papiers de recherche que j'ai lu

## 🌐 Site Web

Addresse : **[simon-riou.github.io/ml-reading](https://simon-riou.github.io/ml-reading)**

## 📖 Organisation

### 📝 Notes
- **Une note par article** dans `docs/notes/YYYY/`
- Notes perso sur l'article
- Filtre par tags [tags dispo](tags/index.md)

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

## 🚀 Utilisation

### Créer une nouvelle note
```bash
python scripts/new_note.py "Nom de l'article"
```

### Synchroniser la bibliographie
```bash
python scripts/sync_bib.py
```

### Développement local
```bash
pip install -r requirements.txt

# Lancer serveur local :
mkdocs serve
```
