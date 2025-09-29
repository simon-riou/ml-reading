# ML/DL Reading Log
- Une note par papier dans `notes/YYYY/`.
- Références uniques dans `CITATION.bib` (clé = citekey des notes).
- Tags contrôlés via `tags.yaml`.
- PDFs éventuels dans `papers/YYYY/` ou liens externes.
- Scripts utilitaires dans `scripts/`.

# Fast search
- By tag : grep -R --line-number -E '^tags:' -n notes -A5 | grep 'transformers'
- By citekey : grep -R "smith2025transformer" notes
- By status : grep -R "status: \"deep-read\"" notes
