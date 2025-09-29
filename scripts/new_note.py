#!/usr/bin/env python3
import sys, datetime, pathlib, re
date = datetime.date.today().isoformat()
title = " ".join(sys.argv[1:]) if len(sys.argv)>1 else "Untitled"
slug = re.sub(r'[^a-z0-9]+','-', title.lower()).strip('-') or "untitled"
fname = pathlib.Path(f"docs/notes/{date[:4]}/{date}-tbd-{slug}.md")
fname.parent.mkdir(parents=True, exist_ok=True)
tpl = f"""---\ntitle: "{title}"\nauthors: []\nvenue: ""\nyear: {date[:4]}\ncitekey: \nlinks:\n  pdf: ""\n  code: ""\n  project: ""\ntags:\n  areas: []\n  methods: []\n  tasks: []\n  status: "to-read"\nreplication:\n  repo: ""\n  data: ""\n  env: ""\n---\n\n## TL;DR\n\n## Contexte\n\n## Idées clés\n\n## Méthode\n\n## Résultats\n\n## Limites\n\n## Liens utiles\n\n## Notes perso\n"""
fname.write_text(tpl, encoding="utf-8")
print(fname)
