#!/usr/bin/env python3
import sys, bibtexparser
with open("CITATION.bib") as f:
    db = bibtexparser.load(f)
ids = [e.get("ID","") for e in db.entries]
dups = sorted({i for i in ids if ids.count(i) > 1})
if dups:
    print("DUPLICATE_KEYS:", *dups); sys.exit(1)
print("OK:", len(ids), "entries")
