import os, re, yaml, frontmatter, mkdocs_gen_files
from pathlib import Path

ROOT = Path(".")
NOTES_DIR = ROOT / "notes"
TAGS_FILE = ROOT / "tags.yaml"
OUT_DIR = Path("tags")

tags_cfg = yaml.safe_load(TAGS_FILE.read_text(encoding="utf-8"))
all_tags = {k: set(v or []) for k, v in tags_cfg.items()}

notes = []
for p in NOTES_DIR.rglob("*.md"):
    post = frontmatter.load(p)
    meta = post.metadata or {}
    tags = (meta.get("tags") or {})
    notes.append({
        "path": p.as_posix(),
        "title": meta.get("title", p.stem),
        "year": str(meta.get("year", "")),
        "status": (tags.get("status") or ""),
        "areas": tags.get("areas") or [],
        "methods": tags.get("methods") or [],
        "tasks": tags.get("tasks") or [],
    })

index = {("status", t): [] for t in all_tags.get("status", [])}
for group in ("areas","methods","tasks"):
    for t in all_tags.get(group, []):
        index[(group, t)] = []

for n in notes:
    for group in ("areas","methods","tasks"):
        for t in n[group]:
            index.setdefault((group, t), []).append(n)
    s = n["status"]
    if s:
        index.setdefault(("status", s), []).append(n)

def mk_link(n):
    return f'[{n["title"]}](/{n["path"]})'

with mkdocs_gen_files.open(OUT_DIR / "index.md", "w") as f:
    f.write("# Tags\n\n")
    for group in ("areas","methods","tasks","status"):
        f.write(f"## {group}\n\n")
        for t in sorted(all_tags.get(group, [])):
            slug = re.sub(r'[^a-z0-9]+','-', t.lower()).strip('-')
            f.write(f"- [{t}](./{group}-{slug}.md) ({len(index.get((group,t),[]))})\n")
        f.write("\n")

for (group, tag), items in sorted(index.items()):
    if not items:
        continue
    slug = re.sub(r'[^a-z0-9]+','-', tag.lower()).strip('-')
    path = OUT_DIR / f"{group}-{slug}.md"
    items_sorted = sorted(items, key=lambda x: (x["year"], x["title"]), reverse=True)
    with mkdocs_gen_files.open(path, "w") as f:
        f.write(f"# {group}: {tag}\n\n")
        for n in items_sorted:
            meta = []
            if n["year"]: meta.append(n["year"])
            if n["status"]: meta.append(n["status"])
            meta_str = " — " + ", ".join(meta) if meta else ""
            f.write(f"- {mk_link(n)}{meta_str}\n")
