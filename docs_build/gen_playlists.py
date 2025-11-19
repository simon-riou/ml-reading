#!/usr/bin/env python3
"""
Generate playlist pages from playlists.yaml
"""

import yaml
from pathlib import Path
from typing import Dict, List, Any


def load_playlists() -> Dict[str, Any]:
    """Load playlists configuration from YAML file"""
    playlists_file = Path("docs/playlists.yaml")
    if not playlists_file.exists():
        return {"playlists": []}

    with open(playlists_file, 'r', encoding='utf-8') as f:
        return yaml.safe_load(f)


def load_citation_bib() -> Dict[str, Dict[str, str]]:
    """Parse CITATION.bib to extract paper metadata"""
    bib_file = Path("CITATION.bib")
    papers = {}

    if not bib_file.exists():
        return papers

    with open(bib_file, 'r', encoding='utf-8') as f:
        content = f.read()

    # Simple BibTeX parser
    entries = content.split('@')
    for entry in entries:
        if not entry.strip():
            continue

        lines = entry.strip().split('\n')
        if not lines:
            continue

        # Extract citekey from first line: article{citekey,
        first_line = lines[0]
        if '{' not in first_line:
            continue

        citekey = first_line.split('{')[1].split(',')[0].strip()

        # Extract fields
        paper_data = {'citekey': citekey}
        for line in lines[1:]:
            line = line.strip()
            if '=' in line:
                key, value = line.split('=', 1)
                key = key.strip().lower()
                value = value.strip().strip(',').strip('{}').strip('"')
                paper_data[key] = value

        papers[citekey] = paper_data

    return papers


def find_note_for_paper(citekey: str) -> str:
    """Find the markdown note file for a given citekey"""
    notes_dir = Path("docs/notes")

    # Search through all note files
    for note_file in notes_dir.rglob("*.md"):
        with open(note_file, 'r', encoding='utf-8') as f:
            content = f.read()
            # Look for venue field in frontmatter
            if f'venue: "{citekey}"' in content or f"venue: {citekey}" in content:
                # Return relative path from playlists/ directory
                return str("../" + str(note_file.relative_to("docs")))

    return ""


def generate_playlist_page(playlist: Dict[str, Any], citation_data: Dict[str, Dict[str, str]]) -> str:
    """Generate markdown content for a playlist page"""

    md = f"# {playlist.get('icon', '📚')} {playlist['title']}\n\n"
    md += f"{playlist['description']}\n\n"

    papers = playlist.get('papers', [])

    if not papers:
        md += "!!! info \"En construction\"\n"
        md += "    Cette playlist est en cours de création. Les papiers seront ajoutés prochainement.\n\n"
        return md

    md += f"!!! tip \"Parcours de lecture\"\n"
    md += f"    Cette playlist contient **{len(papers)} papiers** sélectionnés pour leur impact sur le domaine.\n\n"

    md += "---\n\n"

    # Generate paper list
    for idx, paper in enumerate(papers, 1):
        citekey = paper.get('citekey', '')
        title = paper.get('title', 'Unknown')
        year = paper.get('year', '?')
        why = paper.get('why', '')

        # Get metadata from CITATION.bib
        metadata = citation_data.get(citekey, {})
        authors = metadata.get('author', '').split(' and ')[0] if metadata.get('author') else 'Unknown'

        # Find associated note
        note_link = find_note_for_paper(citekey)

        md += f"## {idx}. {title} ({year})\n\n"

        # Authors
        if authors:
            md += f"**Auteurs**: {authors} et al.\n\n"

        # Why this paper
        if why:
            md += f"!!! quote \"Pourquoi ce papier ?\"\n"
            md += f"    {why}\n\n"

        # Links
        md += "**Liens**:\n\n"

        if note_link:
            md += f"- 📝 [Mes notes]({note_link})\n"

        if metadata.get('url'):
            md += f"- 🔗 [Article]({metadata['url']})\n"

        if metadata.get('eprint'):
            md += f"- 📄 [ArXiv](https://arxiv.org/abs/{metadata['eprint']})\n"

        md += "\n---\n\n"

    return md


def generate_index_page(playlists: List[Dict[str, Any]]) -> str:
    """Generate the main playlists index page"""

    md = "# 🎵 Playlists de Recherche\n\n"
    md += "Des collections organisées de papiers scientifiques qui racontent l'histoire du deep learning.\n\n"
    md += "Chaque playlist regroupe par ordre chronologique les papiers clés qui ont fait avancer un domaine spécifique.\n\n"
    md += "---\n\n"

    for playlist in playlists:
        icon = playlist.get('icon', '📚')
        title = playlist['title']
        description = playlist['description']
        playlist_id = playlist['id']
        paper_count = len(playlist.get('papers', []))

        md += f"## [{icon} {title}]({playlist_id}.md)\n\n"
        md += f"{description}\n\n"

        if paper_count > 0:
            md += f"**{paper_count} papiers** dans cette playlist\n\n"
        else:
            md += "*En construction*\n\n"

        md += "---\n\n"

    return md


def main():
    """Main function to generate all playlist pages"""

    # Load data
    config = load_playlists()
    playlists = config.get('playlists', [])
    citation_data = load_citation_bib()

    # Create playlists directory
    playlists_dir = Path("docs/playlists")
    playlists_dir.mkdir(exist_ok=True)

    # Generate individual playlist pages
    for playlist in playlists:
        playlist_id = playlist['id']
        content = generate_playlist_page(playlist, citation_data)

        output_file = playlists_dir / f"{playlist_id}.md"
        with open(output_file, 'w', encoding='utf-8') as f:
            f.write(content)

        print(f"✅ Generated: {output_file}")

    # Generate index page
    index_content = generate_index_page(playlists)
    index_file = playlists_dir / "index.md"
    with open(index_file, 'w', encoding='utf-8') as f:
        f.write(index_content)

    print(f"✅ Generated: {index_file}")
    print(f"\n🎉 Generated {len(playlists)} playlists")


if __name__ == "__main__":
    main()
