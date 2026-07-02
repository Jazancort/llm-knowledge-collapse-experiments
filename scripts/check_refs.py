"""Check bibliography completeness and consistency."""
import re
from pathlib import Path

bib = Path(__file__).parent.parent / 'paper' / 'cas-refs.bib'
tex = Path(__file__).parent.parent / 'paper' / 'cas-dc-template.tex'

bib_text = bib.read_text(encoding='utf-8')
tex_text = tex.read_text(encoding='utf-8')

# Extract bib entries
bib_entries = {}
for m in re.finditer(r'@\w+\{(\w+),\s*\n(.*?)(?=\n@|\Z)', bib_text, re.DOTALL):
    key = m.group(1)
    body = m.group(2)
    year_m = re.search(r'year\s*=\s*\{?(\d{4})\}?', body)
    title_m = re.search(r'title\s*=\s*\{(.+?)\}', body)
    author_m = re.search(r'author\s*=\s*\{(.+?)\}', body)
    bib_entries[key] = {
        'year': year_m.group(1) if year_m else '????',
        'title': (title_m.group(1)[:60] if title_m else 'NO TITLE'),
        'author': (author_m.group(1)[:50] if author_m else 'NO AUTHOR'),
    }

# Extract cited refs
cited = set()
for m in re.finditer(r'\\cite[pt]?\{([^}]+)\}', tex_text):
    for ref in m.group(1).split(','):
        cited.add(ref.strip())

missing = cited - set(bib_entries.keys())
orphans = set(bib_entries.keys()) - cited

print(f'Citations in tex: {len(cited)}')
print(f'Entries in bib: {len(bib_entries)}')
print(f'Missing in bib: {len(missing)}')
print(f'Orphan entries: {len(orphans)}')
print()

if missing:
    print('ERROR - CITED BUT NOT IN BIB:')
    for m in sorted(missing):
        print(f'  {m}')
    print()

if orphans:
    print('ORPHAN ENTRIES (not cited):')
    for o in sorted(orphans):
        e = bib_entries[o]
        print(f'  {o}: {e["year"]} | {e["author"][:30]}')
    print()

print('=== ALL CITED REFERENCES ===')
for key in sorted(cited):
    if key in bib_entries:
        e = bib_entries[key]
        print(f'  {key}: {e["year"]} | {e["author"][:35]} | {e["title"][:50]}')
