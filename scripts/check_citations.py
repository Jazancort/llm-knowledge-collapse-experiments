"""Check for uncited claims and missing references."""
import re
from pathlib import Path

tex = Path(__file__).parent.parent / 'paper' / 'cas-dc-template.tex'
content = tex.read_text(encoding='utf-8')
lines = content.split('\n')

# Specific technical claims that need citations
checks = [
    ('low intrinsic dimensionality', 'aghajanyan2021'),
    ('entropy.*based.*definition', 'roy2007'),
    ('feed-forward layers function as key-value', 'geva2021'),
    ('causal tracing', 'meng2022'),
    ('neural text degeneration', 'holtzman2020'),
    ('MTLD', 'mccarthy2010'),
    ('Distinct-', 'li2016'),
    ('TriviaQA', 'joshi2017'),
    ('NormalFloat', 'dettmers2023'),
    ('truncated.*cross.*entropy', 'zibakhsh2024'),
    ('learns less and forgets less', 'biderman2024'),
    ('neural scaling laws', 'kaplan2020'),
    ('Anti-Ouroboros', 'adapala2025'),
    ('knowledge collapse', 'keisha2025'),
    ('model collapse.*encompasses', 'schaeffer2025'),
    ('diversity.*impacts', 'wang2024'),
    ('grouped-query attention', None),  # check if cited
    ('dose-response', None),  # methodology metaphor
    ('effective rank', 'roy2007'),
]

print('=== CITATION CHECK ===\n')
issues = []
for pattern, expected in checks:
    found_lines = []
    for i, line in enumerate(lines, 1):
        if line.strip().startswith('%'):
            continue
        if re.search(pattern, line, re.IGNORECASE):
            found_lines.append(i)
    
    if not found_lines:
        continue
    
    # Check first occurrence
    first = found_lines[0]
    context = ' '.join(lines[max(0, first-3):min(len(lines), first+3)])
    
    if expected:
        if expected not in context:
            # Check all occurrences
            any_cited = False
            for fl in found_lines:
                ctx = ' '.join(lines[max(0, fl-2):min(len(lines), fl+2)])
                if expected in ctx or '\\cite' in ctx:
                    any_cited = True
                    break
            if not any_cited:
                issues.append(f'MISSING CITE: "{pattern}" (first at L{first}) -> expected {expected}')
                print(f'  ⚠ "{pattern}" at L{first} -> needs \\cite{{{expected}}}')
    else:
        if '\\cite' not in context:
            print(f'  ? "{pattern}" at L{first} -> no citation nearby (may be OK)')

if not issues:
    print('  All checked claims have citations!')

# Check for "it has been shown" type claims without citations
print('\n=== GENERAL UNCITED CLAIMS ===\n')
uncited = []
gen_patterns = [
    r'it has been shown',
    r'studies have demonstrated',
    r'research has established',
    r'it is well known',
    r'as shown by(?! \w)',  # 'as shown by' without author
]
for i, line in enumerate(lines, 1):
    if line.strip().startswith('%'):
        continue
    for pat in gen_patterns:
        if re.search(pat, line, re.IGNORECASE):
            context = ' '.join(lines[max(0, i-1):min(len(lines), i+2)])
            if '\\cite' not in context:
                uncited.append(f'  L{i}: {line.strip()[:80]}')

if uncited:
    for u in uncited:
        print(u)
else:
    print('  None found!')

# Check bib file for all cited refs
print('\n=== CHECKING BIB FILE ===\n')
bib_path = tex.parent / 'cas-refs.bib'
if bib_path.exists():
    bib = bib_path.read_text(encoding='utf-8')
    cited_refs = set()
    for m in re.finditer(r'\\cite[pt]?\{([^}]+)\}', content):
        for ref in m.group(1).split(','):
            cited_refs.add(ref.strip())
    
    bib_keys = set(re.findall(r'@\w+\{(\w+)', bib))
    
    missing_in_bib = cited_refs - bib_keys
    if missing_in_bib:
        print(f'  CITED BUT NOT IN BIB: {missing_in_bib}')
    else:
        print(f'  All {len(cited_refs)} citations found in bib file!')
    
    unused = bib_keys - cited_refs
    if unused:
        print(f'  Unused bib entries ({len(unused)}): {sorted(unused)[:10]}...')
