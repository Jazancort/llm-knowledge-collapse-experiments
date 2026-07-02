"""Automated consistency check for cas-dc-template.tex"""
import re
from pathlib import Path

tex = Path(__file__).parent.parent / 'paper' / 'cas-dc-template.tex'
content = tex.read_text(encoding='utf-8')
lines = content.split('\n')

issues = []

# 1. Old K0=79 values that should not appear
old_vals = ['94.9\\%', '78.0\\%', '75.9\\%', '77.2\\%', '88.6\\%', '16.9 percentage']
for val in old_vals:
    for i, line in enumerate(lines, 1):
        if val in line and not line.strip().startswith('%'):
            issues.append(f'[OLD_VALUE] Line {i}: "{val}" found: {line.strip()[:80]}')

# 2. Em-dashes (---) not allowed
for i, line in enumerate(lines, 1):
    if '---' in line and not line.strip().startswith('%'):
        issues.append(f'[EM_DASH] Line {i}: {line.strip()[:80]}')

# 3. 'governing' should have been replaced
for i, line in enumerate(lines, 1):
    if 'governing' in line.lower() and not line.strip().startswith('%'):
        issues.append(f'[GOVERNING] Line {i}: {line.strip()[:80]}')

# 4. 'five independent seeds' without 'three to five'
for i, line in enumerate(lines, 1):
    if 'five independent seeds' in line and 'three to five' not in line and not line.strip().startswith('%'):
        issues.append(f'[FIVE_SEEDS] Line {i}: {line.strip()[:80]}')

# 5. Abstract check - 'above 88'
abstract_text = content[content.find('\\begin{abstract}'):content.find('\\end{abstract}')]
if 'above 88' in abstract_text:
    issues.append('[ABSTRACT] Still contains "above 88%" (should be "near 90%")')

# 6. shortauthors
if '\\shortauthors{Azancort}' in content and 'et al' not in content[content.find('\\shortauthors'):content.find('\\shortauthors')+50]:
    issues.append('[METADATA] shortauthors missing "et al."')

# 7. Duplicate labels
labels = re.findall(r'\\label\{([^}]+)\}', content)
dupes = set([l for l in labels if labels.count(l) > 1])
if dupes:
    issues.append(f'[DUPLICATE_LABELS] {dupes}')

# 8. Undefined references
refs = re.findall(r'\\ref\{([^}]+)\}', content)
missing = set(refs) - set(labels)
if missing:
    issues.append(f'[UNDEFINED_REFS] {missing}')

# 9. pressure-gated vs pressure-dependent count
pg = content.count('pressure-gated')
pd = content.count('pressure-dependent')
print(f'pressure-gated: {pg} | pressure-dependent: {pd}')

# 10. K0 and 79 co-occurrence
for i, line in enumerate(lines, 1):
    if 'K_0' in line and '79' in line and not line.strip().startswith('%'):
        issues.append(f'[K0_79] Line {i}: {line.strip()[:100]}')

# 11. 'compensat' check (should not claim quantitative compensation)
for i, line in enumerate(lines, 1):
    if 'compensat' in line.lower() and not line.strip().startswith('%'):
        issues.append(f'[COMPENSAT] Line {i}: {line.strip()[:80]}')

# 12. Check /79 fractions
for i, line in enumerate(lines, 1):
    if '/79' in line and not line.strip().startswith('%') and 'K_0' not in line:
        issues.append(f'[FRAC_79] Line {i}: {line.strip()[:80]}')

# 13. 'order of magnitude' without qualifier
for i, line in enumerate(lines, 1):
    if 'order of magnitude' in line and not line.strip().startswith('%'):
        if not any(q in line for q in ['roughly', 'nearly', 'more than', 'approximately']):
            issues.append(f'[UNQUALIFIED_OOM] Line {i}: {line.strip()[:80]}')

# 14. Check homeostatic count (should be reasonable, not 40+)
homeo_count = sum(1 for line in lines if 'homeostatic' in line and not line.strip().startswith('%'))
print(f'homeostatic occurrences: {homeo_count}')

# 15. Check for stray 'we' at sentence starts (just count)
we_starts = sum(1 for line in lines if re.match(r'^We [a-z]', line.strip()) and not line.strip().startswith('%'))
print(f'"We" at sentence start: {we_starts}')

print(f'\n=== ISSUES FOUND: {len(issues)} ===')
for iss in issues:
    print(f'  {iss}')

if not issues:
    print('  ALL CHECKS PASSED!')
