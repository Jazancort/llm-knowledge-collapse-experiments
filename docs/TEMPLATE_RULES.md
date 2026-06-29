# Template Rules — Elsevier CAS-DC (Engineering Applications of AI)

**Template:** `cas-dc` (Elsevier double-column)
**Source:** Overleaf/Novo_LLM.pdf + main.md
**Class:** `elsarticle` with CAS extensions

---

## Format

- **Layout:** Double-column
- **Class file:** `cas-dc.cls`
- **Citation style:** Numbered (`model1-num-names.bst`) OR author-year (`model2-names.bst`)
- **Bibliography:** BibTeX with DOIs
- **Packages loaded by default:** natbib, geometry, fleqn, graphicx, hyperref

---

## Front Matter Required Elements

1. **Title**
2. **Authors** with affiliations (grouped or footnoted)
   - Include ORCID
   - Role tags (e.g., Researcher, Co-ordinator)
3. **Highlights** (3-5 bullet points, required by EAAI)
4. **Keywords** (typically 4-6)
5. **Abstract**

---

## Highlights Format (EAAI requires these)

```latex
\begin{highlights}
\item Research highlight 1
\item Research highlight 2
\item Research highlight 3
\end{highlights}
```

Ours should be something like:
- Recursive degradation is governed by effective training pressure, not rank alone
- A sharp boundary separates homeostatic and degradative regimes (~5% exposure change flips it)
- Low-rank QLoRA reduces one-time adaptation cost vs FFT by ~5pp across 3 seeds
- Output drift (verbosity, efficiency loss) is a diagnostic signature of above-threshold regimes
- Controlling synthetic exposure restores near-homeostatic stability regardless of selection criterion

---

## Sections Structure (from template)

Standard scientific article:
1. Introduction
2. Related Work / Background
3. Methodology
4. Results
5. Discussion
6. Conclusion
7. Appendix (optional, via `\appendix`)

---

## Figures

- Format: PDF preferred (vector), also accepts PNG/JPG
- Command: `\includegraphics` within `figure` environment
- Cross-reference via `\ref{fig:label}`
- Double-column figures: use `figure*` environment

---

## Tables

- Use `table` environment with `tabular*`
- Can use `multirow.sty`, `array.sty`
- Cross-reference via `\ref{tab:label}`

---

## Math / Theorems

- Standard `\newtheorem` for theorems/lemmas (italic)
- `\newdefinition` for definitions (roman)
- `\newproof` for proofs (upright, no counter)
- Equations: left-aligned (fleqn loaded)

---

## Citations

- Numbered style (default): `\cite{label}`
- Author-year (if needed): `\citep{}`, `\citet{}`
- Use natbib commands
- Include DOIs in bibliography

---

## Author Credit (CRediT)

The template supports CRediT taxonomy:
```latex
\credit{Conceptualization, Methodology, Writing - Original Draft}
```

At end of paper: `\printcredits`

---

## EAAI-Specific Requirements (from Guide for Authors)

**Scope requirement:** "Submitted papers should report some novel aspects of AI used for a real world engineering application and also validated using some public data sets for easy replicability of the research results."

**Our compliance:**
- Novel aspect: effective training pressure framework for recursive PEFT stability
- Engineering application: governance of synthetic fine-tuning pipelines
- Public dataset: TriviaQA (mandarjoshi/trivia_qa on HuggingFace)
- Replicability: all code on GitHub, all configs documented, seeds specified

**Abstract:** ~150-250 words, one paragraph, no equations, no references
**Highlights:** 3-5 short bullet points (mandatory)
**Keywords:** 4-6
**Article type:** Original Research Article (no formal word limit; typical 3000-8000 words with 3-8 figures)
**References:** 30-60 typical; BibTeX with DOIs

---

## Practical Notes for Our Paper

- **Max length:** EAAI typically accepts 15-25 pages (double-column). No strict limit but concise preferred.
- **Supplementary:** Can attach supplementary material for detailed tables/extended results.
- **Code/data:** Include repository link (GitHub) for reproducibility.
- **Reference format:** Numbered, with DOIs. Use BibTeX.
- **Figures:** Plan for ~8 figures. Use vector (PDF) for plots.
- **Color:** Free color in online version. Ensure readability in grayscale for print.

---

## Our Submission Checklist

- [ ] Title page with all authors, affiliations, ORCID
- [ ] Highlights (5 bullet points)
- [ ] Abstract (~250 words)
- [ ] Keywords (5-6)
- [ ] Main text (Introduction through Conclusion)
- [ ] Figures (PDF vector, properly cross-referenced)
- [ ] Tables (properly formatted)
- [ ] References (BibTeX, with DOIs)
- [ ] CRediT author statement
- [ ] Data availability statement (GitHub repo link)
- [ ] Supplementary material (extended tables, detailed metrics)
- [ ] Cover letter
