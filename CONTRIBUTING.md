# Contributing

Contributions to **LLM Knowledge Collapse Experiments** are welcome, including bug fixes, reproducibility improvements, documentation updates, and additional experimental audits.

## Environment Setup

This repository uses Python `>=3.10,<3.13` with `uv` for dependency management (`pyproject.toml`):

```bash
git clone https://github.com/Jazancort/llm-knowledge-collapse-experiments.git
cd llm-knowledge-collapse-experiments
uv sync
```

Use `uv run python <script>` to run any script in `scripts/` inside the locked environment.

## Pull Request Guidelines

1. Create a descriptive branch from `main` (for example `docs/update-readme` or `fix/eval-seed-loading`).
2. Keep changes focused on a single fix or experiment script update.
3. Never commit raw model weights, large adapter checkpoints, or credentials (`.env`, API tokens).
4. Include reproduction commands or verification output in the pull request description when modifying training or analysis scripts in `scripts/` or `src/`.
5. Update `README.md` or `CITATION.cff` whenever public entrypoints, dependencies, or citation metadata change.
