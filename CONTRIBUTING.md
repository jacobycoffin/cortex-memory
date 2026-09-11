# Contributing to Cortex

Thanks for helping make agent memory more useful, portable, and honest.

## Change standard

Every new mechanism should include:

1. a concrete failure mode it addresses;
2. the smallest implementation that can test the idea;
3. an automated scenario test;
4. an ablation or comparison plan;
5. a dashboard or CLI signal when operators need to understand it;
6. safe migration and rollback behavior for schema or lifecycle changes.

Do not justify a feature only by saying "the brain does it." Add a source, label the analogy, state its limit, and measure the agent outcome.

## Local checks

```bash
python3 -m py_compile *.py benchmarks/*.py scripts/*.py
python3 -m unittest discover -v
python3 -m pip install .   # benchmarks import the installed package
python3 scripts/benchmark.py
git diff --check
```

Changes to ranking or recall budgets should add a labeled benchmark or ablation. Changes to lifecycle behavior must remain reversible and default to a preview/shadow mode.

## Pull requests

- keep private memories, databases, vault content, credentials, and raw prompts out of commits;
- explain behavioral and schema changes;
- include before/after evidence without overstating inference speed;
- update `CHANGELOG.md` and the relevant documentation;
- preserve Python 3.10 compatibility and standard-library-only runtime unless a dependency is strongly justified.
