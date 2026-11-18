# Contributing

## Local setup

Use Python 3.12+ and Node.js 24. From the repository root:

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -e '.[dev]'
python -m pytest -q
python scripts/run_smoke_test.py
```

## Changes

Open an issue for a substantial design change before investing in an
implementation. Keep pull requests focused and describe the behavior changed,
the reason, and the validation performed.

Changes to admission, hashing, time, or schema semantics should include a
regression test covering both valid and invalid inputs. Preserve deterministic
behavior and the distinction between proposal transformation and verification.
Tests should construct their own inputs or use temporary directories.

## Repository contents

Keep experiment output under `outputs/`. Do not commit result tables, generated
bundles, model checkpoints, manuscript files, credentials, private topology
data, virtual environments, or caches. Retain attribution for any new fixture.

Run `python -m pytest -q`, `python scripts/run_smoke_test.py`, and
`python -m build` before submitting a change. CI runs the self-contained suite
and validates that the packaged topology can be loaded from an installed wheel.

## Reporting problems

Include a minimal synthetic example, Python and Node.js versions, the command
run, and the expected and actual behavior. Remove tokens, credentials, and
private network identifiers before sharing logs. For a sensitive security
issue, use GitHub private vulnerability reporting if available instead of
posting exploit details in a public issue.
