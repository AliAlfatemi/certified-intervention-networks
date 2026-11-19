# Certified Intervention Networks

[![CI](https://github.com/AliAlfatemi/certified-intervention-networks/actions/workflows/ci.yml/badge.svg)](https://github.com/AliAlfatemi/certified-intervention-networks/actions/workflows/ci.yml)
[![Python](https://img.shields.io/badge/python-3.12%2B-blue)](https://www.python.org/)
[![License: MIT](https://img.shields.io/badge/license-MIT-green)](LICENSE)

**CIN provides replayable admission control for directed graph interventions.**
It checks whether a proposed set of edge denials isolates declared threat
sources from protected assets while preserving required service connections.
The planner proposes actions; the verifier independently evaluates the exact
action set supplied at the admission boundary.

## Features

- Unrestricted directed reachability checks for threat isolation and service preservation.
- Explicit candidate-action sets, action-count budgets, and subset-minimality checks.
- Canonical verification bundles with graph, specification, action, and manifest bindings.
- Deterministic replay against an externally trusted SHA-256 root and external verification time.
- Service-aware planning, comparison policies, synthetic graph generation, and an exact MILP diagnostic.
- A separately implemented Node.js checker for a documented subset of the Python contract.

## Installation

Use Python 3.12 or newer. Node.js 24 is needed for the independent checker and
the cross-language tests. All computation runs on a CPU.

```bash
git clone https://github.com/AliAlfatemi/certified-intervention-networks.git
cd certified-intervention-networks
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -e '.[dev]'
```

On Windows, activate the environment with `.venv\Scripts\activate`.
Runtime dependencies are defined in `pyproject.toml`. The original development
environment pins are preserved in `requirements-lock.txt` as a reference;
the editable installation above resolves dependencies for your platform.

## Quick start

Run a self-contained example that generates a graph, proposes an intervention,
constructs a capsule, and verifies its serialized replay:

```bash
python scripts/run_smoke_test.py
```

Generate a verification bundle:

```bash
mkdir -p outputs
python scripts/verify_certificate.py --output outputs/bundle.json
```

The command prints the bundle root and performs an explicit historical
issuance-time self-check. To replay the bundle, supply a root obtained through
a trusted channel and an external time within its validity interval:

```bash
python scripts/verify_certificate.py \
  --input outputs/bundle.json \
  --expected-bundle-sha256 ROOT_FROM_TRUSTED_CHANNEL \
  --verification-time TIMESTAMP_WITHIN_VALIDITY_WINDOW \
  --output outputs/replay.json
```

Omit `--verification-time` for a check against the current UTC clock.
The generated demonstration uses historical snapshot timestamps, so a live
check can correctly reject it as stale. See [verification](docs/verification.md)
for the contract, time semantics, and Node.js replay instructions.

## Python API

```python
from cin.capsule import CertifiedInterventionPlanner
from cin.graph_generator import BenchmarkConfig, generate_network
from cin.risk_model import ObservedRiskScorer

instance = generate_network(
    BenchmarkConfig(size="small", noise=0.0, hidden=0.0, density="low", seed=18)
)
graph = instance.observed_graph
scorer = ObservedRiskScorer(cutoff=8)
scorer.fit(graph, instance.attack_sources, instance.protected_assets)

capsule = CertifiedInterventionPlanner().synthesize(
    graph,
    instance.attack_sources,
    instance.protected_assets,
    instance.required_service_pairs_observed,
    scorer,
    action_budget=6,
    candidate_actions=sorted(graph.edges),
)
print(capsule.service_safety_certificate["passed"])
print(capsule.interventions)
```

The planner can apply an explicit, recorded `SubsetMinimalRepair` before
submission. The verifier checks the resulting action set without modifying it.

## Development

```bash
python -m pytest -q
python -m build
```

Tests generate their own graphs and temporary bundles. The suite covers
admission predicates, canonical serialization, provenance, schema validation,
tampering, proposal transformations, CLI behavior, and cross-language checks.

A small experiment can be run locally with:

```bash
python run_all.py --quick --seeds 1 --jobs 1 \
  --output outputs/quick --no-paper-figures --require-empty-output
```

## Project layout

```text
cin/                    Python library
  data/public_topologies/  Attributed topology fixture
independent_checker/    Node.js partial semantic checker
scripts/                Verification and development commands
tests/                  Self-contained regression tests
docs/                   Architecture and verification reference
.github/                CI and contribution templates
run_all.py              Synthetic experiment runner
pyproject.toml          Package metadata and dependencies
```

## Scope

An accepted capsule establishes properties of its bound graph and declared
specification. It does not establish that discovery captured the real network
completely. Services are modeled as endpoint reachability. Backend policy
strings are research intents or pseudocode and require separate implementation
and validation before deployment. The Node.js checker implements a subset of
the Python verifier; passing both does not establish full implementation
equivalence.

Read the [architecture](docs/architecture.md), [verification reference](docs/verification.md),
and [contribution guide](CONTRIBUTING.md) for implementation details.

## License

Original project materials are distributed under the [MIT license](LICENSE).
The vendored Abilene topology is attributed in [third-party notices](THIRD_PARTY_NOTICES.md).
