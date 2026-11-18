# Architecture

CIN separates action proposal from admission. Any proposer can supply an
intervention; the checker recomputes the graph predicates using the bound inputs.

## Processing flow

1. Supply a directed snapshot, threat endpoints, service endpoint pairs,
   candidate actions, and an action-count budget.
2. Propose edge denials using a planner or an external decision process.
3. Optionally apply an explicit proposer-side transformation. Record both the
   raw and repaired actions and the deterministic transformation log.
4. Build the capsule, abstract action manifest, and graph witnesses.
5. Verify the exact supplied action set and serialize the complete bundle.
6. Authenticate the bundle root through a separate trusted channel.
7. Replay with that expected root and an external verification clock.

## Modules

| Module | Responsibility |
| --- | --- |
| `capsule.py` | Capsule construction and reference intervention planner |
| `verifier.py` | Admission predicates, strict bundle validation, and replay |
| `canonical_json.py` | Canonical JSON domain, encoding, and strict parsing |
| `repair.py` | Explicit identity and subset-minimal proposal transformations |
| `candidates.py` | Candidate-action capability declarations |
| `simulator.py` | Directed reachability, edge removal, and service evaluation |
| `risk_model.py` | Fixed observed-edge ranking score |
| `baselines.py` | Alternative proposal strategies |
| `exact_optimizer.py` | MILP diagnostic over the declared action universe |
| `graph_generator.py` | Synthetic network and observation generation |
| `public_topologies.py` | Public underlay loading and synthetic overlays |
| `provenance.py` | Snapshot attestation and authorization helpers |
| `controller.py` | Offline compilation and staged-installation simulation |
| `uncertainty.py` | Checks for declared finite completion sets |
| `metrics.py`, `experiments.py`, `plotting.py` | Local evaluation tooling |
| `bare_verifier.py` | A smaller comparison verification format |

## Trust boundary

The checker evaluates directed-edge deletion. A budget limits action count,
not the sum of declared action costs. Required services survive if any directed
path still connects each declared endpoint pair. All-services preservation is
required: the current schema accepts only `service_threshold = 1.0`.

Planning may use bounded path enumeration. Admission uses unrestricted
reachability, so a route longer than the planning cutoff remains relevant.

Canonical witnesses contain the source-reachable closure after intervention,
a surviving path for every required service, and a source-to-asset path reopened
by restoring each selected edge. The last condition establishes subset
minimality, not a global minimum-cost solution.

The hidden graph used by local experiments is available to evaluation only.
The scorer, proposer, and admission checker receive the observed graph.

## Packaged fixture

`cin/data/public_topologies/abilene.json` contains a compact public underlay.
The package includes it in built distributions. Roles, threats, services,
telemetry, and action costs added by the adapter are synthetic. Attribution
and the upstream license are in `THIRD_PARTY_NOTICES.md`.
