# Verification reference

## Inputs and decision

The Python v2 contract binds a directed graph, nonempty source and protected
asset sets, required service pairs, explicit candidate actions, action budget,
proposal transformation, selected actions, manifest, witnesses, and time policy.

An admitted intervention must:

- select distinct actions from the declared candidate set;
- remain within the action-count budget;
- remove every directed source-to-protected-asset path;
- preserve reachability for every required service pair;
- reopen an attack path when any one selected denial is withdrawn;
- match the declared action manifest and bound artifacts;
- satisfy the supported schemas, rollback syntax, and freshness policy.

The verifier checks the exact submitted set. `SubsetMinimalRepair` is an
explicit proposer-side operation whose recorded transformation is replayed
and validated separately.

## Root and serialization

Python strictly parses the input and hashes its canonical JSON representation.
The expected SHA-256 root must come from an authenticated, independent channel.
A root copied from the same untrusted bundle establishes no authenticity.

The CLI writer emits canonical bytes. The Node.js checker hashes those raw
bytes before applying its semantic checks. Use the canonical writer when
passing a bundle between implementations.

## Time policy

V2 distinguishes snapshot time `D`, issuance `E`, external verification time
`V`, and expiry `X`:

```text
D <= E <= V < X
E - D <= max_snapshot_age_at_issue_seconds
X = E + max_capsule_age_seconds
```

Timestamps require an explicit timezone. Snapshot identity and topology version
must agree across the bound objects. A time at expiry is stale.

Synthetic examples use historical timestamps and a declared issuance delay.
Supply an explicit historical verification time for deterministic replay.
Omit it to use the current clock. These fixture times are not operational
freshness recommendations.

## Node.js partial cross-check

```bash
node independent_checker/verify_bundle.mjs \
  --bundle outputs/bundle.json \
  --expected-root ROOT_FROM_TRUSTED_CHANNEL \
  --verification-time TIMESTAMP_WITHIN_VALIDITY_WINDOW
```

The Node.js implementation independently checks core graph reachability,
candidate membership, budget, subset minimality, abstract policy equivalence,
and selected schema and freshness fields. It does not implement the complete
Python contract: exact witness equality, the complete digest map, all object
boundaries, typed-sketch checks, transformation replay, and some rollback
constraints remain outside its coverage. Python and JavaScript also use
different Unicode edge-whitespace trim sets.

## CLI exit codes

| Code | Meaning |
| --- | --- |
| `0` | Verification accepted |
| `1` | Verification completed and rejected the bundle |
| `2` | Invalid invocation or input |

## Operational limits

Replay verifies properties relative to the supplied snapshot and specification.
It does not authenticate discovery, establish hidden-topology safety, or prove
that device rules implement the abstract edge denials. Controller operations
and rollback helpers are offline simulations. Signing helpers include
deterministic experiment keys; those keys must not serve as production trust
anchors.
