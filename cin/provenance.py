"""Signed snapshot byte binding and abstract authorization unit checks.

Ed25519 signatures authenticate supplied snapshot bytes and discovery metadata
relative to a caller-provided trusted key. They do not prove that discovery is
complete, timely, truthful, or operationally authorized, and the deterministic
experiment key is not a production key-management system. Authorization checks
are deterministic policy tests over abstract intervention endpoints.
"""
from __future__ import annotations
from dataclasses import asdict, dataclass
from typing import Dict, Mapping, Sequence, Tuple
import base64, hashlib, json
import networkx as nx
from cryptography.exceptions import InvalidSignature
from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PrivateKey, Ed25519PublicKey
from cryptography.hazmat.primitives import serialization
from .canonical_json import canonical_json_bytes, canonical_json_value
from .verifier import (
    _parse_utc_timestamp,
    canonical_action_set,
    graph_hash,
    require_graph_provenance,
    stable_hash,
    validate_graph_provenance,
)

Edge=Tuple[str,str]

def _b64(data:bytes)->str: return base64.b64encode(data).decode("ascii")
def _unb64(text:str)->bytes: return base64.b64decode(text.encode("ascii"),validate=True)

def deterministic_experiment_keypair(seed:bytes=b"cin-aaai27-reproducible-experiment-key-v1"):
    private=Ed25519PrivateKey.from_private_bytes(hashlib.sha256(seed).digest())
    public=private.public_key()
    private_b64=_b64(private.private_bytes(serialization.Encoding.Raw,serialization.PrivateFormat.Raw,serialization.NoEncryption()))
    public_b64=_b64(public.public_bytes(serialization.Encoding.Raw,serialization.PublicFormat.Raw))
    return private_b64,public_b64

def _payload_bytes(payload:Mapping[str,object])->bytes:
    return canonical_json_bytes(payload)

def sign_snapshot_attestation(graph:nx.DiGraph,private_key_b64:str,public_key_b64:str,*,signer_id:str,discovery_source:str,
                              discovery_evidence:Mapping[str,object],issued_at:str)->Dict[str,object]:
    for field,value in (("private_key_b64",private_key_b64),("public_key_b64",public_key_b64),("signer_id",signer_id),("discovery_source",discovery_source)):
        if type(value) is not str or not value: raise ValueError(f"{field} must be a non-empty string")
    if not isinstance(discovery_evidence,Mapping): raise ValueError("discovery_evidence must be an object")
    evidence=canonical_json_value(dict(discovery_evidence))
    snapshot_time=require_graph_provenance(graph)
    parsed_issued_at=_parse_utc_timestamp(issued_at)
    if parsed_issued_at is None: raise ValueError("issued_at must be an RFC 3339 timestamp with an explicit timezone")
    if parsed_issued_at < snapshot_time: raise ValueError("issued_at must not precede graph snapshot_time")
    core={"schema":"cin.snapshot-attestation.v1","signature_algorithm":"Ed25519","signer_id":signer_id,
          "public_key_b64":public_key_b64,"graph_sha256":graph_hash(graph),"snapshot_id":graph.graph["snapshot_id"],
          "topology_version":graph.graph["topology_version"],"snapshot_time":graph.graph["snapshot_time"],
          "issued_at":issued_at,"discovery_source":discovery_source,"discovery_evidence":evidence}
    private=Ed25519PrivateKey.from_private_bytes(_unb64(private_key_b64)); signature=private.sign(_payload_bytes(core))
    return {**core,"signature_b64":_b64(signature),"attestation_sha256":stable_hash(core)}

def verify_snapshot_attestation(graph:nx.DiGraph,attestation:Mapping[str,object],*,trusted_signers:Mapping[str,str])->Dict[str,object]:
    messages=[]; signer_value=attestation.get("signer_id"); signer=signer_value if type(signer_value) is str else ""; expected=trusted_signers.get(signer)
    _,graph_messages=validate_graph_provenance(graph); messages.extend(graph_messages)
    if attestation.get("schema")!="cin.snapshot-attestation.v1": messages.append("unsupported schema")
    if attestation.get("signature_algorithm")!="Ed25519": messages.append("unsupported signature algorithm")
    if not expected or expected!=attestation.get("public_key_b64"): messages.append("untrusted signer key")
    if attestation.get("graph_sha256")!=graph_hash(graph): messages.append("graph digest mismatch")
    for key in ("snapshot_id","topology_version","snapshot_time"):
        if attestation.get(key)!=graph.graph.get(key): messages.append(f"{key} mismatch")
    attested_snapshot_time=_parse_utc_timestamp(attestation.get("snapshot_time")); attested_issued_at=_parse_utc_timestamp(attestation.get("issued_at"))
    if attested_snapshot_time is None: messages.append("snapshot_time must be an RFC 3339 timestamp with an explicit timezone")
    if attested_issued_at is None: messages.append("issued_at must be an RFC 3339 timestamp with an explicit timezone")
    if attested_snapshot_time is not None and attested_issued_at is not None and attested_issued_at < attested_snapshot_time:
        messages.append("issued_at precedes snapshot_time")
    core={k:v for k,v in attestation.items() if k not in {"signature_b64","attestation_sha256"}}
    if attestation.get("attestation_sha256")!=stable_hash(core): messages.append("attestation digest mismatch")
    if expected:
        try: Ed25519PublicKey.from_public_bytes(_unb64(expected)).verify(_unb64(str(attestation.get("signature_b64",""))),_payload_bytes(core))
        except (ValueError,InvalidSignature,base64.binascii.Error): messages.append("signature verification failed")
    return {"schema":"cin.snapshot-attestation-verification.v1","passed":not messages,"messages":messages,
            "signer_id":signer,"scope":"signature authenticates bound bytes and supplied discovery metadata, not completeness or truth"}

@dataclass(frozen=True)
class AuthorizationPolicy:
    schema:str; policy_id:str; max_actions:int; allowed_source_types:Tuple[str,...]; allowed_destination_types:Tuple[str,...]
    def to_dict(self): return asdict(self)

def authorize_interventions(graph:nx.DiGraph,interventions:Sequence[Edge],policy:AuthorizationPolicy)->Dict[str,object]:
    violations=[]; edges=list(canonical_action_set(interventions,field="authorization interventions"))
    if policy.schema!="cin.authorization-policy.v1": violations.append("unsupported policy schema")
    if len(edges)>policy.max_actions: violations.append("action count exceeds authorization")
    for u,v in edges:
        if not graph.has_edge(u,v): violations.append(f"absent edge {u}->{v}"); continue
        if str(graph.nodes[u].get("type")) not in policy.allowed_source_types: violations.append(f"source type denied: {u}")
        if str(graph.nodes[v].get("type")) not in policy.allowed_destination_types: violations.append(f"destination type denied: {v}")
    return {"schema":"cin.authorization-decision.v1","passed":not violations,"policy_id":policy.policy_id,
            "policy_sha256":stable_hash(policy.to_dict()),"violations":violations,"action_count":len(edges)}
