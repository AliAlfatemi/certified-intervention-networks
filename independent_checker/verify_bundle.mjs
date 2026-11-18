#!/usr/bin/env node
/** Independently written Node.js partial semantic cross-check for CIN bundles. */
import fs from 'node:fs';
import crypto from 'node:crypto';

const args = process.argv.slice(2);
const value = name => { const i=args.indexOf(name); return i>=0 ? args[i+1] : undefined; };
const bundlePath=value('--bundle');
const expectedRoot=value('--expected-root');
const verificationTimeText=value('--verification-time');
if (!bundlePath || !expectedRoot || !verificationTimeText) {
  console.error('usage: verify_bundle.mjs --bundle FILE --expected-root HEX --verification-time ISO');
  process.exit(2);
}
const bytes=fs.readFileSync(bundlePath);
const actualRoot=crypto.createHash('sha256').update(bytes).digest('hex');
let bundle;
try { bundle=JSON.parse(bytes.toString('utf8')); } catch { console.error(JSON.stringify({passed:false,messages:['invalid JSON']})); process.exit(1); }
const messages=[]; const check=(c,m)=>{if(!c)messages.push(m)}; const edgeKey=(u,v)=>JSON.stringify([u,v]);
const compareText=(left,right)=>{const a=Array.from(left,c=>c.codePointAt(0)),b=Array.from(right,c=>c.codePointAt(0));const limit=Math.min(a.length,b.length);for(let i=0;i<limit;i++){if(a[i]!==b[i])return a[i]-b[i];}return a.length-b.length;};
const compareActions=(left,right)=>compareText(left[0],right[0])||compareText(left[1],right[1]);
const pythonTruth=value=>{if(value===null||value===false)return false;if(typeof value==='number')return value!==0;if(typeof value==='string'||Array.isArray(value))return value.length>0;if(typeof value==='object')return Object.keys(value).length>0;return Boolean(value);};
const normalized=value=>{
  if(Array.isArray(value))return value.map(normalized);
  if(value&&typeof value==='object')return Object.fromEntries(Object.keys(value).sort().map(k=>[k,normalized(value[k])]));
  return value;
};
const semanticEqual=(left,right)=>JSON.stringify(normalized(left))===JSON.stringify(normalized(right));
const asciiCanonical=value=>JSON.stringify(normalized(value)).replace(/[\u007f-\uffff]/g,c=>`\\u${c.charCodeAt(0).toString(16).padStart(4,'0')}`);
const canonicalHash=value=>crypto.createHash('sha256').update(asciiCanonical(value)).digest('hex');
const parseRfc3339Micros=text=>{
  if(typeof text!=='string')return null;
  const match=/^(\d{4})-(\d{2})-(\d{2})T(\d{2}):(\d{2}):(\d{2})(?:\.(\d{1,6}))?(Z|([+-])(\d{2}):(\d{2}))$/.exec(text);
  if(!match)return null;
  const year=Number(match[1]),month=Number(match[2]),day=Number(match[3]);
  const hour=Number(match[4]),minute=Number(match[5]),second=Number(match[6]);
  if(year<1||year>9999||month<1||month>12||hour>23||minute>59||second>59)return null;
  const local=new Date(0); local.setUTCFullYear(year,month-1,day); local.setUTCHours(hour,minute,second,0);
  if(!Number.isFinite(local.getTime())||local.getUTCFullYear()!==year||local.getUTCMonth()!==month-1||local.getUTCDate()!==day||local.getUTCHours()!==hour||local.getUTCMinutes()!==minute||local.getUTCSeconds()!==second)return null;
  let offsetMinutes=0;
  if(match[8]!=='Z'){
    const offsetHour=Number(match[10]),offsetMinute=Number(match[11]);
    if(offsetHour>23||offsetMinute>59)return null;
    offsetMinutes=(match[9]==='-'?-1:1)*(offsetHour*60+offsetMinute);
  }
  const fractionalMicros=BigInt((match[7]??'').padEnd(6,'0')||'0');
  const utcMicros=BigInt(local.getTime())*1000n+fractionalMicros-BigInt(offsetMinutes)*60000000n;
  if(utcMicros < -62135596800000000n || utcMicros > 253402300799999999n)return null;
  return utcMicros;
};

const isV1=bundle?.schema==='cin.verification-bundle.v1';
const isV2=bundle?.schema==='cin.verification-bundle.v2';
check(isV1||isV2,'unsupported bundle schema');
const gp=bundle?.graph??{}; check(gp.directed===true,'graph must be directed');
const nodes=new Map();
for(const item of gp.nodes??[]){const id=item?.id; if(typeof id!=='string'||id.length===0||nodes.has(id))messages.push('malformed or duplicate node'); if(typeof id==='string'&&id.length>0)nodes.set(id,item?.attributes??{});}
const edges=new Map(); const adjacency=new Map([...nodes.keys()].map(n=>[n,[]]));
for(const item of gp.edges??[]){const u=item?.source,v=item?.destination;if(typeof u!=='string'||typeof v!=='string'||u.length===0||v.length===0){messages.push('edge endpoints must be non-empty strings');continue;}const k=edgeKey(u,v); if(!nodes.has(u)||!nodes.has(v)||edges.has(k))messages.push('malformed duplicate or dangling edge'); edges.set(k,item?.attributes??{}); if(adjacency.has(u)&&!pythonTruth(item?.attributes?.blocked))adjacency.get(u).push(v);}
for(const arr of adjacency.values())arr.sort(compareText);
const capsule=bundle?.capsule??{}, constraints=bundle?.constraints??{}, threat=bundle?.threat_model??{};
check((isV1&&capsule.schema==='cin.defense-capsule.v1')||(isV2&&capsule.schema==='cin.defense-capsule.v2'),'unsupported capsule schema');
const sources=Array.isArray(threat.attack_sources)?threat.attack_sources:[];
const assets=Array.isArray(threat.protected_assets)?threat.protected_assets:[];
check(sources.length>0&&assets.length>0,'threat endpoints must be nonempty');
check(sources.every(n=>typeof n==='string'&&n.length>0)&&assets.every(n=>typeof n==='string'&&n.length>0),'threat endpoints must be non-empty strings');
check(new Set(sources).size===sources.length&&new Set(assets).size===assets.length,'threat endpoints must be unique');
check(sources.every(n=>nodes.has(n))&&assets.every(n=>nodes.has(n)),'threat endpoint absent');
check(sources.every(n=>!assets.includes(n)),'sources/assets overlap');
const parseActions=(value,label)=>{if(!Array.isArray(value)){messages.push(`${label} must be an array`);return [];}const out=[];for(const e of value){if(!Array.isArray(e)||e.length!==2||typeof e[0]!=='string'||typeof e[1]!=='string'||e[0].length===0||e[1].length===0){messages.push(`${label} contains malformed action`);continue;}out.push([e[0],e[1]]);}return out.sort(compareActions);};
const interventions=parseActions(capsule.interventions,'interventions'); const ikeys=interventions.map(e=>edgeKey(...e));
check(new Set(ikeys).size===ikeys.length,'duplicate intervention'); check(interventions.every(e=>edges.has(edgeKey(...e))),'intervention edge absent');
const candidates=isV2?parseActions(constraints.candidate_actions,'candidate_actions'):[...edges.keys()].map(k=>JSON.parse(k));
const candidateKeys=candidates.map(e=>edgeKey(...e));
check(new Set(candidateKeys).size===candidateKeys.length,'duplicate candidate action');
check(candidates.every(e=>edges.has(edgeKey(...e))),'candidate action edge absent');
if(isV2)check(ikeys.every(k=>candidateKeys.includes(k)),'intervention outside candidate action set');
const budget=constraints.action_budget; check(Number.isSafeInteger(budget)&&budget>=0,'invalid budget'); check(interventions.length<=budget,'budget exceeded'); check(capsule.action_budget===budget,'budget copy mismatch');
const blocked=new Set(ikeys);
function reached(source,blockedSet){const seen=new Set(); if(!nodes.has(source))return seen; const stack=[source]; seen.add(source); while(stack.length){const u=stack.pop(); for(const v of adjacency.get(u)??[]){if(blockedSet.has(edgeKey(u,v))||seen.has(v))continue;seen.add(v);stack.push(v);}}return seen;}
const residual=[]; for(const s of [...new Set(sources)].sort(compareText)){const r=reached(s,blocked);for(const t of [...new Set(assets)].sort(compareText))if(s!==t&&r.has(t))residual.push([s,t]);}
const attackBlocked=residual.length===0; check(attackBlocked,`residual source/asset pairs: ${residual.length}`);
const services=Array.isArray(constraints.required_service_pairs)?constraints.required_service_pairs:[];
check(services.every(p=>Array.isArray(p)&&p.length===2&&typeof p[0]==='string'&&typeof p[1]==='string'&&p[0].length>0&&p[1].length>0),'malformed service pair');
check(services.every(([u,v])=>u!==v&&nodes.has(u)&&nodes.has(v)),'invalid service pair');
let preserved=0;for(const [u,v] of services)if(reached(u,blocked).has(v))preserved++; const serviceFraction=services.length?preserved/services.length:1;
const tau=constraints.service_threshold;check(typeof tau==='number'&&Number.isFinite(tau)&&tau===1,'service threshold must equal 1.0');check(serviceFraction+1e-12>=tau,'service preservation below threshold');

const fresh=constraints.freshness??{}, attrs=gp.graph_attributes??{};
check(capsule.freshness&&semanticEqual(capsule.freshness,fresh),'freshness copy mismatch');
const requiredFreshness=isV2?['expires_at','issued_at','max_capsule_age_seconds','max_snapshot_age_at_issue_seconds','schema','snapshot_id','snapshot_time','topology_version']:['expires_at','issued_at','max_age_seconds','schema','snapshot_id','topology_version'];
check(Object.keys(fresh).sort().join('\u0000')===requiredFreshness.join('\u0000'),'freshness fields missing or unsupported');
check((isV1&&fresh.schema==='cin.freshness.v1')||(isV2&&fresh.schema==='cin.freshness.v2'),'invalid freshness schema');
check(typeof attrs.snapshot_id==='string'&&attrs.snapshot_id.trim()===attrs.snapshot_id&&attrs.snapshot_id.length>0,'invalid graph snapshot id');
check(typeof fresh.snapshot_id==='string'&&fresh.snapshot_id.trim()===fresh.snapshot_id&&fresh.snapshot_id.length>0,'invalid freshness snapshot id');
check(fresh.snapshot_id===attrs.snapshot_id,'snapshot id mismatch');
check(Number.isSafeInteger(attrs.topology_version)&&attrs.topology_version>0,'invalid graph topology version');
check(Number.isSafeInteger(fresh.topology_version)&&fresh.topology_version>0,'invalid freshness topology version');
check(fresh.topology_version===attrs.topology_version,'topology version mismatch');
const issued=parseRfc3339Micros(fresh.issued_at),expires=parseRfc3339Micros(fresh.expires_at),snap=parseRfc3339Micros(attrs.snapshot_time),declaredSnap=parseRfc3339Micros(isV2?fresh.snapshot_time:attrs.snapshot_time),verify=parseRfc3339Micros(verificationTimeText);
check([issued,expires,snap,verify].every(value=>value!==null),'invalid RFC 3339 timestamp');
if(isV2)check(declaredSnap!==null,'invalid declared freshness snapshot_time');
if(isV1&&issued!==null&&snap!==null)check(issued===snap,'issued_at does not equal snapshot_time');
if(isV2&&snap!==null&&declaredSnap!==null&&issued!==null){check(snap===declaredSnap,'freshness snapshot_time mismatch');check(issued>=snap,'issued_at precedes snapshot_time');check(verify!==null&&verify>=issued,'verification time precedes issuance');}
if(issued!==null&&expires!==null)check(expires>issued,'invalid expiry');
if(issued!==null&&expires!==null&&verify!==null)check(verify>=issued&&verify<expires,'verification time outside half-open interval');
if(isV1){check(Number.isSafeInteger(fresh.max_age_seconds)&&fresh.max_age_seconds>0,'invalid max age');if(issued!==null&&expires!==null&&Number.isSafeInteger(fresh.max_age_seconds)&&fresh.max_age_seconds>0)check(expires-issued===BigInt(fresh.max_age_seconds)*1000000n,'freshness interval mismatch');}
if(isV2){check(Number.isSafeInteger(fresh.max_snapshot_age_at_issue_seconds)&&fresh.max_snapshot_age_at_issue_seconds>=0,'invalid maximum snapshot age at issue');check(Number.isSafeInteger(fresh.max_capsule_age_seconds)&&fresh.max_capsule_age_seconds>0,'invalid maximum capsule age');if(snap!==null&&issued!==null&&Number.isSafeInteger(fresh.max_snapshot_age_at_issue_seconds)&&fresh.max_snapshot_age_at_issue_seconds>=0)check(issued-snap<=BigInt(fresh.max_snapshot_age_at_issue_seconds)*1000000n,'snapshot too old at issuance');if(issued!==null&&expires!==null&&Number.isSafeInteger(fresh.max_capsule_age_seconds)&&fresh.max_capsule_age_seconds>0)check(expires-issued===BigInt(fresh.max_capsule_age_seconds)*1000000n,'capsule lifetime mismatch');}

const embedded=capsule.service_safety_certificate??{};
if(isV2){
  const capsuleCandidates=parseActions(capsule.candidate_actions,'capsule candidate_actions');
  check(semanticEqual(capsuleCandidates,candidates),'capsule candidate_actions mismatch');
  const capability=capsule.candidate_action_capability??{};
  check(Object.keys(capability).sort().join('\u0000')===['candidate_action_count','candidate_actions_sha256','enforcement_domain','schema'].join('\u0000'),'candidate action capability fields invalid');
  check(capability.schema==='cin.candidate-action-capability.v1','candidate action capability schema invalid');
  check(capability.enforcement_domain==='directed-edge-denial-on-bound-observed-graph','candidate enforcement domain invalid');
  check(capability.candidate_action_count===candidates.length,'candidate action count mismatch');
  check(capability.candidate_actions_sha256===canonicalHash(candidates),'candidate action digest mismatch');
  const rawProposal=parseActions(capsule.raw_proposal,'raw_proposal');
  const repairedProposal=parseActions(capsule.repaired_proposal,'repaired_proposal');
  check(new Set(rawProposal.map(e=>edgeKey(...e))).size===rawProposal.length,'duplicate raw proposal action');
  check(new Set(repairedProposal.map(e=>edgeKey(...e))).size===repairedProposal.length,'duplicate repaired proposal action');
  check(semanticEqual(repairedProposal,interventions),'repaired proposal mismatch');
  const transformation=capsule.proposal_transformation??{};
  check(typeof transformation.transformation_version==='string'&&transformation.transformation_version.length>0,'invalid transformation version');
  check(Array.isArray(transformation.steps),'invalid repair log');
  const bindings=embedded.bindings??{};
  check(bindings.raw_interventions_sha256===canonicalHash(rawProposal),'raw proposal binding mismatch');
  check(bindings.repaired_interventions_sha256===canonicalHash(repairedProposal),'repaired proposal binding mismatch');
  check(bindings.repair_log_sha256===canonicalHash(transformation.steps??[]),'repair log binding mismatch');
  check(bindings.transformation_version===transformation.transformation_version,'transformation version binding mismatch');
  check(bindings.candidate_actions_sha256===canonicalHash(candidates),'candidate binding mismatch');
  check(bindings.candidate_action_count===candidates.length,'candidate binding count mismatch');
}

const abstract=capsule.abstract_policy_json??{}; check(abstract.schema==='cin.policy.v1','invalid abstract policy schema');
const rules=Array.isArray(abstract.rules)?abstract.rules.map(r=>[r?.source,r?.destination,r?.action]):[]; check(rules.length===interventions.length,'rule count mismatch');
check(rules.every(r=>typeof r[0]==='string'&&typeof r[1]==='string'),'abstract policy endpoints must be strings');
for(let i=0;i<interventions.length;i++)check(rules[i]?.[0]===interventions[i][0]&&rules[i]?.[1]===interventions[i][1]&&rules[i]?.[2]==='deny-edge','abstract policy mismatch');
const rollback=capsule.rollback_condition??{}; check(rollback.schema==='cin.rollback.v1','invalid rollback schema'); check(rollback.metric==='service_endpoint_reachability_fraction'&&rollback.operator==='<','invalid rollback predicate'); check(typeof rollback.threshold==='number'&&rollback.threshold===tau,'rollback threshold mismatch'); check(rollback.action==='restore_previous_policy','invalid rollback action'); check(typeof rollback.restore_policy_hash==='string'&&/^[0-9a-f]{64}$/.test(rollback.restore_policy_hash),'invalid rollback hash');
const sketch=capsule.identifier_omitting_typed_sketch??{}; check(sketch.schema==='cin.typed-sketch.v1'&&sketch.identifier_namespace==='omitted'&&sketch.raw_identifier_count===0,'invalid typed sketch');
let subsetMinimal=attackBlocked&&serviceFraction+1e-12>=tau;
for(let i=0;i<ikeys.length&&subsetMinimal;i++){const trial=new Set(ikeys.filter((_,j)=>j!==i));let stillBlocked=true;for(const s of sources){const r=reached(s,trial);if(assets.some(t=>r.has(t))){stillBlocked=false;break;}}if(stillBlocked)subsetMinimal=false;}
check(subsetMinimal,'intervention set is not subset-minimal');
check((isV1&&embedded.schema==='cin.certificate.v1')||(isV2&&embedded.schema==='cin.certificate.v2'),'invalid embedded certificate'); check((isV1&&embedded.checker_version==='cin-verifier-1.4')||(isV2&&embedded.checker_version==='cin-verifier-1.7'),'unsupported Python checker version');
check(Boolean(embedded.attack_reachability_blocked)===attackBlocked,'embedded attack decision mismatch'); check(Boolean(embedded.service_pairs_preserved)===(serviceFraction+1e-12>=tau),'embedded service decision mismatch'); check(Boolean(embedded.budget_respected)===(interventions.length<=budget),'embedded budget decision mismatch'); check(Boolean(embedded.subset_minimal)===subsetMinimal,'embedded minimality mismatch');
check(actualRoot===expectedRoot,'canonical bundle root mismatch');
const passed=messages.length===0;
console.log(JSON.stringify({schema:'cin.node-independent-check.v1',checker:'nodejs-independent-semantic-checker-1.0',passed,actual_bundle_sha256:actualRoot,expected_bundle_sha256:expectedRoot,attack_reachability_blocked:attackBlocked,residual_attack_pairs:residual,service_preservation:serviceFraction,budget_respected:interventions.length<=budget,subset_minimal:subsetMinimal,messages,scope:'partial independent cross-check: graph validity, reachability, budget, freshness, abstract policy, rollback syntax, subset minimality, and canonical-byte root; not the complete Python verifier contract'},null,2));
process.exit(passed?0:1);
