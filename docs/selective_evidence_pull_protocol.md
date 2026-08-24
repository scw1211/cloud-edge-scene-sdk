# Selective road-set evidence pull protocol

## Purpose

The normal four-edge traffic path uploads compact summaries and decision
digests first for every event. Each shared road set is built once from the
canonical cut-edge endpoint fragment and exposed to its two owners as the same
read-only, content-addressed subscription. The edge keeps feature and raw
evidence in a bounded, short-lived local cache. The cloud requests feature
evidence only for an allowed risk, uncertainty, disagreement, association
conflict, or explicit-review reason. Raw evidence is considered only after
feature coordination remains insufficient and the cloud explicitly pulls it.

This mechanism does not change mutually exclusive sensor ownership and does
not replace the existing durable summary Outbox.

## Trigger and scope

A feature pull is permitted after the initial summary when at least one member
has high or severe risk, crosses the existing edge-uncertainty threshold,
reports Q4 disagreement with the specialist model or safety rule, participates
in a cross-region association conflict, or receives an explicit cloud feature
request. Ordinary events do not proactively send feature or raw evidence.

A raw pull is permitted only when feature evidence is already available and a
canonical overlap conflict still applies to a
`traffic_road_set:<road_set_id>` resource. Checked fields are source/content
SHA, preprocessing version, model version, policy version, output digest, and
action digest. Region-summary gradients, top-k differences, and generic
region-level actions do not create a road-set conflict and cannot trigger raw.

The two owner assessments must identify the same road set and exact owner pair.
Only then may `road_set_conflict_suspected=true` select it for repair.

`required_evidence_level=feature|raw` is not sufficient by itself. Feature
requires one of the closed-set reasons above. Raw additionally requires an
explicit cloud pull after feature coordination remains insufficient.

Each triggered road set must name exactly two aggregation members. Multiple
road sets may trigger in one aggregation window. Requests are de-duplicated by
member, with a maximum of four unique members and five road sets per group.
An already-uploaded authenticated raw fragment counts toward completeness and
is not fetched again. Feature evidence can be independently decoded and used
for normal deterministic decisions, but it cannot repair a raw content or
source-identity disagreement. A canonical conflict therefore requests raw
evidence from its exact two owners only after the feature stage; a raw level
hint without a residual conflict still produces zero raw callbacks.

## Edge cache and capability

`BoundedEvidenceCache` runs inside each edge service. It is bounded by TTL,
entry count, and bytes, and uses LRU eviction after removing expired entries.
It stores the complete normalized evidence set available to that event before
the summary-first planner removes richer levels from the cloud payload.

The cloud event carries only an HMAC capability locator. The capability binds:

- edge callback origin and fixed endpoint;
- scene, aggregation group key, member, and event ID;
- complete expected-member list;
- creation and expiration times;
- available evidence levels; and
- SHA-256 of the cached event content.

The callback is a `POST` to
`/api/v1/framework/evidence-cache/fetch`. The edge validates the capability,
the complete aggregation identity, expiry, content digest, and requested level.
The cloud accepts exact allowlisted origins only, checks response size, and
verifies the returned evidence SHA-256, signed level-content digest, response
HMAC, and every owner/group/window identity field before merging. It then
recomputes the feature matrix, risk output, deterministic policy output and
action from the raw fragment instead of trusting the returned action fields.

Every isolated edge process advertises its own listen origin. The partitioned
launcher rewrites `evidence_pull.public_base_url` per port. Before starting a
non-default port range, it reads the running cloud protocol and fails if any
of the four exact origins is missing from the cloud allowlist; it never widens
the SSRF allowlist or silently edits an external cloud service.

## Fail-closed behavior

The cloud reruns the affected aggregation only after every missing unique
member in the plan succeeds. The first summary result is diagnostic rather than
authoritative while a required feature callback is outstanding. Timeout,
expiry, missing locator, invalid token, identity mismatch, digest mismatch,
unavailable evidence level, response size failure, or inference failure marks
the aggregation as lacking selective-evidence completion, so actions with
`requires_cloud_confirmation=true` remain deferred even if all four summaries
arrived.

Normal members without an allowed feature reason make no callback. Diagnostics
separately record initial summary count, feature pull count, and raw pull count,
and distinguish `requested_members` (the complete two-owner pairs) from
`fetch_target_members` (only members missing the needed evidence).

## Raw-evidence boundary

The cache never fabricates a raw window. The pulled object is the canonical
cut-edge endpoint fragment for one shared road set, not a complete 43-node
partition and not the full 170-node traffic window. It is generated once from
the frozen source before partitioning, SHA-verified through manifest v2, and
mirrored read-only to the two owners without changing their mutually exclusive
primary sensor ownership. Missing raw evidence fails closed with
`requested_level_unavailable`.

## Deployment defaults

The full traffic edge configuration uses a 10-second, 2,048-entry, 64-MiB
cache. The cloud callback timeout is 50 ms and the response limit is 8 MiB.
The checked-in cloud allowlist uses explicit localhost origins for the shared
18101--18104 range, the default isolated 19101--19104 range, and the weak-network
isolated 19301--19304 range.
