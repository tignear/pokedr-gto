# Pio-Inspired Postflop Solver Plan

This document records optimizations that PioSOLVER publicly documents or strongly implies, and maps them to implementation requirements for this project.

The target is not to clone Pio internals. The target is to avoid obviously inferior representations while building a flop solver that can reach about `1 BB/100` exploitability on a practical tree.

## Publicly Documented Facts

PioSOLVER documents these relevant points:

- Postflop inputs are ranges, pot/stack, betting sizes, raise sizes, donk settings, accuracy, solve time, and thread count.
- Solution quality is reported as exploitability per hand; the docs mention solving around `1bb/100` as a practical target.
- It reports a sizeable benchmark tree around `2.2GB`.
- Practical flop trees can be solved quickly on modern CPUs.
- River spots are exact and do not use lossy abstraction.
- It supports suit/isomorphism-related settings.
- It supports `small_strats`, where strategy storage can use `float16`.
- It supports multiple algorithms, including a memory-saving original algorithm.
- It supports small saves / forgotten streets, where later streets can be recalculated or rebuilt.
- The betting tree is schematic: the betting line structure is not different per turn or river card.

## Non-Negotiable Design Consequences

### 1. Do Not Use Lossy Card Bucketing For Normal Postflop

Pio explicitly claims no lossy abstraction for postflop/river exactness. We should not use hand buckets as the main answer for flop solving.

Allowed:

- lossless suit isomorphism
- canonical board classes
- dead-card/live-combo masks
- exact hand strength ordering

Not allowed as the default:

- grouping hands into strategic buckets
- replacing exact blocker effects with bucket averages

### 2. Use A Schematic Betting Tree

The betting tree is a street/action template shared across runouts.

Implementation requirement:

- Store one betting skeleton.
- Apply it to concrete boards through board indices and live masks.
- Do not allocate independent tree nodes for every concrete turn/river runout unless explicitly building a diagnostic full tree.

### 3. Compact Action Storage

Dense `max_actions` storage over all public nodes is a bad default.

Implementation requirement:

- Store action offsets per decision node.
- Regret/strategy slots are contiguous by `(board_class, public_decision, live_combo, action_offset)`.
- Padding slots must not exist in the main state.

### 4. Small Strategy Storage

Pio exposes `small_strats`.

Implementation requirement:

- Keep regrets in `f32`.
- Permit average strategy sums in `f16` or compact fixed-point after correctness tests exist.
- The default can start with `f32`, but memory planner must show `f16` strategy memory.

### 5. Suit Isomorphism Is Required

A single flop has many future turns/rivers that are suit-equivalent with respect to the current board and ranges.

Implementation requirement:

- Canonicalize future board runouts under suit permutations that preserve the root board and suit-sensitive ranges.
- Maintain multiplicity weights.
- Retain exact combo-level blocker handling inside each canonical class.

### 6. Terminal CFV Must Not Be Pair-Quadratic Per Iteration

The current estimate for the default tree is about `2.5e12` terminal pair evaluations per iteration. That is not acceptable.

Implementation requirement:

- For each concrete/canonical board, compute hand strengths once.
- Sort live hands by strength.
- Use prefix sums of opponent reach by strength to compute win/loss/tie contributions.
- Apply blocker correction exactly, not by bucket approximation.
- Cache board strength order and blocker neighborhoods.

The target complexity should be close to:

```text
O(board_count * (H log H + H * blocker_correction))
```

not:

```text
O(board_count * H^2)
```

### 7. Street Rebuild / Streaming Is Required

Full flop-to-river resident state is currently estimated around `18-25GB` for the default tree before overhead. That is too close to hardware limits and leaves no room for buffers.

Implementation requirement:

- Support full resident mode only for tiny diagnostics.
- Main mode stores flop/turn trunk resident.
- River state is streamed or rebuilt deterministically from the schematic tree and current reach/values.
- If river strategy is needed for output, rebuild/replay the relevant river chunk.

This must be full-game CFR state partitioning, not unsafe one-sided resolving.

### 8. Memory Estimation Comes Before Solve

Pio exposes memory estimation and checks before tree build.

Implementation requirement:

- Every solve command must print or check:
  - public decisions by street
  - action slots by street
  - regret memory
  - strategy memory
  - terminal CFV work estimate
  - expected streaming chunk size
- Refuse pathological configs unless explicitly forced.

## Immediate Implementation Order

1. Finish memory/work planner.
2. Add schematic action-offset tree representation.
3. Add board canonicalization and multiplicity.
4. Add exact terminal CFV by strength sort + prefix sums.
5. Add compact CPU CFR state for a tiny but nontrivial tree.
6. Verify exploitability on river and turn subgames.
7. Only then wire flop CFR and target `1 BB/100`.

## Current Status

The current default tree has:

```text
template nodes: 1651
template decisions: 528
schematic action slots: 3.23B
regret+strategy f32 memory: 24.7GB
regret f32 + strategy f16 memory: 18.5GB
terminal pair evals per iter: 2.52e12
flop+turn-only f32 state: 110.8MB
```

This means memory alone is not enough. The terminal CFV algorithm is the first hard requirement.
