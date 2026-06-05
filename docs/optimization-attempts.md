# Optimization Attempts

This file records optimization attempts that failed, were reverted, or were too
small to count as a strategic direction. Add entries before retrying similar
ideas.

## 2026-06-05: Terminal card-prefix blocker correction

- Tried: materialize per-terminal/per-board/per-card prefix sums so blocker
  correction can be read from two private cards instead of looping blocker
  neighbor combos.
- Expected: replace the `O(|N(h)|)` blocker loop in terminal CFV with a few
  prefix reads.
- Result: slower in practice. The table shape is roughly
  `terminal_chunk * board_count * 52 * (combo_count + 1)` prefix pairs, which
  adds a large write/read pass and heavy memory bandwidth pressure.
- Decision: do not retry the full 52-card prefix table unless the layout is
  changed to a much smaller strength-group or on-demand representation.

## 2026-06-05: Workgroup-parallel terminal blocker reduce

- Tried: one workgroup per `(terminal, combo)` and split the blocker-neighbor
  loop across 128 lanes with workgroup reduction.
- Expected: remove the serial `~100` opponent blocker loop from one shader
  invocation.
- Result: slower. On `As7h2c`, `depth=5`, one iteration, terminal CFV increased
  from about `740ms` to about `1454ms`. The extra workgroups, barriers, and
  low arithmetic density outweighed the reduced serial loop.
- Decision: do not retry this shape. If blocker correction is parallelized, it
  needs a different algebraic layout, not per-combo workgroup reduction.

## 2026-06-05: Resident terminal bind groups

- Tried: create terminal partial/reduce bind groups and uniform buffers once in
  the tile cache instead of per iteration.
- Expected: reduce CPU/wgpu setup overhead during terminal CFV.
- Result: no meaningful speedup. `128` lightweight iterations stayed around
  `55s`.
- Decision: setup overhead here is not the dominant cost. Keep attention on
  shader work and data movement.

## 2026-06-05: Workgroup-parallel decision aggregate

- Tried: change `decision_aggregate_tile` from one thread scanning all `1326`
  combos for each `(decision node, card slot)` into one workgroup per row with
  256 lanes reducing combo reach.
- Expected: turn the denominator prepass into a more GPU-shaped reduction and
  reduce `cfv_decision_denominator`.
- Result: only a tiny improvement. One-iteration profile moved
  `cfv_decision_denominator` from roughly `687ms` to `670ms`, and `128`
  lightweight iterations moved from `50.41s` to `49.97s`.
- Decision: reverted. The aggregate scan is not the current lever for a
  meaningful speedup; the extra barriers and workgroups do not buy enough.

## 2026-06-05: Parent-group reach propagation

- Tried: change reach propagation from one invocation per `(edge, combo)` to
  one invocation per `(parent edge group, combo)`, computing regret
  normalization once per parent/combo and looping over that parent's child
  edges.
- Expected: remove repeated regret normalizer scans for sibling actions.
- Result: only a small improvement. On `As7h2c`, `depth=5`, one-iteration
  `cfv_reach_edges` moved from about `1084ms` to about `1063ms`.
- Decision: kept for now because it is correct and modestly faster, but this
  shape is not the main path to a 10x speedup. The likely issue is that chance
  nodes also become parent-group loops, reducing card-edge parallelism. If
  revisited, split decision nodes and chance nodes so only decision reach uses
  parent-group normalization.

## 2026-06-05: Decision-only parent-group reach propagation

- Tried: keep chance reach effectively edge-major while grouping only decision
  edges by parent. This preserves card-edge parallelism for chance nodes and
  keeps regret normalization reuse for decision nodes.
- Expected: recover any lost chance parallelism from parent-group reach.
- Result: no material change. On `As7h2c`, `depth=5`, one-iteration
  `cfv_reach_edges` stayed around `1063ms`.
- Decision: keep only because it is the right algebraic shape for reach, not
  because it is a major speedup. The reach bottleneck is now more likely memory
  traffic/dispatch over tiled tree state than sibling regret normalization.

## 2026-06-05: Single-pass acting-player outputs

- Tried: remove the value-player double dispatch in `denominator_tile` and
  `action_edge_tile`; each decision node already has a single acting player, so
  the shader can select hero/villain values from `node.acting_player`.
- Expected: halve invalid invocations and dispatches in output materialization.
- Result: useful but not enough by itself. One-iteration
  `cfv_decision_denominator` moved from about `660ms` to about `508ms`; `128`
  lightweight iterations moved from `50.41s` to `49.41s`.
- Decision: keep. This is an actual algebra/dataflow cleanup, but terminal and
  reach propagation still dominate.
