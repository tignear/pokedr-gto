# Optimization Attempts

This file records optimization attempts that failed, were reverted, or were too
small to count as a strategic direction. Add entries before retrying similar
ideas.

## 2026-06-06: Resident compact regret plus resident reach context

- Tried: allocate all compact regret chunks for the full `As7h2c` tree and run
  compact reach propagation using the normal resident public-tree context.
- Expected: `44` compact chunks should fit as roughly `5.44GiB` of f32 regrets,
  leaving enough room for reach buffers.
- Result: OOM on D3D12/DZN. Removing terminal/value buffers from the reach smoke
  context was not enough; one-shot submission of all sliced reach buffers also
  OOMed because temporary edge/group buffers stayed alive until submit.
- Decision: do not require all compact regret chunks and all reach work buffers
  to be resident together. Use chunk streaming and batched submits for compact
  reach, then extend the same streaming design to the real solver path.

## 2026-06-06: Protect reach-tile public ranges in compact chunks

- Tried: choose compact private CFR chunk boundaries so every reach-edge tile's
  public-infoset range fits inside a single chunk.
- Expected: allow compact reach propagation to bind one regret chunk per tile
  without splitting reach tiles.
- Result: failed as a layout direction. On the full `As7h2c` tree, the normal
  compact regret-only plan is `44` chunks with about `1.46B` action slots. The
  protected-boundary plan exploded to `465823` chunks because reach-tile ranges
  overlap densely across adjacent public infosets. Total slot count did not
  change, but the buffer/bind-group count would be unusable.
- Decision: keep compact state chunks coarse and split reach-edge work at chunk
  boundaries instead. Do not retry protected chunk boundaries.

## 2026-06-06: Dense resident chunking after natural tree expansion

- Tried: remove the postflop `max_depth` cutoff and keep using the resident
  dense CFR state, with chunking by public-infoset range as a workaround for
  wgpu's single-buffer limit.
- Expected: preserve the existing resident GPU path while allowing the natural
  full public tree to run.
- Result: failed as a strategic direction. `As7h2c` with the default action set
  expanded to about `477k` public infosets / `633M` private infosets. The old
  state layout tried to allocate `private_infosets * max_actions` slots for
  legal actions and then regrets/prediction/strategy; the first failure was a
  `resident legal actions` buffer around `10.1GB`. Chunking avoids a single
  oversized binding, and disabling prediction/strategy-sum reduces memory for a
  no-download timing run, but default `max_aggressive_actions=4` still OOMs on
  D3D12/DZN even with only regret chunks. `max_aggressive_actions=1` completes
  but is not an acceptable abstraction.
- Decision: do not retry plain dense resident chunking as the main fix. The real
  representation needs compact/sparse action state by actual public action edge
  count, not dense `max_actions` slots for every private infoset.

## 2026-06-06: Streamed strength-group card-prefix terminal CFV

- Tried: add an optional `board_count == 1` river-terminal path that streams
  `(terminal, card, strength_group)` prefix pairs through the existing terminal
  scratch buffer, then consumes those prefixes immediately in reduce. This
  avoided full materialization and replaced the blocker-neighbor loop with a
  handful of card-prefix reads.
- Expected: keep the algebraic blocker-correction win while staying within the
  existing scratch budget.
- Result: slower. On `As7h2c`, `depth=5`, `iterations=1`, baseline
  `cfv_terminal` was about `965ms`; the streamed card/group path was about
  `1055ms`. GPU smoke passed with the experimental path, so this was a
  performance failure, not a validation failure. The likely reason is that the
  partial pass becomes `53` serial scans per terminal and adds more dispatch and
  scratch traffic than the saved blocker-neighbor reads are worth.
- Decision: reverted. Do not retry this exact streamed card/group-prefix shape.
  A viable blocker algebra change needs either better intra-workgroup parallel
  prefix construction, a smaller set of blocker aggregates, or a different
  board-table batching formulation.

## 2026-06-06: Strength-group card-prefix sizing

- Tried: quantify a smaller version of terminal card-prefix blocker correction
  that indexes by showdown strength group instead of by all `1326` combo
  positions.
- Expected: preserve the algebraic win of replacing the blocker-neighbor loop
  with two-card prefix reads while reducing the memory footprint enough for GPU
  streaming.
- Result: the unique-board static table is small enough, but the reach-dependent
  terminal-weighted table is still too large if fully materialized. On
  `As7h2c`, `depth=5`, `iterations=1`, trace showed `122598` showdown
  terminals, `12857` unique final-board tables, `162564948` reduce lanes,
  `14211160` terminal-weighted strength groups, and about `6.0GB` of f32-pair
  cells for a full terminal/card/group prefix.
- Decision: do not implement a fully materialized 52-card strength-group prefix.
  The blocker loop still needs an algebraic replacement, but it must be
  streamed/tiled so card prefixes are produced and consumed locally, or it must
  batch terminal columns by board table without storing every card prefix.

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

## 2026-06-05: Dense blocker-neighbor fast path

- Tried: detect the full-combo case where every combo has a dense blocker
  neighbor row and remove the sentinel branch from terminal reduce.
- Expected: reduce branch work inside the hot blocker-correction loop without
  changing the math.
- Result: slightly slower. On `As7h2c`, `depth=5`, `32` lightweight iterations
  were about `15.43s` with the fast path and `15.33s` with the original
  sentinel path.
- Decision: reverted. The sentinel branch is not the meaningful bottleneck;
  the cost is the repeated `combo_bounds` and reach reads across boards.

## 2026-06-05: Single-board terminal reduce shader

- Tried: add a dedicated terminal reduce entry point for `board_count == 1`
  terminal groups. On `As7h2c`, almost all showdown terminals are single-board
  (`122304` terminals versus `294` with `board_count=48`), so this removes the
  outer board loop from the common path.
- Expected: reduce control-flow and index arithmetic in terminal CFV without
  changing the formula.
- Result: slower. `16` lightweight iterations moved from about `9.96s` to
  about `10.18s`.
- Decision: reverted. The hot cost is still the blocker-correction reads and
  reach reads, not the single-iteration board loop overhead.

## 2026-06-05: Terminal prefix scratch budget sweep

- Tried: sweep `POKEDR_GPU_MAX_TERMINAL_PREFIX_PAIRS` over `2M`, `4M`, `8M`,
  and `16M`.
- Expected: find a better chunk-size tradeoff between scratch memory, dispatch
  count, and cache behavior.
- Result: no meaningful speed difference. On `As7h2c`, `depth=5`, `16`
  lightweight iterations stayed around `9.96s` to `10.00s`.
- Decision: keep the existing default. Scratch budget is not the next lever.

## 2026-06-05: Table-major terminal reference ordering

- Tried: sort terminal refs by `(final_board_table, node)` inside each terminal
  group so terminals sharing the same board table are contiguous.
- Expected: improve cache locality for `combo_order`/`combo_bounds` reads and
  prepare the data layout for batched board-table kernels.
- Result: no speedup. On `As7h2c`, `depth=5`, `16` lightweight iterations were
  about `10.05s`, versus about `9.96s` with the original order.
- Decision: reverted. Contiguity alone is not enough; batching must actually
  reuse rank/blocker work inside the shader.

## 2026-06-05: Direct board-table batch4 terminal matvec

- Tried: for `board_count == 1`, skip prefix construction and evaluate up to
  four terminals sharing one final board table in one shader invocation. The
  kernel reused the rank/blocker classification across four reach vectors, but
  scanned all `1326` opponent combos directly.
- Expected: trade extra arithmetic for higher compute occupancy and reuse
  `combo_bounds` classification across terminal columns.
- Result: much slower. On `As7h2c`, `depth=5`, one iteration moved to about
  `15.05s` versus roughly `11.3s` for the prefix/blocker path.
- Decision: reverted. The prefix path is doing important asymptotic work:
  direct all-opponent matvec multiplies too much. A viable batched kernel must
  keep the prefix/rank prefix idea and only batch the blocker correction or
  terminal columns where it does not reintroduce the full opponent loop.

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

## 2026-06-05: Group-major action value materialization

- Tried: change `action_edge_tile` from one invocation per `(edge, combo)` to
  one invocation per `(decision edge group, combo)`, looping sibling actions in
  the shader and writing the same `action_values` table.
- Expected: reduce invocations and parent-side reads by roughly the branching
  factor.
- Result: no meaningful speedup. On `As7h2c`, `depth=5`, one-iteration
  `cfv_output_action_edge` stayed around `338-340ms`.
- Decision: reverted. This confirms `action_edge` is dominated by writing the
  global `action_values` materialization surface, not by invocation count or
  repeated parent reads. The next real optimization must remove or bypass that
  table for most infosets.

## 2026-06-05: Fused complete-group CFR update

- Tried: for complete decision groups, skip global `action_values`
  materialization and update regret/average strategy directly from child CFVs.
  Only split public infosets keep the old action-value fallback path.
- Expected: remove most action-value writes and the old full-infoset CFR update
  pass.
- Result: directionally correct but modest. On `As7h2c`, `depth=5`,
  one-iteration profile moved `cfr_update` from about `62ms` after the first
  fused prototype to about `7ms` once split updates were restricted to aligned
  split ranges. `cfv_output_action_edge` still costs about `225ms` even though
  only `33` split decision edges remain. `128` lightweight iterations moved
  from about `49.41s` to about `48.32s`.
- Decision: keep because it removes a real full-table update pass and preserves
  correctness for complete groups, but do not expect this path alone to produce
  the needed order-of-magnitude gain. The remaining target is eliminating or
  redesigning the split fallback/global action-value output stage.

## 2026-06-05: Decision-child aligned layer tiles

- Tried: reorder layer-local nodes in parent/child order and align the GPU
  node tile size down to the LCM of decision-node child counts. On the current
  flop tree this alignment is `12` because decision fanouts include the usual
  small action counts such as `2`, `3`, and `4`.
- Expected: prevent a decision node's child actions from being split across
  child-tile boundaries, letting every decision group use the fused CFR update
  path.
- Result: kept. On `As7h2c`, `depth=5`, split decision groups moved from
  `22` groups / `33` edges to `0`. `cfv_output_action_edge` moved from about
  `225ms` to effectively zero, and skipping now-unused denominator/strategy
  output passes moved one-iteration CFV from about `2.33s` to about `1.99s`.
  `128` lightweight iterations moved from about `48.32s` to about `46.83s`.
- Validation: GPU smoke passed, and a `32` iteration run with BR metrics
  produced finite strategies with root profile values summing to zero.
- Decision: keep. This is a real tiling/dataflow fix, not just a micro-kernel
  tweak. The next major costs are still reach propagation and terminal CFV.

## 2026-06-05: Reach shader live-mask shortcut

- Tried: use the precomputed child `combo_live` mask in the reach propagation
  shader, and skip regret/card work when a parent combo is structurally dead.
- Expected: reduce wasted reach propagation work on dead combos and avoid
  recomputing chance-card collisions.
- Result: slower. On `As7h2c`, `depth=5`, `128` lightweight iterations moved
  from about `46.83s` to about `47.08s`. The extra branch/atomic mask reads
  were not cheaper than the existing simple card comparisons.
- Decision: reverted. Do not retry this exact shape; if reach propagation is
  optimized, change the dispatch/data layout rather than adding per-edge mask
  checks.

## 2026-06-06: Full compact iteration streaming smoke

- Tried: remove dense private action output materialization for the full default
  flop tree by adding compact backup and compact fused update shaders. The
  smoke path streams reach and complete-group updates over compact public-action
  chunks instead of allocating the old dense action/reach/strategy output
  tables.
- Expected: prove the full tree can execute one CFR iteration core without the
  previous dense-output OOM.
- Result: kept as a smoke/diagnostic path. On `As7h2c`, full default tree:
  `477170` public infosets, `1100854` public actions, `1326` combos,
  `1459732404` compact action slots, `44` chunks, largest chunk `33554430`
  slots, `1470` reach dispatch slices, and `1426` compact update dispatch
  slices. One compact iteration smoke completed without OOM.
- Decision: keep as validation scaffolding, not as the final solver. It proves
  the dense table can be removed, but it still rebuilds/streams too much per
  iteration.

## 2026-06-06: Resident compact regrets on full flop

- Tried: keep compact regrets resident on GPU and run multiple full-tree
  compact iterations. A fully resident `regrets + strategy_sum` state OOM'd
  immediately. A resident-regret plus streamed-strategy diagnostic path ran
  `1` iteration in `34.94s`; after splitting backup and aggregate submissions,
  `2` iterations completed in `41.17s`.
- Expected: if VRAM could hold the compact state, reusing regrets would make
  repeated iterations practical.
- Result: not a viable final direction on the current WGPU/DZN setup. The
  compact regret table alone is about `1,459,732,404 * 4 = 5.44GiB`; adding
  average strategy doubles that before terminal/context/work buffers. In
  practice this spills into shared GPU memory (DDR5), so resident state avoids
  OOM only by becoming bandwidth-bound through shared memory.
- Decision: do not pursue full resident fp32 compact state as the main solver
  path. The next viable direction is either chunk-streamed persistent state
  with explicit host backing and bounded VRAM working sets, or a compressed
  state representation such as fp16/quantized regret and average strategy.
