# Optimization Attempts

This file records optimization attempts that failed, were reverted, or were too
small to count as a strategic direction. Append entries at the bottom before
retrying similar ideas, so attempts stay in chronological order.

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

## 2026-06-06: CPU terminal CFV card-prefix blocker correction

- Tried: replace the per-combo blocker-neighbor loop in CPU terminal CFV with
  per-card strength-order prefixes, so each combo's blocker correction can be
  read from the two private-card prefix arrays.
- Expected: remove the random blocker-neighbor reads and make each side's value
  loop close to `O(combo_count)`.
- Result: much slower. On `As7h2c` with two-combo ranges,
  `1608768` terminal CFV calls on `16` threads moved from about `11.5s` to
  about `37.0s`. The `52 * (combo_count + 1)` prefix construction writes far
  more data per call than the blocker-neighbor loop reads.
- Decision: reverted. Do not retry full per-card combo-position prefixes on CPU.
  Any card-prefix direction needs a smaller strength-group representation or a
  batched formulation that amortizes prefix construction across many reach
  columns.

## 2026-06-06: CPU terminal CFV two-side fused pass

- Tried: build hero and villain prefixes together and compute both output sides
  in one outer combo loop.
- Expected: reduce duplicated prefix/order passes in
  `terminal_cfv_prefix_blocker_into`.
- Result: slower. On `As7h2c` with two-combo ranges,
  `1608768` terminal CFV calls on `16` threads moved from about `11.5s` to
  about `19.6s`. The extra prefix buffer and helper-layer shape hurt optimizer
  and memory behavior more than the saved loop helped.
- Decision: reverted. Keep the simple two-call side pass unless a fused version
  also reduces blocker work or batches several terminal columns.

## 2026-06-06: CPU terminal board phase scratch/index reuse

- Tried: precompute final-board cache indices and range-combo indices, reuse one
  `TerminalCfvScratch` and live reach buffers per worker, and assign contiguous
  chunks to workers instead of strided tasks.
- Expected: remove per-task map lookup and allocation overhead around terminal
  CFV.
- Result: modest improvement. On `As7h2c` with two-combo ranges,
  `1608768` terminal board evaluations on `16` threads improved from about
  `11.2s` to about `9.4s`. Sorting tasks by board cache index was also tried
  and was slower, around `10.5s`.
- Decision: keep scratch/index reuse and contiguous chunks. This does not solve
  terminal CFV throughput; it only removes avoidable scaffolding overhead.

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

## 2026-06-06: River subgame batch baseline

- Tried: add `solve-river` and `solve-river-batch` CLI paths to measure river
  subgames without flop/turn chance expansion. The batch path still solves
  boards sequentially, but it runs in one process and reuses the thread-local
  GPU backend.
- Expected: river-only trees should be small enough that many subgames can be
  solved quickly, and batching should expose how much time is real iteration
  work versus GPU/driver setup.
- Result: promising. On four fixed river boards from `As7h2c`, `64` iterations
  took `4.13s` total, with the first board costing `2.76s` and later boards
  `0.42-0.52s`. At `256` iterations the same four boards took `9.01s`; after
  the first board, later boards were about `1.66-1.73s` each. Each river tree
  had only `6` public decisions, `9` terminals, and `7956` private infosets.
- Decision: keep. This suggests river re-solving/batch solving is a real path
  for practical speed. It is not yet a true batched GPU formulation; the next
  step would be grouping identical river tree shapes so one dispatch sequence
  processes multiple boards/states instead of running each board as a separate
  solve.

## 2026-06-07: Sparse clear for prepared-board live reach

- Tried: replace `out.fill(0.0)` in `reach_on_prepared_board_sparse_into` with
  clearing only the indices recorded in the previous `nonzero` list.
- Expected: reduce per-terminal live-reach writes from two full prepared-board
  combo arrays to only the actually live OOP/IP combos.
- Result: slower on the practical UTG vs BU range. On `As7h2c`, `iterations=1`,
  `16` threads, terminal time moved from about `19.0s` to about `21.7s`.
  Small sparse-only ranges improved only marginally, about `37.9ms` to
  `36.1ms`.
- Decision: reverted. The prefix path is dominated elsewhere; scattered clears
  and `Vec::drain` do not beat the sequential `fill`. Do not retry this shape
  unless the live-reach representation changes enough to eliminate the dense
  prefix input entirely.

## 2026-06-07: Real CFR level-major backup scratch

- Tried: replace the flat state-indexed `Vec<Values>` backup scratch with
  `Vec<Vec<Values>>` grouped by dependency level. Terminal phase wrote level
  zero directly, and backup read lower levels immutably while writing the
  current level. This was meant to make safe layer-parallel backup possible
  without locks or unsafe code.
- Expected: preserve the same values while opening a path to parallel decision
  updates across each backup level.
- Result: slower in the single measurement used at the time. On UTG vs BU `As7h2c`,
  `iterations=4`, `16` threads, DCFR+, the previous path was about
  `35.6s` elapsed with `13.6s` backup. The level-major scratch path produced
  the same root values but regressed to about `56.5s` elapsed with `22.4s`
  backup and `15.9s` terminal. Later repeated baseline runs on the same clean
  HEAD were much slower (`~54s` to `~62s`), so the `35.6s` baseline was likely
  an outlier.
- Decision: reverted. Do not retry `Vec<Vec<Values>>` level-major scratch
  without repeated baseline measurements. If backup is parallelized, keep the
  flat contiguous value storage or use a level plan that writes into contiguous
  ranges without fragmenting the value arrays.

## 2026-06-07: Real CFR acting-combo board-blocker skip

- Tried: in real CFR decision backup, skip regret and strategy-sum updates for
  acting combos whose private cards collide with the current public board,
  setting that combo's acting value to zero.
- Expected: remove work for impossible combo rows without changing the game
  values.
- Result: slower on the practical benchmark. On UTG vs BU `As7h2c`,
  `iterations=4`, `16` threads, DCFR+, the previous path was about `35.6s`
  elapsed with `13.6s` backup. The blocker-skip path produced the same root
  values but regressed to about `55.9s` elapsed with `20.6s` backup. The extra
  branch in the hot combo loop costs more than the skipped blocked rows.
- Decision: reverted. Do not retry per-combo board-blocker branches in the
  backup hot loop. If impossible combos are skipped, use precomputed compact
  live combo ranges/lists so the loop shape is branch-free.

## 2026-06-07: Physical topological phase-state reorder

- Tried: reorder real-CFR `PhaseState`s so root-to-terminal topological order is
  also backup-level contiguous, then route terminal evaluation over the terminal
  suffix instead of all state chunks. This keeps `child_index > parent_index`
  and is the standard setup for owner-computes level-parallel backup.
- Expected: preserve values and provide contiguous backup level ranges without
  the `Vec<Vec<Values>>` locality loss.
- Result: values matched, but performance looked worse against the single
  baseline used at the time. On UTG vs BU
  `As7h2c`, `iterations=4`, `16` threads, DCFR+, the previous path was about
  `35.6s` elapsed with `13.6s` backup. Physical topological reorder was about
  `41.5s` elapsed with `14.9s` backup and `11.6s` terminal. Later repeated
  baseline runs were `~54s` to `~62s`, so this may have been an improvement
  rather than a regression.
- Decision: reverted, but classification is uncertain. Re-test with repeated
  baseline and variant runs before rejecting physical topological order. The
  idea is still valid as metadata; prefer keeping DFS order unless physical
  reorder also improves repeated measurements.

## 2026-06-07: Global flat real-CFR action storage

- Tried: move each `RealInfoset`'s `regrets` and `strategy_sum` vectors into
  solver-wide flat `Vec<f32>` arrays with per-infoset slot ranges. This was
  intended as a first step toward splitting regret/strategy rows by owned
  ranges for lock-free parallel backup.
- Expected: preserve values and make future row ownership explicit.
- Result: values matched, but the benchmark looked worse against the single
  baseline used at the time. On
  UTG vs BU `As7h2c`, `iterations=4`, `16` threads, DCFR+, the previous path
  was about `35.6s` elapsed with `13.6s` backup. Flat global action storage was
  about `41.2s` elapsed with `15.0s` backup. Later repeated baseline runs were
  `~54s` to `~62s`, so this may have been an improvement rather than a
  regression.
- Decision: reverted, but classification is uncertain. Re-test with repeated
  baseline and variant runs before rejecting global flat action storage. If
  action rows are flattened, the same patch should still aim at parallel
  owner-computes update to make the layout change pay off.

## 2026-06-07: Fixed-board multi-column terminal CFV prefix

- Tried: add a diagnostic `terminal_cfv_batch_smoke` path that evaluates
  several independent reach columns for one fixed final board with one shared
  multi-column prefix/blocker table.
- Expected: if final-board rank/blocker reuse was the missing lever, batching
  columns should beat the scalar prefix/blocker call loop while producing
  identical values.
- Result: correct but not useful on CPU. On `As7h2c`, `128` columns,
  `16` threads, all runs had `max_delta=0`, but the best batch width only
  reached parity:
  - width `1`: `0.683x`
  - width `2`: `0.857x`
  - width `4`: `1.014x`
  - width `8`: `0.885x`
  - width `16`: `0.804x`
  - width `32`: `0.678x`
  - width `64`: `0.487x`
- Decision: keep only as a diagnostic/smoke path. The scalar prefix/blocker
  path is already compact enough that this CPU column-major layout mostly adds
  strided memory traffic. A real batched terminal kernel needs a different
  memory layout or GPU-local tiling; do not promote this exact shape to the CFR
  hot path.

## 2026-06-07: Terminal fold scratch and all-in runout cache

- Tried: remove two per-fold temporary opponent-weight vectors by writing fold
  opponent weights directly into the reusable terminal `Values` slot. Also
  store terminal final-board cache indices in each `PhaseState`, so flop/turn
  all-in states no longer rebuild `terminal_boards(board)` and do a BTree lookup
  on every iteration.
- Expected: river all-in cannot avoid the final-board CFV, but flop/turn all-in
  can avoid repeated runout enumeration/cache lookup. Fold terminal values
  should avoid hot allocation.
- Result: kept. On `As7h2c` UTG vs BU, the terminal breakdown was:
  `fold_terminals=647692`, `showdown_terminals=825552`,
  `all_in_terminals=684140`, `flop_all_in_runout_evals=2352`,
  `turn_all_in_runout_evals=98784`, `river_all_in_evals=682080`, and
  `total_evals=1608768`. With terminal profiling enabled, cached runouts moved
  `board_expand_ms` from about `132ms` to `0`, and terminal wall time in the
  profiled run moved from about `2.62s` to about `2.33s`. A profile-free
  one-iteration run reported `terminal_ms=2364ms`.
- Validation: `cargo check --workspace` passed and
  `three_phase_real_cfr_matches_recursive_one_iteration_on_small_ranges` passed.
- Decision: keep. This is not the main CFV breakthrough, but it is a clean
  exact optimization for turn/flop all-in and fold terminals.

## 2026-06-07: Portable SIMD gather for terminal blocker correction

- Tried: use nightly `std::simd` gather operations to sum
  `weaker_blockers`/`stronger_blockers` for
  `terminal_cfv_prefix_blocker_board_targets_into`.
- Expected: blocker correction is a hot loop, so vectorizing several blocker
  reach reads at once might reduce scalar loop overhead.
- Result: failed. The SIMD path matched scalar values within normal f32
  reduction noise, but on `As7h2c` UTG vs BU, `1` DCFR+ iteration with
  `16` threads moved terminal time from about `2.67s` to about `4.54s`.
- Decision: removed. SIMD gather is the wrong shape here: the indices are
  random enough that gather latency and reduction overhead dominate. Do not
  retry simple gather-SIMD for this loop. A viable SIMD approach would need a
  different data layout with contiguous lanes, not gather over `u16` blocker
  lists.

## 2026-06-07: Terminal accumulator touched-index reset

- Tried: make `TerminalAccumulator` remember touched range indices and reset /
  normalize only those indices instead of clearing and scanning the full range
  vectors for every terminal state.
- Expected: reduce per-terminal memory bandwidth outside the CFV kernel.
- Result: not useful on the fixed `As7h2c` UTG vs BU one-iteration benchmark.
  The normal run reported `terminal_ms=2641ms`, worse than the previous
  post-cache baseline around `2364ms`. The branch/push bookkeeping did not
  beat dense clears for these range sizes.
- Decision: reverted. Do not retry touched-index accumulator bookkeeping unless
  terminal value vectors become much larger or the accumulator is redesigned to
  avoid copying dense `Values` slots too.

## 2026-06-07: Side-adaptive sparse terminal CFV

- Tried: choose sparse or prefix independently for each CFV side. The existing
  path only used sparse when both OOP and IP live reaches were below the
  threshold, so one low-density side was forced through prefix if the other side
  was wide.
- Expected: exploit the common shape where one side has near-threshold nonzero
  reach while the other side is wide.
- Result: slower. On `As7h2c` UTG vs BU, one profiled iteration showed
  `mixed_tasks=439128`, so the path did fire, but `terminal_ms` worsened from
  about `2552ms` to about `3377ms`. The direct sparse loop's scattered reads and
  extra branch shape outweighed the saved prefix construction.
- Decision: reverted. Keep the all-or-nothing sparse gate for now; if revisited,
  tune with a side-specific benchmark first rather than changing the production
  terminal phase directly.

## 2026-06-07: Fold/lightweight terminal partition weighting

- Tried: change terminal worker partition weights from `max(board_evals, 1)` to
  a rough cost model where fold terminals weigh `1` and showdown/all-in board
  evals weigh `2`.
- Expected: fold terminals are cheaper than terminal CFV board evals, so the
  partitioner should balance wall time better than raw board-eval counts.
- Result: slower. On `As7h2c` UTG vs BU with terminal profiling, one iteration
  reported `terminal_ms=2591ms`, worse than the raw board-eval partition's
  `~2307ms` profiled run.
- Decision: reverted. The simple raw task-count weight is better for the current
  traversal/cache behavior. Do not tune hand-written weights without a broader
  repeated benchmark harness.

## 2026-06-07: Block-local board-major terminal phase after repartitioning

- Tried: remeasure the existing experimental
  `POKEDR_REAL_CFR_TERMINAL_BLOCK_BOARD_MAJOR=1` path after adding weighted
  terminal partitions. Tiles `64`, `256`, `512`, and `1024` were tested.
- Expected: board-local sorting inside each worker chunk might recover the
  locality win seen in pure terminal-board smoke without global locks.
- Result: still worse. One-iteration `terminal_ms` values were about `2544ms`,
  `2745ms`, `2767ms`, and `2668ms`, all worse than the simple owner-computes
  state path.
- Decision: removed the experimental env path and tile worker code. The local
  sort/reduce shape does not pay for itself; a future board-major design needs a
  different state reduction scheme, not another tile-size sweep.

## 2026-06-07: Terminal side-cache key-owner/shared-cache direction

- Tried: diagnose whether the terminal side-value cache loses major reuse
  because the same `(final_board, side, opponent_reach_signature)` key appears
  in several worker-local caches. Added
  `POKEDR_REAL_CFR_SIDE_CACHE_KEY_PROFILE=1` and
  `POKEDR_REAL_CFR_PROFILE_START_ITER` so this can be measured after warmup
  instead of on the special first uniform iteration.
- Initial one-iteration result was misleading: worker-local keys were about
  `177124`, global unique keys were about `34104`, and cross-worker extra
  touches were about `143020`. That made a shared/key-owner cache look very
  promising, but iteration 1 has uniform or near-uniform reach and is not a
  representative CFR steady-state profile.
- Warmed result: with `16` DCFR+ iterations and profile starting at iteration
  `8`, worker-local misses were around `1.47M-1.57M`, while global unique keys
  were around `1.44M-1.55M`. Cross-worker extra touches were only about
  `27k-29k`, a small fraction of all misses.
- Decision: do not build a mutable shared side cache. A shared mutable
  `HashMap` requires synchronization even for reads because miss handling
  inserts and may rehash; `RwLock`/sharded locks would likely trade CFV work for
  lock contention. A lock-free immutable table or key-owner partition would only
  be worth revisiting if warmed profiles show large cross-worker key reuse.
- Rule: do not use one-iteration terminal side-cache profiles to justify
  cache-layout changes. Measure at a later iteration with
  `POKEDR_REAL_CFR_PROFILE_START_ITER` or use a repeated benchmark harness.

## 2026-06-07: Real CFR chance isomorphism and representative board state

- Tried: make the recursive real-CFR path traverse only exact
  suit-isomorphic representative turn/river cards. Skipped concrete chance
  values are added back by permuting private combo values from the representative
  child. Regret/strategy storage for the recursive path was also changed from
  all ordered turn/river boards to representative ordered boards only.
- Expected: remove duplicated chance subtrees and stop allocating state for
  board slots that the isomorphic traversal never visits.
- Result: kept. On `Td9d6h` UTG vs BU, exact future-board isomorphism reports
  ordered turn-river events `2352 -> 2053`, so the direct chance-work reduction
  is only about `12.7%` for this suit-asymmetric range. The memory/state
  reduction is larger in the real recursive path: `action_slots` moved from
  about `72.5M` to `44.2M`, `storage_gib` from `0.54` to `0.33`, and a
  `4`-iteration release run moved from about `4.81s` to `3.38s`.
- Validation: real-CFR zero-sum smoke and isomorphism unit tests passed.
- Decision: keep. This confirms that the tree/state representation was a real
  bottleneck. The current CLI planner still prints the older full-expansion
  layout estimate, so the next cleanup should make planner output report the
  same representative-board storage used by real CFR.

## 2026-06-08: Representative chance boards in the phase CFR path

- Tried: apply the same exact chance-board isomorphism to the three-phase CFR
  state graph. `PhaseState` now stores representative chance children plus the
  concrete member suit permutations. Reach propagation follows the recursive
  solver shape: representative children receive unscaled private reach, and the
  chance probability is applied only when backing values up through the chance
  state while permuting representative private-combo values back to each
  concrete member.
- Expected: reduce phase-state count, regret/strategy storage, and backup work
  without changing the solved game.
- Result: kept. On `Td9d6h` UTG vs BU with `--run-real-cfr-three-phase`,
  `4` release iterations reported `states=1,478,837`,
  `action_slots=262,856,062`, and `storage_gib=1.96`. Iteration 4 was about
  `2.24s` total across reach, terminal, and backup phases. The CLI's old
  planner estimate still prints full ordered-board storage first and needs to
  be aligned with the real phase solver.
- Validation: `cargo check --workspace`, the representative-state allocation
  test, and the ignored recursive-vs-three-phase one-iteration equality test
  passed. The equality test caught an important mistake: chance reach must not
  be divided during forward propagation because recursive CFR applies chance
  probability only during chance backup.
- Decision: keep. The next tree-side cleanup is planner/reporting consistency,
  then more structural reduction in the action tree itself.

## 2026-06-08: One-sided terminal CFV fast path

- Tried: avoid full terminal CFV work when one side has zero reach on a
  terminal board. This is exact only if the other side's counterfactual values
  are still computed: OOP values depend on IP reach, and IP values depend on OOP
  reach. The first attempt skipped the whole task when either side was empty;
  the recursive-vs-three-phase equality test caught the bug because later CFR
  updates changed. The kept version skips both sides only when both reaches are
  empty, and otherwise computes only the side whose opponent reach is nonzero.
  Board counts are still accumulated so terminal-board averaging is unchanged.
- Expected: reduce terminal side-value cache misses and CFV work in later
  iterations, where many terminal states have one side with no private reach.
- Result: kept. On `Td9d6h` UTG vs BU with `--run-real-cfr-three-phase`,
  `4` release iterations moved terminal phase time from about `4485ms` to
  about `2429ms`; total elapsed moved from about `15.96s` to `10.91s`.
  Iteration 4 preserved the previous root values
  `root_oop_value=-67.446320`, `root_ip_value=67.440964`.
- Validation: `cargo check --workspace` and the ignored
  recursive-vs-three-phase one-iteration equality test passed before the
  benchmark.
- Decision: keep. This is a real terminal-CFV reduction, not a layout-only
  tweak. The broader next target is still batching/grouping terminal side-value
  work to reduce repeated reach mapping and cache misses.

## 2026-06-08: postflop-solver reference comparison

- Tried: rerun the local `postflop-solver` reference benchmark created during
  the solver comparison work. The previous stdout had not been saved, so the
  reproducible example is `/tmp/postflop-solver/examples/bench_flop_16.rs`.
  It uses `Td9d6h`, the UTG-vs-BU ranges from
  `/tmp/postflop_flop_ranges.txt`, `pot=200`, `effective_stack=900`,
  bet sizes `60%, e, a`, raise size `2.5x`, no turn donk, river donk `50%`,
  all-in thresholds `1.5/0.15`, and merge threshold `0.1`.
- Result: `postflop-solver` reported `oop_private_hands=180`,
  `ip_private_hands=265`, `memory_f32_gib=0.671631`, and `16` iterations in
  `1258.678ms` (`78.667ms/iter`).
- Current local comparison: the current phase CFR path on the same board and
  concrete ranges, `--run-real-cfr-three-phase --iterations 16
  --state-threads 16`, reported `states=1,491,293`,
  `action_slots=309,523,075`, `storage_gib=2.31`, and `16` iterations in
  `32651.698ms` (`2040.731ms/iter`). The summed phase times were
  `reach_ms=8883.189`, `terminal_ms=13824.711`, and `backup_ms=5158.402`.
- Takeaway: this implementation is still about `25.9x` slower per iteration
  and about `3.4x` larger in f32 state storage on this comparison. The gap is
  structural, not a single terminal-CFV micro-optimization. The main suspects
  are still action-tree shape/state materialization, reach/backup dataflow, and
  missing node-local compact traversal patterns used by the reference solver.

## 2026-06-08: Parallel-first CFR rewrite plan

- Tried: stop extending the three-phase solver and add a separate
  `parallel_cfr` module that compiles a `(public node, concrete public board)`
  state plan for a future node-local solver. The plan is explicit about
  parallel cut points, action storage, and chance fanout.
- Correction: the first draft incorrectly counted the existing public tree
  shape directly and therefore did not include exact chance isomorphism. That
  was not acceptable because the older `RealCfrSolver` already had exact
  representative chance boards. The kept version uses `next_card_isomorphism`
  during compile planning, stores representative chance children, and records
  concrete member permutation codes for later value backup.
- Rejected experiment: a quick recursive chance-parallel test used multiple
  mutable references to one `RealCfrSolver` through a raw pointer. That is not a
  valid Rust aliasing model and was reverted before commit. It did show that
  child-level chance parallelism can reduce one recursive iteration from about
  `6.9s` to about `2.2s`, but the implementation shape is not reusable.
- Result: kept as a planning foundation, not a solver yet. On `Td9d6h` UTG vs
  BU with enumerated chance and exact representative chance planning, the new
  plan reported `nodes=2,452,240`, `decisions=272,738`, `chances=2,279`,
  `terminals=562,336`, `action_slots=173,821,975`,
  `ip_cfvalue_slots=105,470`, `chance_cfvalue_slots=603,935`, f32
  regret+strategy storage `1.295GiB`, and f32 reference-style storage
  `1.298GiB`. It also reported `concrete_chance_events=109,403`,
  `representative_chance_events=90,423`, and
  `chance_permutation_members=109,403`.
- Takeaway: this is still about `1.9x` the reference implementation's
  `0.671631GiB` f32 storage on the same comparison, but it is far closer than
  the old phase layout and it keeps exact chance isomorphism. The remaining
  storage gap is action-state count, not terminal CFV. The next step is to
  align the action tree and node-local storage semantics with the reference
  solver before implementing traversal.

## 2026-06-08: Arena terminal direct side-value bypass

- Tried: bypass the arena `TerminalSideValueCache` and call the existing
  terminal side-value evaluator directly into reusable scratch buffers. This
  was a check against `postflop-solver`'s terminal evaluator, which uses
  opponent reach sum plus per-card blocker subtraction rather than a nested
  hero-villain combo loop.
- Finding: our terminal side-value core already uses the same algebraic shape
  in `terminal_cfv.rs`: strength-order scan, opponent reach sums, and per-card
  blocker correction. The suspected missing terminal formula was not actually
  missing.
- Validation: the direct path passed
  `arena_cfr_oop_pass_matches_recursive_on_small_ranges`,
  `arena_cfr_exploitability_profile_is_zero_sum_on_small_ranges`, and
  `arena_cfr_parallel_matches_single_thread_on_small_ranges`.
- Result: reverted. On `Td9d6h` UTG vs BU, `4` release arena iterations with
  `16` threads, the direct bypass did not beat the existing cached side-value
  path. A noisy run put the bypass around `2.31s` for `4` iterations, while the
  clean cached baseline remains around `1.45s` for the same run shape.
- Takeaway: do not replace the arena terminal cache with a direct side-value
  bypass. Cache lookup/hash/`Arc` overhead is not the current reference gap.
  The remaining gap is more likely action tree/state shape and node-local
  traversal overhead than the terminal blocker-subtraction math itself.

## 2026-06-08: Arena decision-child parallel recursion

- Tried: copy the reference solver's broad child-parallel idea by splitting
  flop/turn decision children into disjoint subtree storage ranges and running
  them with `rayon::join`. Parent regret/strategy updates stayed after child
  values returned, so the math was intended to be exact.
- Expected: close part of the gap to `postflop-solver`, whose
  `for_each_child` runs nodes with undealt river cards in parallel.
- Result: reverted. Correctness was repairable after fixing a local
  action-row indexing bug, but performance was much worse. On `Td9d6h` UTG vs
  BU, `4` release arena iterations with `16` threads moved from about
  `1.47s` to about `4.85s`; iteration 1 was about `1.41s`.
- Follow-up after action-major storage: a safe `split_at_mut` version that
  directly filled disjoint action rows and split child subtree
  `regrets/strategy_sum` slices was also slower. With broad pre-river decision
  splitting, `16` release iterations moved to about `10971ms`; restricted to
  only very large update-player child subtrees, it was still about `6967ms`.
  The same binary with `--real-cfr-average-strategy local` was about
  `6943ms`, so the slowdown was not caused by reach-weighted averaging alone.
- Reason: this implementation created fresh terminal scratch, side caches, and
  child value buffers per decision task. That destroys terminal side-value
  reuse and adds allocator/task overhead at many small branching points. The
  reference implementation's parallel child traversal is paired with its
  node-local storage and allocator model; copying the split point alone is not
  enough.
- Takeaway: do not retry decision-child parallel recursion in the arena shape.
  If broad child parallelism is revisited, it needs worker-local persistent
  scratch/cache and a traversal layout designed around that from the start.

## 2026-06-08: Arena vs `postflop-solver` reference gap

- Baseline: `Td9d6h`, UTG vs BU ranges from `postflop-solver`, pot `200`,
  effective stack `900`, `postflop-basic`, uncompressed f32 storage.
- Reference (`/tmp/postflop-solver/examples/bench_flop_16.rs`):
  - `RAYON_NUM_THREADS=1`: `7724.364ms / 16 = 482.773ms/iter`.
  - default rayon threads: `1127.811ms / 16 = 70.488ms/iter`.
  - reported memory: `0.671631 GiB` f32.
- Current arena CFR:
  - `--state-threads 1`, `4` iterations: first/last iteration about
    `1687ms` / `2205ms`.
  - `--state-threads 16`, `16` iterations: `6716.006ms / 16 =
    419.750ms/iter`.
  - reported storage: `0.611 GiB`, `399291` states, `163287` decision states,
    `82064240` action slots, `355288` terminal evals per iteration.
- Interpretation: this is not just a missing thread-count knob. Single-thread
  arena traversal is roughly `3-4x` slower than the reference, and parallel
  scaling is also weaker. The profile shows large repeated decision work:
  `326574` strategy builds and `164128480` reach scratch writes per iteration.
  The next structural target is reducing decision-local strategy/reach/action
  value materialization, not another terminal-cache variant.

## 2026-06-08: Direct accumulation at non-updating arena decisions

- Tried: at decision nodes where the acting player is not the CFR update
  player, stop materializing the full `actions * hands` `action_values` table.
  The parent value is just the sum of child values after opponent reach has
  been multiplied into each child, so one reusable child buffer can be
  traversed and added directly into `out`.
- Correctness: `arena_cfr_parallel_matches_single_thread_on_small_ranges` and
  `arena_cfr_oop_pass_matches_recursive_on_small_ranges` pass.
- Result: small improvement only. On `Td9d6h` UTG vs BU, `16` release arena
  iterations with `16` threads moved from about `6716ms` to about `6615ms`.
- Takeaway: this removes exact redundant writes and should stay, but it is not
  the reference gap. The larger cost remains current-player decision work:
  strategy generation, own-reach propagation for strategy averaging, and
  action-value storage/update layout.

## 2026-06-08: Local average strategy experiment

- Tried: add an explicit `--real-cfr-average-strategy local` mode. This changes
  average-strategy accumulation from standard reach-weighted
  `strategy_sum += own_reach * strategy_probability` to local unweighted
  `strategy_sum += strategy_probability`. This is a practical/reference-style
  experiment, not the standard average-strategy definition.
- Follow-up: added a dedicated arena current-player branch for `local` that
  skips update-player own-reach propagation. Alternating CFR terminal values
  only need opponent reach for the updating side, and local averaging does not
  need own reach for deeper average-strategy accumulation.
- Result on `Td9d6h`, UTG vs BU ranges, pot `200`, effective stack `900`,
  `postflop-basic`, DCFR+, `16` release arena iterations with `16` threads:
  reach-weighted was about `6331ms`; local was about `6381ms`. A `4` iteration
  arena profile showed the intended micro-effect: `reach_scratch_writes` fell
  from `164,128,480` to `82,064,240`, and typical `reach_ms` fell from about
  `106-108ms` to about `59-61ms`. End-to-end time did not improve because the
  remaining cost is dominated by child traversal, terminal evaluation,
  strategy build, and update materialization.
- Exploitability check at `16` iterations was also essentially tied:
  reach-weighted `6.694 bb/100`, local `6.593 bb/100`.
- Takeaway: keep the mode for controlled comparison, but do not count it as a
  speed optimization in the current arena traversal. The practical cheat
  removes exactly the reach work it was supposed to remove, but that work is not
  large enough to close the reference gap.

## 2026-06-08: Arena action-major strategy/regret rows

- Reference check: `/tmp/postflop-solver/examples/debug_flop_storage.rs` and the
  local arena solver agree on the key storage count for the `Td9d6h` UTG vs BU
  comparison: both use `82,064,240` regret/strategy f32 slots. The large speed
  gap is therefore not explained by solving a larger regret table.
- Tried: make the arena CFR storage semantics match the reference solver's
  action-major row layout. Arena infoset slots are now read and updated as
  `action * combos + combo` for the alternating arena path. Current and average
  strategy construction also uses row-contiguous normalization with a reusable
  denominator buffer, matching the reference shape more closely than the older
  combo-major loops.
- Correctness: `cargo test -p pokedr-core arena_cfr -- --nocapture` passed, as
  did the full `cargo test -p pokedr-core -p pokedr-cli` suite.
- Result: on `Td9d6h`, UTG vs BU ranges from
  `/tmp/postflop_flop_ranges.txt`, pot `200`, effective stack `900`,
  `postflop-basic`, `16` threads, `16` release arena iterations, the clean run
  improved from the recent reach-weighted baseline around `6331ms` to
  `6030ms`.
- Intermediate failure: action-major temporary strategy without changing the
  backing arena storage was slower. It reduced `reach_ms`, but made the overall
  traversal worse because the storage and update layout still fought the row
  traversal.
- Takeaway: row layout matters, but this only closes a small part of the gap to
  `postflop-solver` (`~1128ms / 16` on the same comparison). The remaining
  structural gap is the recursive/node-local parallel traversal: the reference
  fills per-node `cfv_actions` rows in parallel for nodes before the river is
  dealt, while the arena path still relies mostly on chance-subtree splitting
  and recursive scratch/cache reuse.

## 2026-06-08: Arena unsafe current-player action-row parallelism

- Tried: add a reference-style current-player decision path that fills each
  action row in parallel. The implementation validated child subtree storage
  ranges as disjoint, then used a narrow raw-pointer wrapper to hand each worker
  only its child `regrets/strategy_sum` slice. Parent regret/strategy updates
  still happened after all child action values returned.
- Correctness: `cargo test -p pokedr-core arena_cfr -- --nocapture` passed.
- Result: reverted. On `Td9d6h`, UTG vs BU ranges from
  `/tmp/postflop_flop_ranges.txt`, pot `200`, effective stack `900`,
  `postflop-basic`, `16` threads, `16` release arena iterations, the run was
  `9584ms`. The action-major arena baseline is about `6030-6450ms` for the same
  command.
- Reason: even with direct action-row writes, the arena traversal still needs
  worker-local terminal scratch and `TerminalSideValueCache` instances. That
  splits terminal side-value reuse across action workers. The saved child-row
  write/copy work is much smaller than the duplicated terminal side-cache
  misses and extra worker scratch setup.
- Takeaway: do not retry current-player action-row parallelism as an arena
  overlay, safe or unsafe. The reference solver's speed comes from pairing
  child-row parallelism with a node-local allocator/traversal model. To pursue
  this direction, build a separate node-owned recursive solver shape instead
  of patching the arena traversal.

## 2026-06-08: Node-local CFR rewrite start

- Direction: stop patching the arena traversal for reference-style recursive
  CFR. The tree/action-slot counts already match the reference solver, so the
  remaining gap is traversal/storage shape.
- Added: a separate `NodeLocalCfrSolver` skeleton with node-owned
  `regrets/strategy_sum` rows and an unsafe cell wrapper matching the reference
  model more closely. It currently builds storage and reports counts; it does
  not solve yet.
- Validation: on `Td9d6h`, UTG vs BU ranges, pot `200`, effective stack `900`,
  `postflop-basic`, the node-local skeleton reports `399291` states, `163287`
  decision states, and `82064240` action slots, matching the arena/reference
  comparison. Storage is `0.611GiB` for f32 regret+strategy rows.
- Takeaway: this is the correct base for pursuing reference-style recursion.
  Do not add more action-parallel overlays to `ArenaAlternatingCfrSolver`.

## 2026-06-08: Node-local CFR first executable path

- Added: sequential alternating CFR recursion to `NodeLocalCfrSolver`, including
  exact chance-isomorphism accumulation, fold values, showdown/all-in terminal
  values, and DCFR+/CFR+ regret/strategy updates.
- Correctness smoke: node-local small tests pass, and the full `Td9d6h` UTG vs
  BU first iteration matches arena pass values:
  `oop_pass_value=201.607941`, `ip_pass_value=-525.215759`.
- Performance progression on `Td9d6h`, UTG vs BU, pot `200`, effective `900`,
  `postflop-basic`, `1` release iteration:
  - naive terminal prepare per call: `17630ms/iter`;
  - precomputed terminal board cache: `1969ms/iter`;
  - terminal side-value cache plus no hot-path strategy/children clones:
    `1197ms/iter`.
- Takeaway: this path is now executable but not competitive. The next necessary
  step is reference-style pre-river child parallelism in the node-local solver,
  not more arena changes.

## 2026-06-08: Node-local pre-river child parallelism

- Changed: split pre-river chance nodes and multi-action decision children
  across Rayon tasks in the node-local traversal. Each task owns its output
  vector and scratch space; the parent reduces child values and applies the
  current-node regret/strategy update once, so the CFR update equation is
  unchanged.
- Correctness smoke: focused node-local tests pass. The first full-range
  iteration still matches the arena pass values:
  `oop_pass_value=201.607941`, `ip_pass_value=-525.215759`.
- Result on `Td9d6h`, UTG vs BU, pot `200`, effective `900`,
  `postflop-basic`:
  - previous sequential node-local after terminal side cache: `1197ms/iter`;
  - parallel node-local, `1` release iteration: `303ms/iter`;
  - parallel node-local, `16` release iterations: `4473ms` solve time, with
    per-logged iteration times around `260-310ms`.
- Takeaway: node-local parallel recursion is the right direction and is already
  materially faster than the arena path. The remaining gap to the reference
  solver is no longer tree size; it is hot traversal/cache structure and
  terminal-side work inside the recursive solver.
- Failed follow-up: adding a subtree terminal-count cutoff
  (`>=4096` represented terminal board evaluations) made the same 16-iteration
  run slower, `5996ms` solve time. Small pre-river child parallelism is still
  contributing enough to keep; do not reintroduce this simple cutoff.
- Failed follow-up: replacing the exact reach-signature cache key with a
  bit-mixed fingerprint key also slowed the same 16-iteration run to `6318ms`.
  The key construction is not the current 3x gap; keep the exact signature path
  until a broader cache layout change is made.
- Follow-up: worker-local scratch reuse with Rayon `map_init` reduced the same
  16-iteration run to about `4029ms`; changing chance-node parallel aggregation
  to fold/reduce brought the best single run to about `3869ms`.
- Profile: with `POKEDR_NODE_CFR_PROFILE=1`, 16 iterations still created about
  `932k` scratch objects and saw about `1.36M` terminal side-cache hits versus
  `4.33M` misses. This confirms that cache reuse is still fragmented by nested
  parallel traversal.
- Failed follow-up: writing opponent-decision child values directly into a
  parent `action_values` buffer, matching the reference `cfv_actions` shape more
  closely, slowed the 16-iteration run to about `4209ms`. In this implementation
  the larger parent buffer traffic outweighed the avoided child vector returns.
- Failed follow-up: suppressing nested Rayon parallelism around recursive
  node-local parallel sections made the same 16-iteration run much slower
  (`~19s`). Nested work stealing is expensive, but turning it off at this level
  starves the traversal. Do not reintroduce a simple global "already in
  parallel" guard.
- Scaling check after worker-local scratch reuse/chance fold-reduce on
  `Td9d6h`, UTG vs BU, `postflop-basic`, 16 release iterations:
  `RAYON_NUM_THREADS=1` took `38331ms`, `4` took `11799ms`, `8` took `7962ms`,
  and `16` took `6348ms`. This is real parallel speedup but only about 6x at
  16 threads, so the next candidate is coarser, locality-aware scheduling or a
  persistent worker scratch/cache layout, not more tiny Rayon overlays.

## 2026-06-08: Persistent node-local worker scratch pool

- Changed: replaced per-task `NodeLocalScratch::new` in parallel node-local CFR
  sections with a solver-owned worker scratch pool. A pooled scratch is taken at
  task start and returned on drop; terminal side caches are retained within an
  iteration but cleared at iteration boundaries to avoid unbounded cache growth.
- Correctness smoke: `cargo check -p pokedr-core -p pokedr-cli` and focused
  `cargo test -p pokedr-core node_local_cfr -- --nocapture` pass.
- Result on `Td9d6h`, UTG vs BU, pot `200`, effective `900`,
  `postflop-basic`, 16 release iterations:
  - before pool: normal run about `6348ms`, profile run about `5911ms`, with
    `scratch_allocations=931863`, `terminal_cache_hits=1356101`,
    `terminal_cache_misses=4328507`;
  - after pool: normal run about `5788ms`, profile run about `5336ms`, with
    `scratch_allocations=188`, `terminal_cache_hits=1834167`,
    `terminal_cache_misses=3850441`.
- Takeaway: persistent worker scratch/cache fixes the allocator churn and
  improves side-cache locality, but only buys about 9-10% wall time. The
  remaining bottleneck is not scratch construction; it is terminal-side work
  plus fine-grained recursive task/reduce overhead.

## 2026-06-08: Disable node-local terminal side cache by default

- Rechecked the reference benchmark:
  `/tmp/postflop-solver/examples/bench_flop_16.rs` on `Td9d6h`, UTG vs BU,
  pot `200`, effective `900`, same `60%, e, a` / `2.5x` tree, reports
  `solve_ms=1025.455`, `per_iter_ms=64.091`.
- Added node-local profile counters for terminal time, fold time, and
  showdown/all-in time. With the side cache enabled, 16 iterations showed about
  `7.53M` terminal calls, `3.29M` fold calls, `4.23M` showdown calls,
  `terminal_ms=54194` aggregate CPU time, with showdown dominating
  (`showdown_ms=40660`). The side cache had more misses than hits:
  `terminal_cache_hits=1832327`, `terminal_cache_misses=3852281`.
- Result: on the same 16-iteration run, forcing the side cache on took
  `4477ms`; disabling it took `3343ms`; default-off after the change took
  `3319ms`. The current signature/hash/entry clone cache is a net slowdown for
  this workload.
- Changed default: node-local terminal side cache is now disabled unless
  `POKEDR_NODE_CFR_TERMINAL_SIDE_CACHE=1` is set.
- Takeaway: the remaining gap to the reference solver is about `3.2x` on this
  benchmark. The next target is the terminal evaluation kernel itself, not
  cache lookup plumbing.
