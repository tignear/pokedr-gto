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

## Suspected Remaining Gaps Versus Production Solvers

This section is a design memo, not a claim about PioSOLVER internals. It lists
places where this implementation may still be structurally worse than a mature
postflop solver. Each item should be treated as a hypothesis until measured or
proved.

## 2026-06-07 Structural Diagnosis

The current speed gap is unlikely to be explained by a single missing micro
optimization. The largest structural difference is the direction of the
calculation.

### OSS Check

Two OSS solvers were inspected locally:

- `b-inary/postflop-solver`
- `bupticybee/TexasSolver`

This corrected an earlier hypothesis. These OSS solvers are not obviously using
a full final-board-major batched terminal operator. Both expose a recursive CFR
shape where terminal evaluation is called from the recursive solve.

The important differences are still structural:

- They solve one player at a time / alternating-style, so a recursive pass
  carries one opponent reach vector and returns one player's CFV vector. The
  hot terminal evaluator does not build both players' terminal values for every
  call.
- Terminal showdown evaluation is a rank-sorted two-pass scan using one scalar
  opponent reach sum plus a `52`-entry card blocker sum. This is exactly
  `O(live_hands)` for a terminal board, not pair-quadratic.
- Chance isomorphism is built into the tree traversal. `postflop-solver`
  explicitly skips isomorphic turn/river chances and applies hand swaps when
  summing the skipped chance values. TexasSolver has a similar suit/color
  isomorphism path for chance nodes.
- The public tree is schematic with chance nodes sharing an action child rather
  than storing independent action trees per card.
- State is node-local and compact. `postflop-solver` can store regrets,
  strategies, and chance CFVs in compressed `i16` plus scale form.
- Hot loops intentionally use unsafe/index-specialized code and avoid allocator
  churn. `postflop-solver` even has a custom stack allocator feature for solve
  recursion.

Implication: before inventing a more complex board-major batched design, the
first thing to match is the simpler proven structure:

```text
schematic betting tree with isomorphic chance folding
alternating CFR pass
  opponent reach vector only
  one player CFV vector only
  terminal showdown as sorted O(H) scan with 52-card blocker sums
  exact chance isomorphism
  no per-terminal both-side Values object
```

The tree side is probably the first-order issue. If chance-equivalent turn/river
cards are expanded as independent states, every later optimization pays the same
work multiple times. The solver should first make the public tree schematic and
isomorphism-aware, then simplify the CFR pass to the one-player shape above.
Batched final-board operators may still be useful later, but they are not
required to explain why the current implementation is slower than OSS CPU
solvers.

Current real-CFR shape:

```text
for each public terminal state/path:
  build that path's live reach vectors
  for each final board under that terminal:
    compute exact terminal CFV for one reach column
  backup values through that path
```

This is exact, but it repeats the same final-board strength/blocker machinery
for many independent reach columns. Because each path has different reach, a
naive cache is not valid beyond special cases such as the uniform first
iteration.

The production-solver-like target shape should be:

```text
for each street block / final board class:
  collect many reach columns that share board tables and action skeleton
  evaluate terminal CFV as a batched exact operator
  scatter/reduce the resulting counterfactual columns into owned parent blocks
```

In symbols, the scalar terminal call computes one column:

```text
v_h = T_b(h, r_opp)
```

where `b` is a final board and `r_opp` is the opponent reach column for one
terminal path. The missing structural optimization is not caching `v_h`, because
`r_opp` changes by path. It is evaluating many columns

```text
V_b[:, j] = T_b(R_opp[:, j])
```

with shared board-local strength order, tie groups, live-combo maps, and blocker
tables, while keeping the exact blocker correction. A CPU implementation that
simply stores columns side by side was already tested and did not help; the
useful version needs board-major ownership and a memory layout where the batched
columns are consumed by the backup without writing large temporary surfaces.

Practical implications:

- CFR variant tuning cannot close a 7x-20x implementation gap by itself.
- More per-call terminal CFV micro-kernels are unlikely to produce 10x if the
  tree still duplicates isomorphic chance work.
- Exact suit/chance isomorphism is not a cosmetic compression pass; it changes
  how much tree is solved per iteration.
- The next exact solver architecture should be board/street/block-major, not
  path-state-major.
- The main validation target is unchanged: the batched operator must match the
  scalar exact terminal CFV on small trees and preserve root profile values and
  exploitability.

### A. State Representation Density

Current risk:

- The real CFR path still stores and touches many public states and private
  rows in a straightforward expanded representation.
- River and late-street rows dominate storage and update bandwidth.

Possible production-solver difference:

- Store only actual action rows and live private rows.
- Keep street-local state blocks that match cache and thread ownership.
- Avoid carrying resident state for rows that can be deterministically rebuilt
  from the betting skeleton and current boundary values.

Validation requirement:

- Report action slots, live private rows, and bytes touched per iteration by
  street.
- Show that any compressed representation produces the same root values on a
  small exact tree.

### B. Exact Board Isomorphism

Current risk:

- Concrete turn and river boards are handled separately even when suit
  permutations make them equivalent under the root board and input ranges.

Possible production-solver difference:

- Canonicalize future boards under suit permutations that preserve all
  suit-sensitive inputs.
- Carry multiplicity weights for each canonical board.
- Keep exact combo-level blockers inside the canonical class.

Why this is not lossy:

- Suit isomorphism is exact only when the suit permutation preserves the public
  board, both ranges, and the action abstraction.
- If ranges are suit-asymmetric, fewer boards are equivalent.

Validation requirement:

- For a tiny tree, solve all concrete boards and the canonical/multiplicity
  version and compare profile values and exploitability.

### C. Terminal CFV Batch Formulation

Current risk:

- Terminal CFV is faster than pair-quadratic, but the solver still calls it a
  very large number of times per iteration.
- Each call builds reach-dependent prefixes for one terminal state/board
  column.

Possible production-solver difference:

- Batch many terminal states that share a final board, strength ordering, and
  blocker tables.
- Treat opponent reach columns as a dense/sparse matrix and compute multiple
  CFV columns per pass.
- Amortize prefix construction and blocker correction across many terminal
  columns.

Validation requirement:

- Benchmark by fixed final board with many reach columns.
- Compare against the current per-call prefix/blocker path with identical
  output columns.

Current measurement:

- On `As7h2c` UTG vs BU, terminal board tasks are perfectly balanced by final
  board: `1,608,768` tasks, `1,176` unique final boards, exactly `1,368`
  tasks per final board.
- In current traversal order, those tasks are not board-major:
  `282,240` same-board runs, average run length `5.7`, max run length `13`.
- A static board-major task list would be about `61 MiB` with the current
  diagnostic `TerminalBoardTask` representation, so the schedule itself is not
  the memory blocker.
- A terminal-board smoke pass improved from about `2,084ms` to about `1,762ms`
  on `16` threads when sorted by final board. This shows locality is real for
  CFV-only work.
- Added `solve-flop --terminal-board-reuse` to measure whether final-board
  task batching has algebraic reuse, not just cache locality. On `As7h2c` UTG
  vs BU at the initial strategy, every final board had exactly `1368` terminal
  board tasks, but only `16` unique OOP reach vectors, `13` unique IP reach
  vectors, and `46` unique `(OOP reach, IP reach)` pairs. That is about
  `29.7x` task-to-reach-pair reuse per final board.
- Added `--terminal-board-reuse-after-cfr` because the initial result was
  mostly uniform-strategy reuse. On the same spot after `1` DCFR+ iteration,
  the average unique reach-pair count increased to about `351`, leaving only
  `3.9x` pair reuse. Side-value reuse is more meaningful because OOP values
  only depend on IP reach and IP values only depend on OOP reach; after `1`
  iteration the side reuse factors were about `4.1x` for OOP value and `23.6x`
  for IP value. After `4` iterations, pair reuse dropped to `1.9x`, while
  side reuse was still about `2.9x` for OOP value and `2.6x` for IP value.
  This is mathematically valid reuse, but it is still too weak to be the main
  terminal CFV optimization unless later converged strategies collapse back to
  fewer distinct reach vectors.

Open blocker:

- Real CFR terminal phase cannot simply sort tasks by final board because each
  terminal state's final-board contributions must be accumulated back into one
  `Values` row.
- A naive board-major implementation would need locks, atomics, or a huge
  `(terminal_state, final_board) -> Values` intermediate, all of which likely
  lose.
- A valid design needs owner-computes state accumulation, board-local CFV
  reuse, or a two-level reduction with bounded scratch.
- The reuse numbers above suggest the next viable design is not another local
  sort. It should build per-final-board reach-pair groups, evaluate each unique
  pair once, and scatter the reused board values back to all terminal states in
  that group.
- The after-CFR measurements weaken this plan: exact reach-pair grouping is a
  diagnostic and maybe a small optimization, not the expected order-of-magnitude
  improvement. Exact side-value grouping is more promising than pair grouping,
  but the measured `2.6x-2.9x` after four iterations says it should be treated
  as a bounded terminal-CFV optimization, not the whole solver breakthrough.
- Implemented a per-worker exact side-value cache and made it the default
  terminal path. Set `POKEDR_REAL_CFR_TERMINAL_SIDE_CACHE=0` to force the old
  no-cache path. It keys by final board, value side, and exact opponent reach
  signature, then reuses only the side values that are mathematically
  independent of the acting side's own reach. On `As7h2c` UTG vs BU, `4` DCFR+
  iterations improved from about `24.97s` total / `8.23s` terminal to about
  `21.29s` total / `5.14s` terminal after removing per-hit signature allocation
  and reusing the terminal prefix scratch. This validates the direction, but
  the total solver speedup is still modest because reach and backup phases
  remain large.
- A longer `16` iteration check still favored the side-value cache: baseline
  was about `72.71s` total / `35.41s` terminal, while the side cache was about
  `66.16s` total / `28.26s` terminal. The speedup persists, but it decays as
  strategy-dependent reach vectors diversify.
- A `64` iteration side-cache run completed in about `223.01s` total with
  `102.58s` in terminal CFV. Per-iteration terminal time stayed around
  `1.4s-1.6s` near the end (`1.39s` on iteration `64`), so the cache path did
  not obviously degrade over a longer run. The container does not currently
  have `/usr/bin/time`, so max-RSS was not captured for this run.
- After making the side cache default, an env-free `4` iteration run measured
  about `22.15s` total / `5.53s` terminal, matching the previous opt-in cache
  path. `POKEDR_REAL_CFR_TERMINAL_SIDE_CACHE=0` remains available for old-path
  comparisons.

Block-local experiment:

- Added an experimental `POKEDR_REAL_CFR_TERMINAL_BLOCK_BOARD_MAJOR=1` path.
- It processes each worker-owned terminal-state chunk in small tiles, sorts only
  the tile's final-board tasks by `cache_index`, and accumulates back into
  tile-local `TerminalAccumulator`s. This preserves owner-computes writes and
  uses no locks or atomics.
- On `As7h2c` UTG vs BU, `4` DCFR+ iterations, `16` threads:
  - current state-major reference: about `7,827ms` terminal time in the most
    recent comparable run
  - tile `16`: about `8,644ms`
  - tile `64`: about `7,877ms`
  - tile `256`: about `8,464ms`
  - tile `512`: about `7,712ms`
  - tile `1024`: about `8,017ms`
- Result: bounded block-local board-major is correct but only a small,
  noise-sized improvement at best. It does not yet capture the `~15%` win seen
  in pure terminal-board smoke, because per-state accumulation still dominates
  enough to offset much of the board locality gain.
- Follow-up on 2026-06-07 after weighted terminal partitions also failed:
  tiles `64`, `256`, `512`, and `1024` reported one-iteration terminal times
  around `2544ms`, `2745ms`, `2767ms`, and `2668ms`, all worse than the simple
  owner-computes path. The experimental env path was removed; resurrect it only
  from history if a different reduction scheme is also introduced.

### D. Street/Subgame Streaming

Current risk:

- Full flop-to-river state is too large, so later-street storage and value
  scratch dominate memory pressure.

Possible production-solver difference:

- Keep flop/turn trunk state resident.
- Stream or rebuild river chunks deterministically.
- Pass exact counterfactual boundary values back to the trunk.

Correctness caveat:

- This is not the same as arbitrary one-sided resolving.
- If a lower street is solved independently, the boundary condition must
  preserve the counterfactual values expected by the upper game.

Validation requirement:

- Start with river chunks whose boundary ranges and pot/stack state are fixed.
- Then test turn-to-river streaming against a full resident tiny tree.

### E. Action Tree Canonicalization: Mostly Not Valid

Earlier notes suggested that action lines with the same relative shape might be
canonicalized. That is generally not a valid optimization.

Why it is suspect:

- Absolute pot size, remaining stack, minimum raise, all-in threshold, and SPR
  affect legal actions and EV scale.
- `bet 50%, call` after a different previous pot is not the same game state.
- Raise sizing after a prior bet can change stack commitment and future legal
  options even if the text shape looks similar.

What is valid:

- Use one schematic betting template to generate legal concrete actions for
  each state.
- Share static metadata for the template where it is truly independent of pot
  and stack.
- Canonicalize only after proving that pot, stack, street, player, legal action
  set, and value scaling are equivalent or correctly transformed.

Default decision:

- Do not merge action states by "same looking line" as an optimization.
- Keep this as a rejected or highly constrained direction until there is a
  formal equivalence relation and a small-tree proof.

### F. Lazy Or Reduced Average Strategy Writes

Current risk:

- Strategy sums and/or strategy rows may be written more often than needed for
  convergence and output.

Possible production-solver difference:

- Use CFR+ or DCFR-style averaging schedules.
- Update average strategy less frequently, or reconstruct output strategy from
  checkpoints where valid.
- Store average strategy in `f16` or fixed-point while keeping regret in `f32`.

Correctness caveat:

- The average strategy is the output policy. Skipping or changing writes must
  preserve the intended weighting scheme.

Validation requirement:

- Compare exploitability and root strategy at fixed iteration counts against
  the current exact averaging path.

### G. Cache-Local Tree And Row Layout

Current risk:

- The tree is logically ordered, but not necessarily ordered for cache reuse.
- Different phases may traverse values, reaches, regrets, and strategy sums in
  incompatible orders.

Possible production-solver difference:

- Store states by street, dependency level, board class, and row ownership.
- Make each worker own contiguous row ranges.
- Keep hot loops branch-light and mostly sequential in memory.

Validation requirement:

- Measure bytes touched and worker imbalance per phase.
- Compare physical layouts with repeated benchmarks, not one-off runs.

### H. Exploitability Evaluation Cost

Current risk:

- Full best-response/exploitability evaluation is too expensive to run often.

Possible production-solver difference:

- Use exact exploitability at sparse intervals.
- Use cheaper convergence proxies between exact checks.
- Report practical accuracy in `BB/100` while avoiding exact BR every
  iteration.

Correctness caveat:

- A proxy cannot replace exact exploitability as the final acceptance metric.

Validation requirement:

- Track which proxy correlates with exact exploitability on fixed benchmark
  trees.

### I. Exact Future-Board Suit Isomorphism

Current risk:

- Future turn/river cards are currently treated mostly as concrete boards.
  That repeats equivalent chance branches on boards where suits are symmetric
  under the public board and both input ranges.

Possible production-solver difference:

- Enumerate only representative future-card classes.
- For omitted chance branches, add the representative branch after applying
  the corresponding private-combo suit swap.
- Keep multiplicity alone only for scalar diagnostics; it is not sufficient
  for CFR values because private hand indices move under suit swaps.

Correctness caveat:

- A suit permutation is valid only when it preserves the public board as a set,
  preserves both player ranges and preserves the action abstraction. Exact-suit
  ranges intentionally reduce or eliminate suit symmetry.

Validation requirement:

- First validate class counts and multiplicity sums without changing CFR.
- Then wire chance reduction into the solver and compare against the concrete
  chance tree on small ranges for one or more full iterations.

Initial measurement:

- With full symmetric ranges across all `22100` concrete flops, ordered
  turn-river events average `2093.120` representatives versus `2352` concrete
  events, eliminating about `11.0%`.
- The effect is highly board dependent. `As7h2c` keeps all `49` turn events
  and only removes `36` ordered turn-river events after some turns, while
  `AsKsQs` reduces turn events from `49` to `23` and ordered turn-river events
  from `2352` to `1585`.

Initial implementation:

- Terminal runouts are now grouped by exact suit isomorphism before terminal
  CFV evaluation. This is intentionally limited to showdown/all-in terminal
  runouts; merging non-terminal chance states requires carrying swapped private
  values through downstream decision nodes and is a separate change.
- Each terminal class stores the representative board plus the suit
  permutation for every concrete member. The terminal accumulator applies the
  corresponding private-combo permutation when adding representative CFVs back
  to the concrete value vectors.
- Validation: isomorphism unit tests verify multiplicity preservation and
  permutation mapping, and the ignored small-range recursive-vs-three-phase
  CFR test still matches after the terminal runout grouping.
