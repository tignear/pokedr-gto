# Full-Game Preflop Solver Design

This document specifies the next preflop solver direction: solve preflop as one
full game that includes postflop public chance and postflop action trees, rather
than using unsafe scalar boundary values.

The design intentionally borrows from the current `NodeLocalCfrSolver`, the
public solver survey in `preflop-solver-survey.md`, and the OSS/commercial
solver lessons recorded in `pio-inspired-solver-plan.md`.

## Goal

Build a heads-up full-game CFR solver that can decide limp, raise, call, fold,
and all-in frequencies preflop while accounting for postflop play in the same
game.

Initial target:

- two-player zero-sum chip EV, then HU Spin ICM as an affine EV scale;
- rank-class user ranges only;
- exact private combos internally;
- full-game public tree with exact public-card suit isomorphism;
- alternating DCFR+/scheduled DCFR+;
- exploitability target reported as `bb/100`.

Non-goals for the first implementation:

- multiway ICM;
- exact 6-max from root;
- learned value functions;
- unsafe postflop boundary tables;
- exact-suit user ranges;
- pair-loop full-range terminal evaluation.

## Core Decision

Use a full-game public tree, but do not build a naive full concrete tree.

The tree is "full game" semantically:

```text
preflop decisions
  fold terminals
  all-in showdown terminals
  called non-all-in branches
    flop chance representatives
      postflop public betting tree
        turn/river chance representatives
        postflop terminals
```

The tree is schematic physically:

```text
preflop action skeleton
postflop action skeleton
chance isomorphism classes
node/state indices into skeleton + public board class
combo permutation maps for representative chance outcomes
```

This avoids the main failure mode of full-game solving: allocating one
independent postflop action tree per concrete flop/turn/river.

## Why Not Safe Solving First

Safe solving is the right tool when a blueprint already exists and a local
subgame must be refined, especially for off-tree actions. It is not obviously
faster for the first in-process HU Spin/preflop solver because it adds:

- boundary constraint construction;
- opponent CFV bounds;
- subgame setup and synchronization;
- extra correctness surface at every boundary;
- re-solving or value-table management.

If the full-game state fits in memory and the tree is schematic, one in-process
full-game CFR traversal should be simpler and likely faster than repeatedly
crossing a boundary. Safe solving remains the fallback only when memory or
real-time off-tree handling forces decomposition.

## Existing Code To Reuse

### Keep

- `RangeSpec` and `ComboWeight`.
  - User input should remain rank-class and weighted, not exact suit combos.
  - Internally, concrete combos remain the exact private state.
- `cards` and hand evaluation helpers.
- `isomorphism`:
  - suit permutations;
  - full-deck flop representatives;
  - next-card representatives;
  - private combo permutation maps.
- `terminal_cfv`:
  - sorted-strength terminal CFV;
  - blocker-correction formula;
  - prepared terminal board tables.
- `ActionAbstraction`, sizing parsing, and postflop tree generation ideas.
- `RealCfrConfig` / CFR variant definitions.

### Reuse Carefully

`NodeLocalCfrSolver` is a correctness and API reference, not the final
full-game engine.

Useful parts:

- action-major regret/strategy row semantics;
- profile EV and best-response testing patterns;
- terminal CFV integration;
- chance permutation handling;
- snapshot/viewer concepts.

Parts that should not be copied blindly:

- per-public-node owned storage for a fully expanded postflop tree;
- both-side value materialization at every traversal point;
- caches that grow with visited terminal boards;
- pair-loop exact terminal evaluation;
- any hidden fallback that changes exactness.

The new solver should be written as a new module rather than gradually bending
`NodeLocalCfrSolver` into a preflop solver. Keep the old solver as the postflop
reference until the new path matches invariants.

## Data Model

### FullGameConfig

```text
FullGameConfig
  blinds:
    small_blind
    big_blind
    ante
  stacks:
    sb_stack
    bb_stack
  utility:
    ChipEv | HuIcm { first_prize, second_prize }
  ranges:
    sb_range: RangeSpec
    bb_range: RangeSpec
  preflop_actions: PreflopActionTemplate
  postflop_actions: ActionAbstraction
  chance:
    flop: Isomorphic
    turn: Isomorphic
    river: Isomorphic
  solver:
    variant
    target_bb100
    max_iterations
    threads
```

### Public State

Use a unified enum:

```text
FullPublicState =
  Preflop(PreflopState)
  Postflop(PostflopState)
```

Preflop state:

```text
PreflopState
  pot
  sb_stack
  bb_stack
  sb_commit
  bb_commit
  to_call
  min_raise_to
  last_raise_size
  raises_this_round
  player_to_act
  line_kind
```

Postflop state can wrap the current `PublicState` with seat mapping:

```text
PostflopState
  public_state: PublicState
  oop_seat: BB
  ip_seat: SB
```

For HU after preflop, BB is OOP and SB/button is IP.

### Public Tree Nodes

```text
FullPublicNode
  id
  state
  kind:
    Decision { player, actions, action_offset }
    Chance { chance_kind, representatives, multiplicities, combo_swaps }
    Terminal { terminal_kind }
  children
```

Chance nodes must carry enough data to map child values from representative
coordinates back to the parent coordinate system:

```text
ChanceRepresentative
  public_cards
  multiplicity
  permutation_to_representative
  sb_combo_permutation
  bb_combo_permutation
```

## Preflop Action Template

Start with a realistic but small HU template:

```text
SB root:
  Fold
  Limp
  RaiseTo 2.0bb
  RaiseTo 2.5bb
  AllIn only when stack/pot threshold says jam is natural

BB vs limp:
  Check
  RaiseTo 3.0bb
  RaiseTo 4.0bb
  AllIn only near shallow-stack threshold

Facing raise:
  Fold
  Call
  RaiseTo geometric or jam candidate

Facing re-raise:
  Fold
  Call
  AllIn
```

Rules:

- do not allow all-in at every node by default;
- all-in appears when SPR/commit threshold makes it strategically natural;
- after a raise and re-raise, reduce sizing choices aggressively;
- preflop min-raise and stack legality must be exact;
- called non-all-in branches reach flop chance, not scalar terminals.

This template should be data-driven and printed by the planner. Bad defaults
are dangerous because preflop strategy is highly sensitive to action set.

## CFR State Layout

Use compact per-decision rows:

```text
DecisionInfo
  public_node
  player
  action_offset
  action_count
  combo_count

regrets:       Vec<f32>
strategy_sum: Vec<f32> initially f32, later f16/fixed optional
```

Index:

```text
slot = decision.action_offset
     + combo_local_index * decision.action_count
     + action_index
```

Do not use dense `max_actions` padding.

Do not allocate independent strategy/regret state for unreachable/dead combos
if the public board is already known. For preflop nodes all combos are live; for
postflop nodes, live masks should be used for iteration and output. Whether
dead combo rows exist physically should be decided by the memory planner:

- simple mode: fixed 1326 rows with dead masks;
- compact mode: per-board live combo lists and permutation maps.

The first implementation may use fixed 1326 rows for simpler correctness, but
the planner must report the compact estimate so we do not normalize around the
wrong memory shape.

## Traversal

Use alternating CFR. One update pass computes one side's counterfactual values
against the opponent reach.

```text
for iteration in 1..=max_iterations:
  update_side(SB, root, sb_reach, bb_reach)
  update_side(BB, root, sb_reach, bb_reach)
  maybe_report_exploitability()
  stop if target_bb100 reached
```

The recursive shape should be:

```text
fn update_side(side, node, sb_reach, bb_reach) -> ValueVectorForSide
```

At a decision node:

- if `node.player == side`, compute all action child values, regret deltas, and
  strategy sum for each private combo;
- if `node.player != side`, propagate opponent reach through each action by the
  opponent strategy and sum returned values.

At a chance node:

- traverse only representative chance children;
- multiply by multiplicity;
- apply combo permutation maps when moving between parent and representative
  coordinates;
- never treat chance as a strategy action.

At a terminal:

- fold: direct vector value from pot/commits and opponent reach mass;
- all-in before river: chance-integrated all-in CFV using exact board runouts;
- postflop showdown/all-in: current terminal CFV operator.

The hot path must avoid allocating child value vectors per action when action
count is small. Use worker-local scratch:

```text
Scratch
  action_values: [Vec<f32>; MAX_ACTIONS]
  child_reach: Vec<f32>
  terminal_values: Vec<f32>
  chance_accumulator: Vec<f32>
```

## Parallelism

The design should be parallel from the start.

Preferred split:

- side update pass is split by updated private combo blocks or by top-level
  subtree chunks;
- each worker writes disjoint regret/strategy slots;
- terminal CFV uses per-worker prepared-board scratch;
- no global result `Vec<TaskResult>` for hot traversal;
- no mutex in the inner terminal/decision loop.

Safe Rust is preferred, but the solver may use small, justified unsafe wrappers
only around disjoint row mutation if profiling proves the safe split cannot
express the layout. Any unsafe block needs:

- a comment stating the aliasing invariant;
- tests comparing single-thread and multi-thread results;
- Miri-compatible small tests where feasible.

## Terminal Values

### Postflop Terminal CFV

Use the known exact formula:

```text
value_h = pot_share(h, villain_reach, board)
        - invested_h * villain_live_reach_sum
```

The `pot_share` term is computed by:

- final-board hand strength order;
- prefix sums of opponent reach by strength group;
- `52` card blocker sums;
- exact tie handling.

Do not reintroduce:

- pair-quadratic terminal loops for full range;
- full card-prefix tables that were measured slower;
- unbounded board caches.

### Preflop All-In CFV

An all-in before the river is just a terminal chance problem:

```text
EV_h = equity_hv * win_utility
     + tie_hv    * tie_utility
     + loss_hv   * loss_utility
```

For HU ICM, map chip stacks through the affine HU ICM function. For chip EV,
use chip delta.

Implementation choices:

1. Small smoke:
   - cache exact combo-pair all-in equities by suit-isomorphic key.
2. Production:
   - build a vector all-in CFV operator that consumes opponent reach and public
     board prefix, using the same terminal CFV style as postflop.

The pair-cache smoke is acceptable only for tiny tests. It must have a hard
memory limit.

## Exploitability

Exploitability must be implemented before trusting strategy output.

For two-player zero-sum chip EV:

```text
exploitability = (BR_SB(average_BB) + BR_BB(average_SB)) / 2
```

Report:

- `bb/hand`;
- `bb/100`;
- profile zero-sum delta;
- root strategy EV;
- BR values by side.

For HU ICM, exploitability in payout units should also be convertible back to
blind units through the HU affine slope when useful. The first reports can
remain chip EV until the chip path is stable.

Best response traversal must use the same public chance representatives and
permutation maps as CFR. A separate concrete enumerator is useful only for
small regression tests.

## Planner Before Solver

The first command should be a planner, not a solver:

```text
pokedr-cli plan-full-game --config docs/...
```

It must print:

- preflop public nodes and decisions;
- number of called-preflop boundary groups;
- representative flop classes per boundary;
- representative postflop nodes/decisions/chances/terminals;
- action slots;
- regret memory;
- strategy memory;
- compact strategy estimate;
- terminal CFV calls per iteration;
- exact all-in terminal work;
- estimated per-iteration wall time from current benchmark constants;
- expected convergence time to `1bb/100` using last known postflop curves.

The command should refuse obviously impossible configs unless `--force` is
given.

Current implementation status:

- `pokedr-cli plan-full-game` exists.
- It builds the HU preflop public action tree, groups equal postflop boundary
  pot/stack states, surveys full-deck flop isomorphism, and estimates each
  boundary's representative postflop tree using the current postflop builder.
- It reports both per-boundary and total representative subgame action slots,
  storage, and terminal work.
- It does not solve CFR yet.
- The current estimate deliberately counts each public preflop boundary path as
  a distinct strategy state even if the pot/stack tuple is equal, because the
  public history and reached ranges can differ.

## Implementation Order

### Phase 1: Static Planning

1. Add `full_game` module with config structs.
2. Add preflop action template and legality tests.
3. Add full-game planner that combines preflop boundary groups with
   full-deck flop isomorphism.
4. Print memory/work estimates.
5. Add tests:
   - no illegal min-raise;
   - no always-on all-in;
   - rank-class ranges preserve full suit permutations;
   - full/full flops collapse to `1755` classes.

### Phase 2: Tiny Full-Game Smoke

1. Build a unified public tree for tiny ranges such as `AA` vs `KK`.
2. Use fixed 1326 combo rows first for simplicity.
3. Implement serial alternating CFR over the unified tree.
4. Use exact terminal CFV where possible and hard-capped pair-cache only for
   preflop all-in smoke.
5. Tests:
   - profile zero-sum;
   - single-thread deterministic;
   - average strategy rows normalize;
   - exploitability decreases on a tiny tree.

### Phase 3: Production Traversal Shape

1. Replace pair all-in smoke with vector all-in CFV.
2. Add worker-local scratch and parallel update.
3. Compact action rows and remove per-call allocations.
4. Integrate postflop terminal CFV directly into traversal.
5. Add BR/exploitability over average strategy.
6. Benchmark against `NodeLocalCfrSolver` on equivalent fixed-flop subtrees.

### Phase 4: Memory Optimization

1. Add compact live-combo rows for postflop nodes.
2. Add `f16`/fixed-point strategy sums behind validation.
3. Add save/load format for solved full-game tree and viewer consumption.
4. Add optional street rebuild only if resident memory is the blocker. This
   must preserve full-game CFR semantics and cannot become unsafe resolving.

## Correctness Invariants

These checks should be automated:

- no user-facing exact combo ranges;
- all legal actions preserve chip accounting;
- terminal fold utility is zero-sum;
- terminal showdown/all-in utility is zero-sum in chip EV;
- HU ICM utility matches affine chip EV scaling;
- strategy rows sum to 1 over legal actions for live private combos;
- chance multiplicity sums equal concrete card counts;
- chance representative permutations preserve private combo values;
- profile EV is zero-sum;
- best response exploitability is non-negative;
- exact concrete tiny-tree traversal matches isomorphic traversal.

## Performance Invariants

These should be logged in every benchmark:

- allocations per iteration;
- terminal CFV calls and time;
- decision traversal time;
- chance traversal time;
- regret/strategy update time;
- memory high-water estimate;
- thread count and work distribution;
- exploitability improvement per second.

Do not accept a change that improves one microbenchmark while making
full-game `bb/100` per second worse.

## Expected Risks

1. Full resident postflop state may still be too large for wide ranges.
2. Terminal CFV will dominate unless all-in and postflop terminals are vector
   operators, not private-pair loops.
3. Chance isomorphism can become invalid if ranges become suit-asymmetric.
4. Safe Rust may need intrusive data layout changes to express disjoint
   parallel writes.
5. Exploitability calculation may be more expensive than solve iteration; use
   intervals, but keep exact checks for validation.
6. HU Spin ICM is easy; 3+ player ICM is not affine and should not be bolted
   onto this path.

## References

- [Preflop Solver Survey](preflop-solver-survey.md)
- [Pio-Inspired Postflop Solver Plan](pio-inspired-solver-plan.md)
- [Terminal CFV Batched Matvec Plan](terminal-cfv-batched-matvec.md)
- [CFR Optimization Survey](cfr-optimization-survey.md)
- Oskari Tammelin, "Solving Large Imperfect Information Games Using CFR+",
  https://arxiv.org/abs/1407.5042
- Noam Brown and Tuomas Sandholm, "Safe and Nested Subgame Solving for
  Imperfect-Information Games", https://arxiv.org/abs/1705.02955
- Noam Brown and Tuomas Sandholm, "Solving Imperfect-Information Games via
  Discounted Regret Minimization", https://arxiv.org/abs/1809.04040
- Matej Moravcik et al., "DeepStack: Expert-Level Artificial Intelligence in
  No-Limit Poker", https://arxiv.org/abs/1701.01724
