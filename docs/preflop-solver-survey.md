# Preflop Solver Survey

This note summarizes how practical no-limit Hold'em preflop solvers are
implemented, based on public literature, commercial-solver documentation, OSS
postflop solver structure previously inspected for this project, and the
current `pokedr-gto` codebase.

The purpose is not to document a finished preflop solver in this repository.
The active implementation surface is postflop public-tree CFR, range parsing,
exact suit isomorphism, and terminal CFV. This document records what the next
preflop implementation should look like.

## Short Answer

A credible preflop solver is a full-game or safely depth-limited
extensive-form solver:

```text
preflop public trunk
  private ranges over all 1326 combos per player
  legal/action-abstracted preflop betting
  fold/all-in terminals
  called non-all-in branches
    -> flop chance over suit-isomorphic representative flops
    -> postflop public tree or safe boundary value function
```

The solver still uses CFR-family updates, but the expensive part is not
preflop action count. The expensive part is that every non-all-in preflop line
creates a postflop distribution whose CFVs depend on both players' path reaches.
Treating those leaves as a scalar equity or fixed precomputed table is not a
correct full-game solution unless the boundary is constructed with safe-solving
constraints or is part of the same CFR game.

## What Public Systems Do

### Libratus-Style Systems

Libratus is the clearest public model for a high-quality no-limit poker agent.
The public descriptions are consistent with a three-part architecture:

- compute an offline blueprint over an abstracted full game;
- solve/re-solve subgames in real time with safe nested subgame solving;
- inspect opponent actions that expose abstraction holes and add missing
  actions back into the strategy.

Brown and Sandholm's safe/nested subgame solving paper states the central
constraint: an imperfect-information subgame cannot generally be solved in
isolation because the optimal strategy can depend on unreached parts of the
game. Their method gives a way to refine a blueprint and respond to off-tree
actions without unsafe action translation.

Implication for preflop: the preflop strategy is not just a chart. It is the
root of a game whose postflop leaves must be tied to the trunk strategy. If we
split preflop and postflop, the boundary must carry counterfactual values and
constraints, not just average equity.

### DeepStack-Style Systems

DeepStack does not solve the whole game to the end at every decision. It uses
continual re-solving with a learned value function for depth-limited leaves.
That makes it practical in real time, but it shifts work into value-network
training and validation.

Implication for this project: a neural/value-function boundary is a valid
future direction, but it is a different correctness story. The exact baseline
should first be a full-game or safe-boundary CFR formulation, then approximate
leaf values can be compared against that baseline.

### Pluribus / Multiplayer Systems

Pluribus adds a useful warning for multiplayer no-limit poker: it does not have
the same clean zero-sum equilibrium guarantees as heads-up. It uses an offline
blueprint and limited-depth search in play, but the theoretical guarantees are
weaker than heads-up zero-sum solving.

Implication: heads-up payout-adjusted solving can be made much cleaner than 3+
player ICM. Multiway preflop ICM solving should be treated as a separate,
harder problem.

## What Commercial Solver Docs Imply

PioSOLVER's public docs expose several relevant implementation constraints:

- inputs are ranges, pot/stack, bet sizes, raise sizes, donk settings, target
  accuracy/solve time, and threads;
- solution quality is exploitability per hand, with `1bb/100` described as a
  normal practical target;
- ranges are not just combo counts: real hand frequency depends on blockers and
  opponent range matchups;
- EV is computed over explicit matchups and betting sequences;
- exploitability is based on maximum-exploitability-strategy EV versus profile
  EV.

For preflop this means:

- range weights must stay as reach weights through the tree;
- chance-card frequencies are matchup-weighted, not just board-count weighted;
- an action line's range is prior reach multiplied by strategy at every earlier
  decision;
- exact suit combo inputs are a bad user-facing abstraction because they
  destroy suit symmetry and make isomorphism less effective. Rank-class ranges
  are the right default input model.

## CFR Variant Choice

The literature does not point to a separate "preflop-only" algorithm. It points
to the same CFR family, with implementation details doing most of the work:

- CFR+ remains the baseline practical solver family.
- DCFR variants discount old regrets/strategy weight and can outperform CFR+
  in tested large imperfect-information games.
- Alternating updates matter for practical performance and memory shape.
- Recent parallel CFR work frames practical play as completing enough
  iterations within a time budget, with parallelism by infoset and by tree node.

Recommended default for this project:

- use alternating DCFR+/scheduled DCFR+ as the first full-game preflop solver;
- keep CFR+ as a baseline for regression tests;
- measure root exploitability in `bb/100`, not just iteration count;
- do not assume PDCFR+ is better without local curves on the same tree.

## Tree Representation

The preflop tree should be separate from the current postflop `PublicTree`
state type, but it should share lower-level concepts:

```text
PreflopState:
  button / blind roles
  street = preflop
  pot
  stacks
  commits
  last raise / min raise
  raises on preflop
  player to act

PostflopState:
  current existing PublicState
```

The important part is that both are one public game, not disconnected solves.
A non-all-in called preflop action should become:

```text
PreflopBoundary(group_id)
  -> FlopChance(isomorphic representative flops, multiplicity, combo swaps)
  -> Postflop schematic tree
```

The tree must be schematic:

- store the betting skeleton once;
- use public chance isomorphism for flops, turns, and rivers;
- keep multiplicities and private-combo permutations for skipped equivalent
  chance outcomes;
- do not allocate a separate action subtree for every concrete board unless
  explicitly dumping/debugging.

## Private-Hand Representation

Use concrete combo indices internally, not 169 hand classes, for the exact
solver.

169 classes are useful for:

- UI aggregation;
- coarse estimates;
- smoke tests;
- fast heuristic prototypes.

They are not enough for exact preflop solving because:

- blockers affect flop frequencies and terminal values;
- postflop suits matter through board texture;
- an action path can break simple rank-class symmetry through public cards and
  reaches;
- ICM/all-in utilities depend on exact matchup equity.

The user-facing range language should accept rank-class tokens such as
`TT+`, `A2s+`, `KQo`, and weights such as `88+:0.7`. It should reject exact
suit combo tokens such as `AhAd` in normal configs because they create
suit-asymmetric inputs that make exact public-card isomorphism much less
effective.

## Boundary Values

There are three viable boundary approaches.

### 1. Full-Game CFR

This is the cleanest correctness model:

```text
CFR iteration traverses preflop and postflop in one game.
At every postflop node, reaches are the actual reaches induced by preflop play.
Regrets and average strategies live in one consistent state.
```

Pros:

- no unsafe boundary assumptions;
- exploitability means what it should mean;
- preflop limp/raise/jam frequencies are solved against the actual postflop
  consequences.

Cons:

- memory and runtime are large;
- needs exact isomorphism and schematic storage;
- terminal CFV remains the bottleneck.

### 2. Safe Boundary / Safe Subgame Solving

This is the Libratus-compatible split:

```text
preflop blueprint/trunk values
  -> boundary subgame with constraints that preserve opponent safety
```

Pros:

- lets us solve pieces;
- handles off-tree or finer postflop actions;
- can be made theoretically safe if constraints are correct.

Cons:

- substantially more complex than "call postflop solver and plug in EV";
- needs opponent counterfactual-value bounds, not just hero EVs;
- easy to accidentally create a boundary that the trunk can exploit.

### 3. Approximate Leaf / Value Function

This is the DeepStack-style direction.

Pros:

- likely needed for real-time broad preflop solving;
- can make full preflop traversal feasible.

Cons:

- requires training/validation;
- exactness is gone;
- must be benchmarked against exact small/full-game solves.

For this repository, the recommended order is:

1. exact full-game smoke on small but realistic HU trees;
2. exact isomorphism and terminal CFV reuse;
3. safe boundary only after the full-game semantics are validated;
4. learned/approximate values only after exact comparisons exist.

## HU Payout / ICM Notes

Heads-up ICM has an important simplification. With two players and fixed total
chips `T`, the hero's ICM value is affine in chips:

```text
ICM_H(h, v) = P2 + (P1 - P2) * h / T
```

Therefore a heads-up postflop subgame can be solved in chip EV and then scaled
to payout dollars by `(P1 - P2) / T`. This does not hold for 3+ players because
multiway ICM is nonlinear in stacks.

Practical consequence:

- Heads-up payout-adjusted solving is a good first target when the payout model
  is affine in chips.
- 3+ player preflop ICM should not share the same implementation unless it
  treats chip EV and payout EV as different games.

## Isomorphism Requirements

Exact public-card isomorphism is not optional.

At minimum:

- full-deck flops collapse from `22100` concrete unordered flops to `1755`
  rank/suit representatives for full suit-symmetric ranges;
- paired boards need explicit coverage;
- monotone/two-tone/rainbow textures have different remaining suit classes;
- after each chance representative, private combo indices must be permuted
  into the representative coordinate system;
- multiplicity must be applied when summing chance values.

Input range restrictions matter. If a user can input `AhAd` but not all suit
equivalent `AA` combos, many suit permutations stop preserving the ranges and
isomorphism collapses. That is why exact combo tokens should stay internal-only.

## Implementation Shape To Build

### Data

```text
PreflopGame
  players: SB/BB or seat indices
  stacks, blinds, ante, payout model
  sb_range, bb_range as RangeSpec
  action template
  postflop template

UnifiedPublicTree
  preflop nodes
  flop chance representatives
  postflop schematic nodes
  chance multiplicities
  combo permutation maps

CfrState
  node-local action offsets
  regrets by update side / infoset / combo / action
  strategy sums, possibly compact
  scratch pools per worker
```

### Iteration

Use alternating side updates:

```text
for t in 1..=iterations:
  update SB over all SB private combos / reach vectors
  update BB over all BB private combos / reach vectors
  apply DCFR+/schedule discounts
  accumulate average strategy with the selected weighting schedule
```

The hot traversal should return one side's CFV vector against the opponent
reach, not allocate both players' values at every node.

### Terminal CFV

For postflop terminals:

- use the existing sorted-strength + card-blocker terminal CFV shape;
- avoid pair-quadratic private loops;
- reuse static board strength/blocker tables per representative board;
- stream bounded scratch rather than caching unbounded terminal boards.

For preflop all-in terminals:

- exact all-in equity can be cached by suit-isomorphic `(hero combo, villain
  combo)` key;
- for full-range production, prefer a range-vector all-in CFV operator over a
  private-pair loop;
- map chip results through HU ICM only if the game is heads-up.

## What Not To Build

Do not build these as the main solver:

- a push/fold-only chart solver;
- a 169-hand-class-only solver;
- a preflop CFR that replaces called postflop branches with fixed equity;
- a boundary table computed once before CFR from root ranges;
- a solver that accepts exact suit combo range input in user config;
- a full concrete board tree with no suit isomorphism;
- a pair-loop full-game solver except as a tiny correctness smoke.

These can all be useful as tests or diagnostics, but they are not the path to a
trustworthy preflop solution.

## Open Questions For This Codebase

1. Can the existing `PublicTree` become the postflop subtree under a new
   preflop trunk without duplicating node/state code?
2. Should preflop regrets live in the same `NodeLocalCfrSolver` layout, or
   should a new unified solver own both preflop and postflop nodes?
3. Can terminal CFV be reused across many boundary groups without invalid
   caching of path-dependent reaches?
4. What is the smallest HU short-stack tree that is both realistic and converges to
   `1bb/100` fast enough to iterate?
5. How much exact memory is saved if strategy sums are compacted to `f16` or
   fixed-point while regrets stay `f32`?
6. Where should safe-subgame constraints live if we later split full-game CFR
   into trunk plus postflop subgames?

## Recommended Next Steps

1. Add a preflop tree planner only. It should print public node counts,
   representative flop count, action slots, estimated regret/strategy memory,
   terminal CFV work, and impossible configs before solving.
2. Add a small HU full-game tree with rank-class ranges only and exact flop
   isomorphism.
3. Wire an alternating DCFR+ smoke solver over that unified tree using bounded
   scratch and existing terminal CFV.
4. Validate profile zero-sum, best-response exploitability, and monotonic
   exploitability reduction on tiny exact games before widening ranges.
5. Only after exact semantics pass, add safe boundary or approximate value
   layers.

## Sources

- Oskari Tammelin, "Solving Large Imperfect Information Games Using CFR+",
  https://arxiv.org/abs/1407.5042
- Matej Moravcik et al., "DeepStack: Expert-Level Artificial Intelligence in
  No-Limit Poker", https://arxiv.org/abs/1701.01724
- Noam Brown and Tuomas Sandholm, "Safe and Nested Subgame Solving for
  Imperfect-Information Games", https://arxiv.org/abs/1705.02955
- Noam Brown and Tuomas Sandholm, "Solving Imperfect-Information Games via
  Discounted Regret Minimization", https://arxiv.org/abs/1809.04040
- Noam Brown et al., "Deep Counterfactual Regret Minimization",
  https://arxiv.org/abs/1811.00164
- Christian Kroer, Gabriele Farina, and Tuomas Sandholm, "Solving Large
  Sequential Games with the Excessive Gap Technique",
  https://arxiv.org/abs/1810.03063
- Boning Li and Longbo Huang, "Real-Time Parallel Counterfactual Regret
  Minimization", https://arxiv.org/abs/2605.19928
- PioSOLVER technical details,
  https://piosolver.com/docs/technical_details/
- PioSOLVER numbers/ranges/matchups/exploitability documentation,
  https://piosolver.com/docs/viewer/numbers_in_piosolver/
- GTO Wizard, "How Solvers Work",
  https://blog.gtowizard.com/how-solvers-work/
- Carnegie Mellon University, "Carnegie Mellon Reveals Inner Workings of
  Victorious AI", https://www.cmu.edu/news/stories/archives/2017/december/ai-inner-workings.html
