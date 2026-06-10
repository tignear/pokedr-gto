# Full-Game Solver Concepts

This document explains the concepts needed to build a heads-up no-limit
full-game solver in `pokedr-gto`. It is deliberately explicit because several
terms sound interchangeable but imply very different implementation choices.

## Full-Game Semantics

Full-game semantics means the solved strategy belongs to one game from the root
to terminal nodes.

For HU Hold'em this means:

```text
preflop decisions
  fold terminals
  all-in showdown terminals
  called non-all-in branches
    flop chance
    postflop betting
    turn chance
    postflop betting
    river chance
    postflop betting
    showdown/fold terminals
```

The key property is that every postflop reach is induced by the preflop
strategy in the same CFR run. Preflop does not see a fixed scalar "flop value."
It sees counterfactual values produced by the postflop game under the current
strategy profile.

Why it matters:

- Limp, raise, call, and jam frequencies depend on postflop consequences.
- Postflop ranges are not externally supplied after preflop; they are reach
  weights created by the earlier actions.
- Exploitability can be measured at the root of the whole game.

What it does not require:

- It does not require allocating one giant concrete tree in RAM.
- It does not require solving every chunk simultaneously.
- It does not forbid streaming state from disk.

## Physical Representation

Physical representation is how the game is stored and traversed.

The same full game can be represented in a bad way:

```text
one independent full postflop tree per concrete preflop line and concrete board
```

or in a better way:

```text
preflop trunk
shared postflop action skeletons
public-card isomorphism representatives
combo permutation maps
streamed regret/strategy chunks
shared terminal tables
```

The first version is conceptually simple but quickly impossible. The second
version is harder to implement but keeps the same game semantics while avoiding
duplicate static structure.

## Public State

Public state is the part of the game visible to both players.

For preflop:

- blinds and antes;
- pot;
- stack behind;
- each player's committed amount;
- player to act;
- previous raise size;
- number of raises;
- line shape such as limp, raise, call, all-in.

For postflop:

- board cards;
- street;
- pot;
- stack behind;
- each player's committed amount;
- player to act;
- to-call amount;
- current betting round state;
- whether donk betting is allowed.

Public state determines legal public actions, but it is not enough to decide
strategy. Strategy also depends on private hand and reach weights.

## Private Combos

Private combos are concrete two-card hands such as `AhAd`, not 169 hand
classes.

The exact solver needs concrete combos because blockers matter:

- `AhAd` and `AcAs` can have different legality on a board.
- A combo can block opponent value or bluff combos.
- Turn/river cards can make suits strategically different.

169 classes are still useful for:

- UI matrices;
- aggregated strategy display;
- hand-detail grouping;
- rough estimates.

They are not enough for exact CFR state.

## Range Weight

A range weight is the probability mass assigned to a private combo.

Initial range weights come from user input such as:

```text
TT+,A2s+,KTs+,QJs,ATo+
```

During CFR, reach weights are updated by strategy:

```text
reach_after_action(combo)
  = reach_before(combo) * strategy(info_set, combo, action)
```

This is why a fixed postflop range table is not enough for full-game solving.
The postflop range after a preflop branch is created by the preflop strategy.

## Reach

Reach is the probability that a private combo arrives at a node under the
current strategy.

There are two player reaches:

```text
r_oop[h]  for each OOP combo h
r_ip[v]   for each IP combo v
```

At an OOP decision, OOP's strategy changes `r_oop`; IP reach is unchanged. At
an IP decision, IP's strategy changes `r_ip`; OOP reach is unchanged.

Chance nodes do not choose a strategy. They transform board state and private
combo legality, and apply chance probabilities during value backup.

## Information Set

An information set groups states that a player cannot distinguish.

In this solver, a practical postflop information set is roughly:

```text
public node + acting player + private combo
```

The acting player knows:

- public betting history;
- public board;
- their own private combo.

They do not know the opponent's combo. Therefore the value of an action is
computed against the opponent's reach distribution, not against one known hand.

## Counterfactual Value

Counterfactual value, or CFV, is the value of an action or node from the
perspective of one player, weighted by the opponent's reach and chance, while
treating the updating player's own reach counterfactually.

For a terminal showdown board `b`, OOP's value for combo `h` has the form:

```text
CFV_oop[h]
  = sum over legal IP combos v:
      r_ip[v] * payoff_oop(h, v, b)
```

IP's value is symmetric:

```text
CFV_ip[v]
  = sum over legal OOP combos h:
      r_oop[h] * payoff_ip(h, v, b)
```

This is why terminal evaluation is expensive. It is a range-vs-range operation
with blockers.

## Terminal CFV

Terminal CFV evaluates fold, showdown, or all-in terminal nodes.

Fold terminal:

- no hand evaluator is needed;
- payoff is determined by pot and committed chips;
- values still depend on which player folded.

River showdown terminal:

- board is complete;
- compare every legal private combo against opponent reach;
- apply blocker correction.

Flop/turn all-in terminal:

- betting is over before the board is complete;
- evaluate all remaining runouts, or an exact isomorphic equivalent;
- average over legal runouts.

Terminal CFV is usually one of the heaviest parts of exact solving.

## Action Skeleton

The action skeleton is the public betting structure without concrete chance
cards.

Example:

```text
check
bet 60%
bet geometric
all-in
fold
call
raise 2.5x
```

A good solver should not create an independent action skeleton for every turn
or river card if the action options are structurally the same. It should share
the skeleton and attach chance-specific metadata separately.

This is one reason `postflop-solver` is compact: its action tree treats
turn/river chance as the same action at the skeleton level.

## Action Abstraction

No-limit Hold'em has a continuous action space. A solver must choose a finite
set of bet and raise sizes.

Example abstraction:

```text
first bets: 60% pot, geometric, all-in
raises: 2.5x previous bet
river donk: 50% pot
```

Action abstraction controls both quality and size:

- too small: strategy misses important actions;
- too large: tree becomes impossible to solve;
- badly shaped: river raise chains explode without adding useful strategy.

The abstraction is part of the solver design, not just a UI option.

## Close-Size Merging

Close-size merging removes nearly identical bet sizes.

If several generated actions are strategically close, such as:

```text
bet 13.2bb
bet 13.7bb
all-in 14.0bb
```

keeping all of them creates extra states with little strategic value.

Pio-style merging compares bet sizes by ratio and removes close smaller sizes
after sorting from large to small. This reduces river state without changing the
overall action abstraction intent.

## Force All-In

Force all-in converts a non-all-in bet or raise into all-in when the remaining
stack after a call would be too small.

Example:

```text
pot = 10bb
effective stack = 14bb
candidate raise leaves 1bb behind after call
```

That non-all-in branch often creates useless extra future actions. Force all-in
collapses it into the all-in action.

This is especially important on the river because there is no future chance
card to absorb small residual stacks. Without this rule, river raise chains can
become enormous.

## Add All-In

Add all-in means all-in is included as an option when stack size is close enough
to a natural bet or raise size.

This is different from force all-in:

- add all-in adds another action;
- force all-in replaces a near-all-in action.

Used carefully, add all-in improves action coverage. Used carelessly, it creates
too many all-in branches.

## Public-Card Isomorphism

Public-card isomorphism removes equivalent chance cards under suit
permutations.

Example:

On a monotone flop, the three non-board suits can be equivalent for some future
cards. Instead of evaluating all concrete cards separately, evaluate one
representative and map private combos through a suit permutation.

The solver must carry:

- representative card;
- concrete cards represented by it;
- multiplicity or concrete event count;
- combo permutation maps.

This is exact only when ranges and state are suit-symmetric enough for the
chosen permutation group. User-facing exact suit ranges can destroy this
benefit.

## Combo Permutation

When a public-card representative stands in for an isomorphic concrete card,
private combos must be permuted too.

If hearts and diamonds are swapped, then:

```text
AhAd -> AdAh
KhQh -> KdQd
```

The value vector from the representative child must be mapped back to the
parent's combo coordinate system.

In `NodeLocalCfrSolver`, this is represented by permutation codes and applied
during chance backup.

## Schematic Storage

Schematic storage means static structure is stored once and reused.

Static data:

- action skeleton;
- public chance representative metadata;
- combo permutation maps;
- terminal board tables;
- hand strength ordering;
- legal combo masks.

Dynamic data:

- regrets;
- strategy sums;
- current reach;
- temporary CFV buffers.

The dynamic data is what grows with game state. Static data should not be
duplicated for every preflop boundary and board class.

## Regret

Regret is the cumulative advantage of having chosen an action compared with the
current strategy.

For action `a`:

```text
regret[a] += action_value[a] - node_value
```

CFR updates future strategy using regret matching:

```text
strategy[a] = positive_regret[a] / sum_positive_regrets
```

For each decision node, regret is stored per acting private combo and action:

```text
regret[combo, action]
```

This is why memory grows with:

```text
decision nodes * acting combos * actions
```

## Strategy Sum

The average strategy is usually computed from accumulated strategy sums.

At each iteration:

```text
strategy_sum[combo, action] += reach_weight * current_strategy[combo, action]
```

The final output strategy is normalized from `strategy_sum`.

Some CFR variants discount or reset strategy sums. `postflop-solver`'s public
docs mention resetting cumulative strategy at powers of four for its DCFR
variant.

## Alternating CFR

Alternating CFR updates one player's regrets at a time.

One iteration can be viewed as:

```text
update OOP regrets against current IP strategy
update IP regrets against current OOP strategy
accumulate average strategy
```

Alternating updates are common in practical poker solvers and should not be
treated as interchangeable with simultaneous updates.

## DCFR / DCFR+

Discounted CFR changes how older regrets and strategy sums are weighted.

The idea is that early iterations are often noisy. Discounting can make the
solver adapt faster and reduce the number of iterations needed.

DCFR+ combines discounting with CFR+-style positive regret treatment. Exact
formula choices matter and should be benchmarked on exploitability, not chosen
by intuition.

## Exploitability

Exploitability measures how much a best response can win against the strategy.

For a two-player zero-sum game:

```text
exploitability
  = (BR_oop_vs_ip_strategy - profile_value_oop)
  + (BR_ip_vs_oop_strategy - profile_value_ip)
```

Reported in poker as:

```text
bb/100
```

A target like `1bb/100` means the strategy is exploitable by about one big
blind per 100 hands.

Exploitability is the correct stopping target. Iteration count is only a proxy.

## Zero-Sum Invariant

In chip EV heads-up poker:

```text
V_oop(strategy) + V_ip(strategy) = 0
```

If profile values are not approximately zero-sum, either:

- payoffs are signed incorrectly;
- range weights are normalized inconsistently;
- chance probabilities are wrong;
- fold/showdown values are using different baselines;
- evaluation is mixing different games.

Zero-sum checks are basic correctness tests.

## Safe Subgame Solving

Safe subgame solving is used when a local subgame is solved separately from a
blueprint.

The problem:

```text
Solving an imperfect-information subgame in isolation can increase full-game
exploitability.
```

Safe solving adds constraints or gadget games so the refined subgame cannot
make the full strategy worse against a best response.

This is relevant if full-game CFR is too large and we want to solve postflop
chunks separately from preflop. Without safety, a boundary value can be wrong.

## Boundary Value

A boundary value is a value supplied at the edge of a depth-limited solve.

Examples:

- scalar equity table;
- counterfactual value vector;
- neural value function;
- safe-solving opponent CFV bound.

A scalar equity boundary is usually not enough for no-limit poker because
action-dependent ranges matter. A correct boundary must preserve enough
counterfactual information for both players.

## Chunk Streaming

Chunk streaming means only part of dynamic solver state is resident at once.

Example chunk key:

```text
(preflop boundary group, flop isomorphism class)
```

For each chunk:

1. load regret/strategy state;
2. run traversal/update for that chunk;
3. write updated regret/strategy state;
4. keep static skeleton and terminal tables shared.

Streaming solves memory pressure but not compute pressure. If each chunk still
does enormous terminal CFV work, disk streaming alone will not make the solver
fast.

## Disk State

Disk state should contain dynamic arrays, not rebuildable static data.

Good disk state:

- regret chunks;
- strategy-sum chunks;
- checkpoint metadata;
- solve progress;
- config hash.

Bad disk state:

- duplicated action skeletons;
- duplicated terminal tables;
- duplicated combo permutation maps;
- viewer-only summaries that can be regenerated.

The disk layout must be append-safe or checkpoint-safe because long solves will
be interrupted.

## River Explosion

River explosion is the growth of decision nodes and action slots on the river.

Why river dominates:

- no future chance node separates actions;
- every bet/raise/call/fold line stays on the same street;
- small pot with high remaining stack allows many non-all-in raises;
- each public decision stores per-combo action regrets;
- full-game solving multiplies this by many preflop boundaries and flop classes.

The main controls are:

- tighter river raise abstraction;
- close-size merging;
- force all-in;
- max raise count;
- removing strategically useless donk branches;
- pruning actions with strongly negative regret when exact bounds allow it.

## Exactness

Exactness means solving the specified abstract game exactly enough, not solving
real no-limit Hold'em with every possible bet size.

Exact:

- exact private combos;
- exact public-card chance within the abstraction;
- exact terminal showdown/fold payoff;
- exact CFR updates for the chosen action abstraction.

Not exact relative to real poker:

- finite bet sizes;
- merged close sizings;
- omitted rare branches;
- ICM approximations;
- depth-limited value functions.

This distinction matters. A solver can be exact for its abstract game while the
abstract game is still strategically incomplete.

## Practical Build Order

For `pokedr-gto`, the clean implementation order is:

1. Keep node-local postflop CFR as the exact postflop engine.
2. Build preflop trunk and boundary groups.
3. Reuse postflop action skeletons and chance isomorphism metadata.
4. Add a full-game traversal that returns postflop CFVs to preflop.
5. Add small full-game zero-sum and exploitability tests.
6. Add chunked regret/strategy storage only after the in-memory small version is
   correct.
7. Add disk streaming for large boundary/flop chunks.
8. Add diagnostics for river node/action-slot explosion.
9. Tune action abstraction and DCFR schedules against `bb/100` convergence.
