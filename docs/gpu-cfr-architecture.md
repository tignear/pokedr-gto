# GPU CFR Architecture Notes

This document records the current architectural gap between the public-tree GPU
path and the shape we need for a materially faster solver.

## Current Iteration Shape

For each iteration, the current public-tree GPU path does:

1. propagate reach through the public tree;
2. evaluate terminal values into per-node private combo values;
3. back up child values to parent node values;
4. materialize output tables:
   - `action_values[I, a]`;
   - `reach_weights[I]`;
   - `strategy_weights[I]`;
5. run CFR update from those tables.

Let `I = (public infoset, private combo)` and `A(I)` be legal actions. The
update is:

```text
sigma(I,a) = regret_match(R[I,*], a)
v(I)       = sum_a sigma(I,a) * q(I,a)
R'[I,a]    = discount(R[I,a]) + reach_weight(I) * (q(I,a) - v(I))
S'[I,a]    = avg_discount(S[I,a]) + strategy_weight(I) * sigma(I,a)
```

The GPU currently writes `q(I,a)`, `reach_weight(I)`, and `strategy_weight(I)`
to global buffers and then reads them back in the CFR update kernel.

## Architectural Problem

The expensive part is not just a slow shader. The problem shape is wrong:

```text
child_values -> action_values buffer -> CFR update
reach state   -> reach_weights buffer -> CFR update
range mass    -> strategy_weights buffer -> CFR update
```

That creates large global-memory traffic every iteration for values that are
consumed immediately. A better GPU shape is:

```text
child_values + reach state + range mass + regrets -> regrets/strategy_sum
```

In other words, the decision-node output materialization and CFR update should
be fused at the decision node/private combo grain wherever the node's child
values are available in the same kernel.

## Why This Is Not a One-Line Fusion

The current value buffers are tiled by layer:

```text
layer tile -> hero_values[node_in_tile, combo]
layer tile -> villain_values[node_in_tile, combo]
```

A decision node can have child edges that land in a different child tile. The
current `action_edge_tile` handles this by iterating edge tiles and scattering
each action value into the global `action_values` table. That table is acting
as a gather surface across child tiles.

So the real blocker is:

```text
fused_update(I) needs all q(I,a), but q(I,a) may live in multiple child tiles.
```

Any correct fused design has to solve one of these:

1. guarantee that all children of a decision node are in the same child value
   tile;
2. add a compact per-decision-node child locator table so one kernel can gather
   all child values directly;
3. change value storage from per-tile buffers to a bindable paged/global layout;
4. keep the `action_values` gather surface, but fuse `denominator` and
   `strategy_weights` into the action pass to remove at least one intermediate
   pass.

Option 2 is the most promising next implementation target. It keeps the current
memory limits and turns the per-edge scatter into a per-decision/combo gather.

## Target Kernel Shape

A future `decision_update_tile` should run one invocation per
`(decision node, private combo)`:

```text
input:
  node descriptor
  child locator[action] = (child tile, child slot, action)
  hero/villain child value buffers for referenced child tiles
  hero/villain parent reach
  public range aggregate or compact blocker aggregate
  regrets / prediction

compute:
  q[a] for all legal actions
  reach_weight(I)
  strategy_weight(I)
  sigma[a]
  v = dot(sigma, q)

write:
  regrets[I,a]
  prediction[I,a]
  strategy_sum[I,a]
```

This removes the full-size `action_values` table from the training loop. Metrics
can still materialize it on demand in a slower diagnostic path.

## Practical Next Step

Before changing storage, add output-stage profiling to split the currently
combined `cfv_decision_denominator` phase into:

```text
decision_aggregate
decision_denominator
strategy_aggregate
action_edge
cfr_update
```

If `action_edge + cfr_update` is a large fraction, implement option 2. If
`decision_denominator` dominates, first fuse/replace denominator and strategy
weight generation.
