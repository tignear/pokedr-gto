# Postflop GPU CFR Plan

This note fixes the data dependencies before more GPU solver work. The main
trap is confusing reusable card evaluation with path-dependent range equity.

## Objects

Let:

- `c_h, c_v` be hero and villain private combos.
- `b` be a final five-card board.
- `z` be a terminal public history.
- `n` be a public decision node.
- `a` be an action at `n`.
- `p(z)` be the pot at terminal `z`.
- `i_h(z)` be hero's total investment at terminal `z`.
- `pi_h(n, c_h)` and `pi_v(n, c_v)` be private reach probabilities at public
  node `n`.
- `pi_c(n)` be public chance reach at node `n`.
- `w_v(c_v)` be the initial villain range weight.

The reusable showdown table is not range equity. It is only hand strength:

```text
S[b, c] = evaluate_7_cards(c, b)
```

This is path-independent. For a fixed final board and combo, the hand strength
does not depend on betting history, pot size, or range.

Pair showdown payoff at a terminal is:

```text
W[b, c_h, c_v] =
  1.0 if S[b, c_h] > S[b, c_v]
  0.5 if S[b, c_h] = S[b, c_v]
  0.0 otherwise

U(z, c_h, c_v) = p(z) * W[b(z), c_h, c_v] - i_h(z)
```

`S` is reusable. `U` is only partly reusable: the comparison part is reusable,
but `p(z)` and `i_h(z)` are path-dependent.

## Path-Dependent Quantities

The opponent range at a node is not a static preflop/flop range. It is the
initial range multiplied by strategy reach along the path:

```text
R_v(n, c_v) = w_v(c_v) * pi_v(n, c_v)
R_h(n, c_h) = pi_h(n, c_h)
```

Therefore range equity at a node/action is path-dependent:

```text
EQ_h(n, a, c_h) =
  sum_{c_v legal} R_v(n, c_v) * V_h(child(n, a), c_h, c_v)
  / sum_{c_v legal} R_v(n, c_v)
```

This cannot be cached by board alone. It changes when earlier actions change
villain reach.

For villain decision nodes, the sign flips because `V_h` is hero utility:

```text
Q_v(n, a, c_v) =
  - sum_{c_h legal} R_h(n, c_h) * V_h(child(n, a), c_h, c_v)
    / sum_{c_h legal} R_h(n, c_h)
```

## CFR Accumulators

For an acting player `P` at public infoset `I(n)`, private combo `c`, and action
`a`, action value is the opponent-reach weighted mean:

Hero node:

```text
A[I, h, a] += sum_v pi_c(n) * R_v(n, v) * V_h(child(n, a), h, v)
M[I, h, a] += sum_v pi_c(n) * R_v(n, v)
```

Villain node:

```text
A[I, v, a] += sum_h pi_c(n) * R_h(n, h) * (-V_h(child(n, a), h, v))
M[I, v, a] += sum_h pi_c(n) * R_h(n, h)
```

After reducing all opponent combos:

```text
action_value[I, c, a] = A[I, c, a] / M[I, c, a]
```

Reach and average-strategy weights are:

Hero node:

```text
reach_weight[I, h]    += sum_v pi_c(n) * R_v(n, v)
strategy_weight[I, h] += pi_c(n) * pi_h(n, h)
```

Villain node:

```text
reach_weight[I, v]    += sum_h pi_c(n) * R_h(n, h)
strategy_weight[I, v] += pi_c(n) * R_v(n, v)
```

Then regret matching uses the existing dense CFR update:

```text
regret[I,c,a] += reach_weight[I,c] *
  (action_value[I,c,a] - sum_a sigma[I,c,a] * action_value[I,c,a])
```

## GPU Work Split

The correct reusable GPU table is:

```text
strength[final_board_index][combo] -> u32
```

Memory scale:

```text
O(|B_final| * |C|)
```

Avoid this table as a resident/cache target:

```text
equity_matrix[final_board_index][hero_combo][villain_combo] -> f32
```

Memory scale:

```text
O(|B_final| * |C|^2)
```

For `|C| = 1326` and `|B_final| = 1225`, `strength` is a few MB while the full
matrix is multiple GB.

The intended GPU iteration is:

1. Build or reuse `strength[b, c]` for final boards reachable from the public
   tree.
2. For combo-pair chunks, run public-tree forward reach:

```text
(pi_h, pi_v, pi_c)[node, pair]
```

3. Run public-tree backward value. At showdown terminals compute:

```text
W = compare(strength[b, c_h], strength[b, c_v])
V_h = p(z) * W - i_h(z)
```

4. Reduce chunk contributions into `A`, `M`, `reach_weight`, and
   `strategy_weight`.
5. Apply dense CFR regret/strategy update on GPU.

The chunk boundary may split combo pairs, but it must not change math. The
chunk outputs are additive numerators and denominators; division happens only
after all chunks are reduced.

## Non-Goals And Known Bad Paths

- Do not CPU-traverse `(hero_combo, villain_combo, public_tree)` for production
  solving.
- Do not cache `board -> 1326 x 1326 equity matrix` for every terminal board.
- Do not silently fall back to CPU solving when the GPU public-tree path fails.
- Do not compare full production trees against a CPU solver as the main
  correctness test; use small deterministic fixtures for equality and invariants
  for large runs.

## Implemented Direction

The implementation has moved away from pair chunks. It now uses the ordinary
counterfactual value vector shape:

```text
reach_h[n, h]
reach_v[n, v]
value[n, c] for one player at a time
```

The reusable showdown object is still the final-board strength table:

```text
strengths: array<u32> indexed by board_index * combo_count + combo
```

The value shader computes terminal payoff by comparing two strength lookups and
reducing over the opponent reach vector. It computes hero values, aggregates
hero infosets, then reuses the same value buffer for villain values and
aggregates villain infosets. This avoids keeping both `value_h` and `value_v`
resident at the same time.

## First Measurement

The first implementation stores strengths as exact `f32` packed hand values.
The packed hand strength is below the integer precision limit of `f32`, so
comparisons remain exact.

Observed fixed flop `As7h2c`, one iteration:

```text
depth=2 before strength table: 8.61s
depth=2 after strength table:  2.68s
depth=2 CFV vector path:        5.25s
```

For `depth=5`, strength-table construction succeeds and reports:

```text
nodes=15499
final_boards=3528
pair_chunks=8141
```

This means the showdown matrix bottleneck is removed, but the next bottleneck is
still too large:

```text
O(pair_count * node_count)
```

The CFV vector implementation removes this pair-count factor and depth 5 now
completes:

```text
depth=5 CFV vector path:
nodes=15499
final_boards=3528
public_infosets=447
private_infosets=1185444
elapsed=40.13s
```

This is a correctness-oriented full-range GPU path, not yet a fast solver. The
current bottlenecks are:

- value dispatch is split by public node to avoid D3D12 TDR/device loss;
- `reach_h` and `reach_v` are still resident for every `(node, combo)`;
- terminal value reduction still loops opponent combos inside each terminal
  node shader.

The next speed target is to batch value work by node class and street/window so
the shader does more useful work per dispatch without triggering TDR, while
keeping resident memory below DZN/wgpu buffer limits.

## Ordinary Solver Shape

The next target is the usual counterfactual value vector shape:

```text
reach_h[n, h]
reach_v[n, v]
value_h[n, h]
value_v[n, v]
```

instead of:

```text
reach[n, h, v]
value[n, h, v]
```

This changes the main resident memory from:

```text
O(|N| * |C|^2)
```

to:

```text
O(|N| * |C|)
```

Chance nodes still need care, but they do not force `pair x node` storage. For a
public chance card `r` at node `n`, every legal private pair has the same chance
denominator:

```text
d(n) = |deck \ board(n)| - 4
```

because a non-colliding private pair always removes four private cards from the
remaining public deck. Therefore public chance reach can stay scalar:

```text
pi_c(child_r) = pi_c(n) / d(n)
```

and private reach vectors only need card-removal masks:

```text
reach_h[child_r, h] = reach_h[n, h] * 1[card r not in h]
reach_v[child_r, v] = reach_v[n, v] * 1[card r not in v]
```

Terminal vector values are then opponent-range convolutions, not stored pair
values:

Hero terminal vector:

```text
value_h[z, h] =
  1[reach_h[z,h] > 0] * pi_c(z) *
  sum_v 1[not collide(h,v)] * reach_v[z,v] *
    (p(z) * W[b(z), h, v] - i_h(z))
```

Villain terminal vector:

```text
value_v[z, v] =
  1[reach_v[z,v] > 0] * pi_c(z) *
  sum_h 1[not collide(h,v)] * reach_h[z,h] *
    (-(p(z) * W[b(z), h, v] - i_h(z)))
```

Decision nodes propagate vectors:

Hero decision:

```text
value_h[n,h] = sum_a sigma_h[n,h,a] * value_h[child_a,h]
value_v[n,v] = sum_a value_v[child_a,v]
```

Villain decision:

```text
value_h[n,h] = sum_a value_h[child_a,h]
value_v[n,v] = sum_a sigma_v[n,v,a] * value_v[child_a,v]
```

Chance nodes sum children:

```text
value_h[n,h] = sum_r value_h[child_r,h]
value_v[n,v] = sum_r value_v[child_r,v]
```

Regret action values at decision nodes use child vectors plus opponent reach
denominators. For hero:

```text
action_value[I,h,a] = value_h[child_a,h] /
  (pi_c(n) * sum_v 1[not collide(h,v)] * reach_v[n,v])
```

For villain:

```text
action_value[I,v,a] = value_v[child_a,v] /
  (pi_c(n) * sum_h 1[not collide(h,v)] * reach_h[n,h])
```

This is the correct escape from the current `pair x node` bottleneck. It still
has pairwise work at terminal convolutions and denominator reductions, but it no
longer multiplies every pair by every public node.
