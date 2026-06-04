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

The solver target is not "recursive CFR that happens to run in GPU shaders".
The target is a sequence of vector and sparse-matrix operations over a fixed
public tree. CPU code may build the sparse structure, but each iteration should
be GPU-resident linear algebra:

```text
reach propagation:   r_next = T(sigma) * r
terminal CFV:        v_z    = U_z * r_opp
chance/decision DP:  v_n    = B_n(sigma) * v_children
infoset aggregate:   A      = G * v
regret update:       R      = f(R, A, reach_weight, strategy_weight)
```

Where `T`, `U_z`, `B_n`, and `G` are implicit sparse/dense operators encoded as
GPU buffers. They do not need to be materialized as full matrices, but kernels
must use their row/column structure rather than CPU-style recursive traversal.

The correct reusable card table is:

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
2. Run public-tree forward reach as a sparse transition operator. For each
   public edge `e = n -> child(n,a)`:

```text
r_h[child, h] += T_h[e, h] * r_h[n, h]
r_v[child, v] += T_v[e, v] * r_v[n, v]
```

   At a hero decision edge, `T_h[e,h] = sigma[I(n),h,a]` and
   `T_v[e,v] = 1`. At a villain decision edge this is reversed. At chance
   edges, `T_*` is the card-blocking mask.

3. Run terminal CFV as matrix-vector products. For each terminal `z`:

```text
V_h[z, h] = sum_v U_h[z, h, v] * r_v[z, v]
V_v[z, v] = sum_h U_v[z, h, v] * r_h[z, h]
```

   This is the main dense GEMV-like operation. A GPU implementation must split
   rows/columns into tiles:

```text
partial[z, h, tile] = sum_{v in tile} U_h[z,h,v] * r_v[z,v]
V_h[z,h]            = sum_tile partial[z,h,tile]
```

   The same applies to villain values. `U` is implicit: the shader compares
   `strength[b,h]` and `strength[b,v]`, applies pot/investment, and multiplies
   by reach.

4. Run non-terminal backup as a sparse edge operator over reverse topological
   layers:

```text
V_p[n, c] = sum_{child} B_p[n, child, c] * V_p[child, c]
```

   At own decision nodes, `B` contains strategy probabilities. At opponent
   decision nodes and chance nodes, `B` is usually a masked sum.

5. Aggregate action values and CFR weights as a sparse gather:

```text
A[I,c,a] = sum_{n maps to I} V[child(n,a), c]
M[I,c,a] = sum_{n maps to I} opponent_reach_mass[n,c]
```

6. Apply dense CFR regret/strategy update on GPU.

The chunk boundary may split combo pairs, but it must not change math. The
chunk outputs are additive numerators and denominators; division happens only
after all chunks are reduced.

## Board-Major Terminal CFV With Local Blocker Correction

The current showdown terminal kernel is terminal-major:

```text
for terminal z:
  for private combo h:
    for opponent combo v:
      for runout board B:
        compare strength[B,h] and strength[B,v]
```

For a flop or turn terminal this mixes the runout average with the
hero-villain pair loop. That prevents the GPU from using the natural final-board
parallelism and repeatedly performs the same board-local strength comparisons.

For a public board `P`, let:

- `D = Deck \ P`.
- `R_m(P)` be the full unordered set of missing runout cards, with
  `m = 5 - |P|`.
- `B = P union r` be a final board for `r in R_m(P)`.
- `C` be the private combo set.
- `r_v(v)` be villain reach at the terminal.
- `s_B(c)` be the sortable hand strength of combo `c` on final board `B`.
- `N(h) = { v in C : v intersects h }` be the blocker neighborhood of hero
  combo `h`.

For legal private pairs `h intersect v = empty`, and when `R_m(P)` is the full
runout set, the pair-specific valid-board denominator is constant:

```text
K(h,v)
 = |{ r in R_m(P) : r intersects h = empty and r intersects v = empty }|
 = choose(|D| - 4, m)
 = K
```

This constant-denominator identity is false for truncated or sampled runout
lists. Board-major terminal CFV therefore requires full runouts, or a different
sampling semantics.

With full runouts, hero terminal CFV can be reordered as:

```text
V_H(h)
 = (1/K) * sum_{r in R_m(P), r intersects h = empty}
     [ pot * W_B(h) - invested_H * T_B(h) ]
```

where `B = P union r`, and board-local nonblocked win/tie/total masses are:

```text
W_B(h) = Win_B(h) + 0.5 * Tie_B(h)
T_B(h) = Total_B(h)
```

For a fixed final board `B`, first compute raw board-legal villain masses,
ignoring hero blockers:

```text
C_B = { v in C : v intersects B = empty }

WinRaw_B(h)
  = sum_{v in C_B, s_B(v) < s_B(h)} r_v(v)

TieRaw_B(h)
  = sum_{v in C_B, s_B(v) = s_B(h)} r_v(v)

TotalRaw_B
  = sum_{v in C_B} r_v(v)
```

These raw masses can be obtained from a board-local strength grouping and a
prefix sum over group mass. Hero blockers are then subtracted by a small
neighborhood loop:

```text
BlockWin_B(h)
  = sum_{v in C_B intersect N(h), s_B(v) < s_B(h)} r_v(v)

BlockTie_B(h)
  = sum_{v in C_B intersect N(h), s_B(v) = s_B(h)} r_v(v)

BlockTotal_B(h)
  = sum_{v in C_B intersect N(h)} r_v(v)
```

Therefore:

```text
Win_B(h)   = WinRaw_B(h)   - BlockWin_B(h)
Tie_B(h)   = TieRaw_B(h)   - BlockTie_B(h)
Total_B(h) = TotalRaw_B    - BlockTotal_B(h)
```

No `+ r_v(h)` correction appears in this formulation. The neighborhood
`N(h)` is a set, not the inclusion-exclusion expression
`has_card[a] + has_card[b]`, so the combo `h = {a,b}` is subtracted exactly
once.

This transforms the board-local work from:

```text
O(|C|^2)
```

to roughly:

```text
O(|C| + G_B + |C| * |N(h)|)
```

where `G_B` is the number of distinct hand-strength groups on board `B`, and
`|N(h)|` is about 90 board-legal combos on a river board. The design avoids a
global `prefix_card[52][G_B]` table; blocker correction is computed locally
from the small neighborhood instead of being materialized as 52 prefix lanes.

## Non-Goals And Known Bad Paths

- Do not CPU-traverse `(hero_combo, villain_combo, public_tree)` for production
  solving.
- Do not cache `board -> 1326 x 1326 equity matrix` for every terminal board.
- Do not silently fall back to CPU solving when the GPU public-tree path fails.
- Do not compare full production trees against a CPU solver as the main
  correctness test; use small deterministic fixtures for equality and invariants
  for large runs.

## Implemented Direction And Gap

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

The value path computes hero values, aggregates hero infosets, then reuses the
same value buffer for villain values and aggregates villain infosets. This
avoids keeping both `value_h` and `value_v` resident at the same time.

The current terminal CFV path is partially matrix-shaped:

```text
partial[z, c, tile] = sum_{opp in tile} U[z,c,opp] * reach_opp[z,opp]
value[z,c]          = sum_tile partial[z,c,tile]
```

This removed the worst per-thread opponent loop, but it is not yet the full
linear-algebra solver. The remaining gaps are:

- reach propagation is still one shader thread per combo walking all public
  nodes;
- non-terminal backup is still scheduled by CPU node chunks, not by reverse
  sparse layers;
- action aggregation still searches public nodes inside the shader instead of
  using a compact gather table;
- terminal partial chunks are synchronized separately to respect the DZN/wgpu
  storage binding limit.

The next implementation step is therefore not another recursive traversal
optimization. It is to add explicit sparse operator buffers:

```text
decision_edges[player, public_infoset, action, child]
chance_edges[parent, child, card]
aggregate_rows[private_infoset, action] -> node/action rows
reverse_layers[k] -> node range/list
```

and make reach, backup, and aggregate kernels consume those buffers directly.

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

Blocked final boards are not neutral showdowns. If a private pair blocks every
runout represented by a terminal, that pair has no valid public outcome in that
terminal branch and contributes zero mass to terminal CFV:

```text
valid(z,h,v) = { b in boards(z) | b disjoint from h and v }

if |valid(z,h,v)| = 0:
  contribution(z,h,v) = 0
```

Do not substitute `0.5` equity for this case. A tie equity only applies to a
valid board where both hands can actually reach showdown.

### Blocker Aggregates For Denominators

The denominator reductions do not need a combo-vs-combo loop. For each public
decision infoset `I`, store 53 aggregate slots per player:

```text
A_p[I,0]      = sum_c r_p[n(I), c]
A_p[I,k + 1]  = sum_{c contains card k} r_p[n(I), c]
```

For acting combo `c = {x,y}`, the opponent non-colliding mass is:

```text
M_opp[I,c] =
  A_opp[I,0] - A_opp[I,x+1] - A_opp[I,y+1] + r_opp[n(I), c]
```

The last term restores the exact combo `{x,y}`, which was subtracted once for
each card. This changes decision denominators from:

```text
O(|private_infosets| * |C|)
```

to:

```text
O(|public_infosets| * 53 * |C|) + O(|private_infosets|)
```

This is the same algebra used by fold terminal values and should be preferred
wherever only card blockers, not hand strength order, determine legal opponent
mass.

### Strength-Prefix Terminal Target

Showdown terminal CFV still has genuine strength-dependent pairwise work. For a
fixed terminal `z` and final board `b`, define opponent reach buckets by hand
strength:

```text
B_v[s] = sum_{v: S[b,v] = s} r_v[z,v]
P_v[s] = sum_{t < s} B_v[t]
E_v[s] = B_v[s]
```

Ignoring blockers for the moment, hero combo `h` with strength `s_h` has:

```text
win_mass  = P_v[s_h]
tie_mass  = E_v[s_h]
eq_mass   = win_mass + 0.5 * tie_mass
total     = P_v[+inf]
```

Card blockers must subtract the same prefix/equal aggregates restricted to
opponent combos containing either private card of `h`. Therefore the useful GPU
object is not an equity matrix; it is a per-board prefix table with blocker
lanes:

```text
prefix_total[b, strength_rank]
prefix_card[b, card, strength_rank]
equal_total[b, strength_rank]
equal_card[b, card, strength_rank]
```

Then each terminal CFV row becomes a small number of prefix lookups per final
board instead of scanning every opponent combo. This is the next fundamental
speed target for terminal CFV.

The sort itself is not iteration-dependent:

```text
order[b,*] = argsort_c(S[b,c])
group_start[b,c], group_end[b,c]
```

This can be built once when the subgame is created. A GPU radix sort is a good
fit here, but it belongs to subgame setup, not the CFR loop. Inside each CFR
iteration the dynamic input is only reach:

```text
sorted_reach[pos] = r_opp[z, order[b,pos]]
prefix_reach      = scan(sorted_reach)
```

The high-value kernel work is therefore parallel scan / segmented scan over
`(terminal, board)` rows, followed by O(1) prefix lookups for each private combo.
Replacing CPU precomputed `order` with GPU sort should not change this operator
interface.

### Blocker Card Aggregate Feasibility

For a fixed terminal `z`, final board `b`, and hero combo `h = (a,b)`, the
current reduce kernel subtracts all opponent combos that intersect `h`.
For opponent reach vector `r[v]`, board-legal combos only, and hero strength
group `g_h`, the exact correction can be written with card aggregates:

```text
block_total(h) =
  total_card[a] + total_card[b] - r[h]

block_win(h) =
  prefix_card[a, g_h] + prefix_card[b, g_h]

block_tie(h) =
  equal_card[a, g_h] + equal_card[b, g_h] - r[h]
```

The `- r[h]` term appears for total and tie because combo `h` contains both
hero cards and is counted in both card lanes. It does not appear in
`block_win`, because `h` is in the equal group, not in a lower-strength group.
The same equations apply symmetrically with hero reach when computing villain
values.

This is mathematically equivalent to the current neighbor-combo loop:

```text
for v in C:
  if v intersects h and legal(v,b):
    subtract r[v] from total
    subtract r[v] from win if S[v] < S[h]
    subtract r[v] from tie if S[v] = S[h]
```

The hard part is not the formula; it is building the aggregate cheaply on GPU.
A naive table

```text
prefix_card[z,b,card,position]
```

has `52 * (combo_count + 1)` entries per `(z,b)` row, which is usually too much
memory traffic. A strength-group table is smaller:

```text
prefix_card[z,b,card,group]
equal_card[z,b,card,group]
```

but it requires a way to accumulate reach into `(card, group)` buckets. Without
portable `fp32` atomics, that accumulation cannot be a simple scatter-add.
Reasonable implementations are:

```text
1. group/card gather:
   one invocation per (z,b,card,group) scans the combos in that group

2. sorted-order fused scan:
   while scanning strength order for total prefix, also emit card-lane group
   masses for the two cards of each combo

3. two-stage deterministic reduce:
   produce per-tile card/group partials, then reduce tiles
```

The expected speedup is bounded by the measured blocker cost. On `As7h2c`,
depth 5, WSL/DZN profiling showed reduce dropping from about `381ms` to
`244ms` when blocker correction was disabled, so blocker work is roughly
`135ms` of the stage-profile run. Card aggregate can target that cost, but it
does not remove prefix construction or board iteration by itself.

## Next Mathematical Cut

The current implementation is still too close to "run a public tree program on
the GPU". The next solver step is to freeze one explicit algebraic interface
between the public tree and the kernels. The interface should be small enough
that each operation can be checked independently.

For a fixed subgame, define:

```text
C = set of private combos, |C| = 1326 before board blockers
N = public nodes
D = public decision nodes
Z = terminal public nodes
E = public edges
```

For each player `p in {h, v}` and combo `c in C`, each iteration stores:

```text
r_p[n,c]       private reach at public node n
u_p[n,c]       counterfactual value at public node n
sigma_p[d,c,a] current regret-matched strategy
R_p[d,c,a]     cumulative regret
S_p[d,c,a]     cumulative average strategy numerator
```

The iteration should be decomposed into these operators:

```text
1. sigma = regret_match(R)
2. r     = ForwardReach(sigma)
3. u_Z   = TerminalCfv(r)
4. u     = BackwardValue(sigma, u_Z)
5. q,m   = ActionValueGather(r, u)
6. R,S   = CfrUpdate(R,S,q,m,r,sigma)
```

Only steps 1 and 6 are dense per-infoset operations. Steps 2, 4, and 5 are
sparse public-tree operators. Step 3 is the only dense combo-vs-combo operation.

### Forward Reach

Use public topological layers. For each edge `e = n -> child(n,a)`:

Hero acting node:

```text
r_h[child,h] += r_h[n,h] * sigma_h[I(n),h,a]
r_v[child,v] += r_v[n,v]
```

Villain acting node:

```text
r_h[child,h] += r_h[n,h]
r_v[child,v] += r_v[n,v] * sigma_v[I(n),v,a]
```

Chance edge for public card `k`:

```text
r_h[child,h] += r_h[n,h] * 1[k notin h]
r_v[child,v] += r_v[n,v] * 1[k notin v]
```

The public chance probability `pi_c(n)` should be represented separately as a
scalar per node or folded into terminal/action denominators. Do not duplicate it
inside every private reach vector unless measurement proves it is cheaper.

### Terminal CFV

At showdown terminal `z`, the expensive operation is:

```text
u_h[z,h] = pi_c(z) * sum_v M_z[h,v] * r_v[z,v]
u_v[z,v] = pi_c(z) * sum_h N_z[v,h] * r_h[z,h]
```

where:

```text
M_z[h,v] = 1[not collide(h,v)] * (p(z) * W[b(z),h,v] - i_h(z))
N_z[v,h] = -M_z[h,v]
```

`M_z` and `N_z` are implicit. The shader derives them from:

```text
strength[board(z), combo]
pot[z]
hero_invested[z]
combo card masks
```

Fold terminals are not combo-vs-combo dense products. They are rank-one style
opponent reach reductions:

Hero wins by villain fold:

```text
u_h[z,h] = pi_c(z) * (p(z) - i_h(z)) *
           sum_v 1[not collide(h,v)] * r_v[z,v]
u_v[z,v] = -pi_c(z) * i_v(z) *
           sum_h 1[not collide(h,v)] * r_h[z,h]
```

Villain wins by hero fold is symmetric. This is why fold terminals should stay
on a specialized path.

### Backward Value

Use reverse public layers. For decision node `n`:

Hero acting:

```text
u_h[n,h] = sum_a sigma_h[I(n),h,a] * u_h[child(n,a),h]
u_v[n,v] = sum_a u_v[child(n,a),v]
```

Villain acting:

```text
u_h[n,h] = sum_a u_h[child(n,a),h]
u_v[n,v] = sum_a sigma_v[I(n),v,a] * u_v[child(n,a),v]
```

Chance node:

```text
u_p[n,c] = sum_k 1[k notin c] * u_p[child_k,c]
```

If public chance probability is stored separately, do not divide here; keep
`u_p` and the denominators in the same convention until action-value gather.

### Action Value And Denominator Gather

At decision node `d`, action values are child CFVs normalized by opponent
counterfactual reach mass.

Hero acting:

```text
q_h[d,h,a] = u_h[child(d,a),h] / m_h[d,h]
m_h[d,h]   = pi_c(d) * sum_v 1[not collide(h,v)] * r_v[d,v]
```

Villain acting:

```text
q_v[d,v,a] = u_v[child(d,a),v] / m_v[d,v]
m_v[d,v]   = pi_c(d) * sum_h 1[not collide(h,v)] * r_h[d,h]
```

The same denominator is used for all legal actions at `(d,c)`. This matters:
do not compute one denominator per action unless chance or abstraction semantics
actually make opponent reach action-dependent.

Average strategy reach uses the acting player's own reach:

Hero acting:

```text
s_weight_h[d,h] = pi_c(d) * r_h[d,h]
```

Villain acting:

```text
s_weight_v[d,v] = pi_c(d) * r_v[d,v]
```

### CFR Update

For acting player `p` at `(d,c)`:

```text
node_value[d,c] = sum_a sigma_p[d,c,a] * q_p[d,c,a]
R_p[d,c,a]     += m_p[d,c] * (q_p[d,c,a] - node_value[d,c])
S_p[d,c,a]     += t * s_weight_p[d,c] * sigma_p[d,c,a]
```

For CFR+:

```text
R_p[d,c,a] = max(R_p[d,c,a], 0)
```

For DCFR-style discounting, apply the discount to old regret before adding the
new instantaneous regret. The current code already follows this shape for dense
updates; the important part is that `q`, `m`, and `s_weight` come from the same
reach convention.

## Convergence Diagnostics

Arena win rate is too noisy to be the first convergence signal. Add solver-local
metrics before trusting match results.

For every sampled fixed flop and iteration count, record:

```text
root_strategy_l1_delta(t) =
  sum_{h,a} |avg_sigma_t[root,h,a] - avg_sigma_{t/2}[root,h,a]|
  / number_of_legal_entries

root_value_delta(t) =
  mean_h |q_t[root,h,best_avg_action] - q_{t/2}[root,h,best_avg_action]|

regret_mass(t) =
  sum_{d,c,a} max(R[d,c,a], 0)

illegal_strategy_mass(t) =
  sum_{illegal d,c,a} S[d,c,a]
```

Expected invariants:

```text
illegal_strategy_mass(t) == 0
all regrets, strategy sums, reaches, and values are finite
for every legal (d,c): sum_a current_sigma[d,c,a] = 1
for every legal (d,c) with positive strategy sum:
  sum_a avg_sigma[d,c,a] = 1
```

For small trees, additionally compute a one-step best-response check by fixing
one player's average strategy and enumerating the other player's actions. This
is not a full production exploitability calculation, but it catches sign errors,
wrong denominator choices, and strategy applied to the wrong player.

The current `solve-flop-metrics` output includes two best-response gap proxies:

```text
root_br_gap(t)
local_br_gap(t)
```

`root_br_gap` is the average root one-step best-action improvement over the
current average strategy. `local_br_gap` applies the same one-step check at all
infosets and weights by reach. These are not full recursive exploitability
numbers, so do not present them as final game exploitability. They are still a
better tuning signal than root strategy movement or raw regret mass: a variant
that merely freezes early can have a small `root_strategy_l1_delta` while still
having a bad best-response gap.

## DCFR+ Parameter Notes

`CfrVariant::DcfrPlus { alpha, gamma }` uses:

```text
regret_discount(t) =
  0                                  if t <= 1
  (t - 1)^alpha / ((t - 1)^alpha + 1.5) otherwise

average_strategy_discount(t) =
  ((t - 1) / t)^gamma                if t > 1
  1                                  otherwise
```

`alpha` controls how quickly old regrets recover from the first-iteration reset.
Higher values make the old regret discount approach `1` faster after the first
few iterations. `gamma` controls how aggressively early average-strategy mass is
discounted. Higher values make the reported average strategy depend more on
later iterations.

Single-flop sweep on `As7h2c`, depth `5`, full terminal runouts, using the
current `root_br_gap` and `local_br_gap` proxies:

```text
CFR+ 128:
  root_br_gap  = 0.942252
  local_br_gap = 4.020251

DCFR+ alpha=1.5 gamma=8.0 128:
  root_br_gap  = 0.734763
  local_br_gap = 3.610855

DCFR+ alpha=2.5 gamma=8.0 128:
  root_br_gap  = 0.710786
  local_br_gap = 3.595675

DCFR+ alpha=1.5 gamma=12.0 128:
  root_br_gap  = 0.712415
  local_br_gap = 3.601657
```

Current interpretation:

- DCFR+ is better than CFR+ on this fixed flop, but not by an order of
  magnitude.
- `alpha=2.5, gamma=8.0` is the current single-flop best among tested values.
- The gain over nearby settings is small, so do not hard-code this as a global
  default without testing other flop textures.
- The old literature-style starting point `alpha=1.5, gamma=4.0` was on the
  edge of the first grid; expanding `gamma` improved the BR gap.

## Immediate Implementation Order

1. Add a `solve-flop-metrics` CLI path that runs one fixed flop for iteration
   counts like `1,2,4,8,16,32` and prints root strategy deltas, value deltas,
   regret mass, illegal mass, runtime, and peak-ish resident sizes.
2. Add compact gather buffers for action values:

```text
decision_rows: [node, public_infoset, acting_player, first_child, child_count]
```

   The kernel should stop searching public nodes and consume this table.
3. Split terminal CFV into two explicit kernel families:

```text
showdown_cfv_tiles
fold_cfv_reductions
```

   Keep their outputs in the same `u_p[z,c]` convention.
4. Only after the metrics are stable, benchmark arena matches at increasing
   iteration counts. A match win alone is not evidence of convergence.
