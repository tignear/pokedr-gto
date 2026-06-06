# Terminal CFV Batched Matvec Plan

The current terminal CFV kernel evaluates each `(terminal, hero_combo)` row
independently. For river terminals (`board_count == 1`) this repeats the same
final-board rank/blocker logic for every terminal that shares the same final
board table.

For a fixed final board `b`, define the hero payoff matrix

```text
A^H_b[h, v] =
    0                                      if h blocks v or h/v is illegal on b
    pot                                   if h beats v on b
    0.5 * pot                             if h ties v on b
    0                                      if h loses to v on b
  - hero_invested
```

and the villain matrix analogously:

```text
A^V_b[v, h] =
    0                                      if v blocks h or v/h is illegal on b
    pot                                   if v beats h on b
    0.5 * pot                             if v ties h on b
    0                                      if v loses to h on b
  - villain_invested
```

For all terminals `t` that share board table `b`, terminal CFV is then

```text
U^H_b[:, t] = chance_scale_t / denom_t * A^H_b * reach^V_t
U^V_b[:, t] = chance_scale_t / denom_t * A^V_b * reach^H_t
```

The existing prefix/blocker implementation is an efficient single-vector
matvec. The next meaningful optimization is to batch several `reach_t` columns
for the same final board so the rank/blocker decisions for `A_b[h, v]` are
computed once per tile and reused across terminal columns.

Observed on `As7h2c`, `depth=5`:

```text
board_count=1:  20 groups, 12808 tables, 122304 terminals, 9.55 terminals/table
board_count=48:  1 group,    49 tables,    294 terminals, 6.00 terminals/table

total showdown terminals:       122598
unique final-board tables:       12857
terminal reduce lanes:       162564948
terminal-weighted strength groups: 14211160
max strength groups per board:     185
```

Reach-sharing diagnostics after `8` DCFR iterations on the same tree:

```text
average strategy:
  showdown terminals:          122598
  river showdown terminals:    122304
  board tables:                  1225
  raw reach signatures:        110654
  normalized reach signatures: 110655
  support signatures:            1225

current strategy:
  raw reach signatures:         90447
  normalized reach signatures:  89671
  support signatures:           86237
```

Implications:

- Board-parallel reduce is not the main lever because almost all terminals are
  already single-board river terminals.
- The average batch width per final board table is about `10` terminals, so a
  batched kernel must be careful: it can win by reusing rank/blocker checks and
  coalescing reach reads, but it cannot rely on very wide GEMM columns.
- Exact or scalar-multiple reach sharing is weak. For average strategy, a board
  table with `104` river terminals typically still has `~92-96` normalized reach
  signatures. Support alone is shared by board table, but the actual reach
  weights are not. A batching design should not assume many terminals can share
  the same opponent reach vector.
- Materializing dense `A_b` for every board table is too large. A practical GPU
  kernel should generate `A_b` tiles from combo/card/bounds data on the fly.
- A fully materialized `(terminal, card, strength_group)` prefix also looks too
  large: for the same trace it is about `6.0GB` of f32-pair cells. The useful
  formula is still:

```text
block_total(h) = prefix_card[c1, G] + prefix_card[c2, G] - reach[h]
block_win(h)   = prefix_card[c1, g(h)] + prefix_card[c2, g(h)]
block_tie(h)   = equal_card[c1, g(h)] + equal_card[c2, g(h)] - reach[h]
```

  but those card/group aggregates need to be produced and consumed inside a
  tile or small board-table batch rather than stored for every terminal.
- Previous full 52-card prefix and per-combo workgroup blocker reductions were
  slower; do not retry those shapes unchanged.
