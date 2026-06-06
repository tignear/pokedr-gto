# River Batch Solver

The river solver is the boundary between a smaller flop/turn trunk and many
fixed-river subgames.

## Input Contract

Each river subgame is defined by:

- `PublicState` at the river root:
  - 5-card board
  - pot, investments, effective stack
  - `to_call`, acting player, raise/check counters
- OOP combo reach weights, length `COMBO_COUNT`
- IP combo reach weights, length `COMBO_COUNT`

The reach weights are boundary conditions from the previous streets. They are
not fixed preflop priors. A trunk solver must update them according to the path
that reached the river board before calling the river batch solver.

## Output Contract

The implementation returns the solved `DenseCfrState`, summary data, and
per-combo root CFVs:

- `oop_cfv[COMBO_COUNT]`
- `ip_cfv[COMBO_COUNT]`

Those CFVs are what the flop/turn trunk should back up through river chance
edges. They are currently produced from the average strategy profile by
aggregating pairwise profile payoffs against the opponent's boundary reach
weights.

## Current Status

`RiverBatchSolver` is a clean wrapper around the existing public-tree CFR path.
It can solve many fixed river boards in one process and reuses the thread-local
GPU backend. It now exposes the boundary CFVs needed by a trunk solver. It is
still sequential per board; it is not yet a true batched GPU kernel.

`solve-river-runouts` measures the current sequential baseline. On `As7h2c`
with `64` generated river boards, `8` iterations took about `10.4s`
(`0.162s/board`) and `1` iteration took about `7.4s` (`0.116s/board`). That
projects to hundreds of seconds for `49*48` ordered runouts, so the current
wrapper is not close to a sub-second batched river solver.

The intended next step is grouping identical river tree shapes so one dispatch
sequence processes multiple `(state, oop_weights, ip_weights)` inputs.
