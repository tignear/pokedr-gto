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

The first implementation returns the solved `DenseCfrState` plus summary data.
The next boundary output should add per-combo root CFVs:

- `oop_cfv[COMBO_COUNT]`
- `ip_cfv[COMBO_COUNT]`

Those CFVs are what the flop/turn trunk should back up through river chance
edges.

## Current Status

`RiverBatchSolver` is a clean wrapper around the existing public-tree CFR path.
It can solve many fixed river boards in one process and reuses the thread-local
GPU backend. It is still sequential per board; it is not yet a true batched GPU
kernel.

The intended next step is grouping identical river tree shapes so one dispatch
sequence processes multiple `(state, oop_weights, ip_weights)` inputs.
