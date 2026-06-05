# Agent Notes

## GPU checks

GPU-visible commands must be run with escalated permissions in this environment.
This includes `vulkaninfo`, `gpu-info`, `gpu-smoke`, and workspace tests when the
GPU smoke integration test is expected to prove anything.

Use the pinned toolchain explicitly:

```bash
cargo test --workspace
```

Run that command with escalation. A non-escalated run can hide `/dev/dxg` or
DXCore/Vulkan access and may print `no GPU adapter visible to wgpu` even when the
GPU is working. Do not diagnose GPU breakage from a non-escalated GPU test.

## Optimization notes

When an optimization attempt fails or is reverted, record it in the repo before
moving on. Include the date, what was tried, why it was expected to help, the
measurement or failure mode, and why it was not kept. This avoids retrying the
same card-prefix, parallel-reduce, buffer-layout, or driver-workaround ideas
without new evidence.

## CFR references

`rs_poker` 4.1.0 has an Arena CFR implementation under
`src/arena/cfr`, but it is not a full-range public-tree postflop solver. It
uses a lazily expanded arena tree plus `little_sorry::RegretMatcher` and
rollout/simulation rewards. Treat it as a reference for simple tree storage and
action-generation plumbing, not as a correctness oracle for this solver's CFV,
BR, or exploitability calculations.
