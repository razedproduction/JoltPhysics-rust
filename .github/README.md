# JoltPhysics-rust

[![Rust](https://github.com/razedproduction/JoltPhysics-rust/actions/workflows/rust.yml/badge.svg)](https://github.com/razedproduction/JoltPhysics-rust/actions/workflows/rust.yml)
[![License: MIT](https://img.shields.io/badge/license-MIT-blue.svg)](../LICENSE)

A work-in-progress reimplementation of [Jolt Physics](https://github.com/jrouwe/JoltPhysics) in Rust.

Jolt Physics is written by Jorrit Rouwe and released under the MIT license (see [LICENSE](../LICENSE)); this port is
derived from it and keeps the same license. This repository is a fork of `jrouwe/JoltPhysics`: the original C++ sources
are kept untouched as the reference implementation (see [README.md](../README.md) for the original project), the Rust
port lives in [`crates/`](../crates).

## Layout

| Crate | Contents |
|---|---|
| [`jolt`](../crates/jolt) | facade crate re-exporting everything |
| [`jolt-core`](../crates/jolt-core) | core utilities and math (`Jolt/Core`, `Jolt/Math`) |
| [`jolt-geometry`](../crates/jolt-geometry) | geometry, AABB trees (`Jolt/Geometry`, `Jolt/AABBTree`, `Jolt/TriangleSplitter`) |
| [`jolt-physics`](../crates/jolt-physics) | simulation (`Jolt/Physics`, `Jolt/Skeleton`) |
| [`jolt-test-utils`](../crates/jolt-test-utils) | shared test helpers (port of `UnitTests/UnitTestFramework.h`), not published |

Cargo features: `double-precision`, `cross-platform-deterministic`, `debug-renderer`, `profiler`.

## Status

Early scaffolding. See [port/PORTING_PLAN.md](../port/PORTING_PLAN.md) for the plan and slice status,
[port/TEST_STATUS.md](../port/TEST_STATUS.md) for ported unit tests and [port/DEVIATIONS.md](../port/DEVIATIONS.md) for
intentional differences from the C++ original.

## Building

Requires stable Rust (MSRV 1.85, edition 2024).

```bash
cargo test --workspace
cargo test --workspace --features jolt/double-precision
```

Before submitting changes:

```bash
cargo fmt --all
cargo clippy --workspace --all-targets --all-features -- -D warnings
python tools/port/test_status.py
```
