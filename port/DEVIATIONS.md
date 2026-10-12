# Deviations from C++ Jolt Physics

Every intentional difference between the Rust port and the C++ original (base commit in
[`PORTING_PLAN.md`](PORTING_PLAN.md) §1). Bugs found in the original are ported as-is and listed here unless the
maintainer decides to fix them.

| Area | Original behavior | Rust behavior | Reason | Approved? |
|---|---|---|---|---|
| Scope | `Jolt/Compute`, `Jolt/Physics/Hair`, `Jolt/Shaders` (GPU compute: DX12/Vulkan/Metal/CPU) | not ported | out of initial scope | planned |
| Reflection | custom RTTI, `Factory`, `ObjectStream` text/binary serialization of settings | not ported; optional `serde` derives on settings instead | Rust has no need for a hand-rolled RTTI | planned |
| Memory | `Allocate`/`Free` hooks, `STLAllocator`, `STLLocalAllocator`, `TempAllocator` | global Rust allocator, `Vec` | idiomatic; revisit for `TempAllocator` if profiling shows a cost | planned |
| Containers | `Array<T>` (own vector) | `Vec<T>`; most of `ArrayTest.cpp` / `STLLocalAllocatorTest.cpp` not ported | std provides the same contract | planned |
| FMA | used when `USE_FMADD` and not `CROSS_PLATFORM_DETERMINISTIC` | never used | determinism; matches the cross-platform-deterministic build | planned |

## Unported tests

| C++ test | Reason |
|---|---|
| | |

## Bugs found in the original

| Location | Description | Ported as-is? |
|---|---|---|
| | | |
