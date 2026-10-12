# PORTING_PLAN: Jolt Physics (C++) → Rust

Source of truth for the port. Update the status table (§4) as slices land; record every behavior change in
[`DEVIATIONS.md`](DEVIATIONS.md); regenerate [`TEST_STATUS.md`](TEST_STATUS.md) with `python tools/port/test_status.py`.

## 1. Contract

- **Scope**: the `Jolt/` library (rigid bodies, collision, constraints, character, ragdoll, vehicles, soft bodies).
  Out of scope for now: `Jolt/Compute` + `Jolt/Physics/Hair` (GPU), `Jolt/Shaders`, `JoltViewer`, `Samples`,
  `TestFramework` (renderer/UI). `HelloWorld` and `PerformanceTest` get Rust ports as examples/benches.
- **Fidelity**: behavior-equivalent, verified by the ported `UnitTests/`. Stretch goal (milestone M6): bit-identical
  simulation with the C++ `CROSS_PLATFORM_DETERMINISTIC=ON` build, proven by the `PerformanceTest -validate_hash`
  hashes in `.github/workflows/determinism_check.yml`. Every design choice must keep that goal reachable (see §3).
- **Integration**: pure Rust library crates (`jolt` facade). No C ABI for now.
- **Constraints**: stable Rust, edition 2024, MSRV 1.85; `std`; no `unsafe` except in modules that need it (SIMD
  intrinsics, lock-free containers, job system), each block with a `// SAFETY:` comment; Windows/Linux/macOS, x86-64 and
  aarch64. Allowed deps: well-known crates only, added when a slice needs them (likely: `bitflags`, `smallvec`,
  `parking_lot`, `serde` (optional), `criterion` (dev)).
- **Original**: C++17, MIT, by Jorrit Rouwe. Tracked upstream: `https://github.com/jrouwe/JoltPhysics`, remote `upstream`.
  **Base commit of the port: `3c91764e`** (README.md: add link to Nim bindings, #2156). When merging newer upstream,
  `git diff 3c91764e..upstream/master -- Jolt UnitTests` lists what to re-port; then bump this hash.
- **Oracle**: (1) `UnitTests/` (592 doctest cases) translated to Rust; (2) the runnable C++ build, once a C++ toolchain is
  set up (see §5); (3) determinism hashes from upstream CI.

## 2. Inventory summary

| Item | Notes |
|---|---|
| Size (`Jolt/`, .h/.cpp/.inl) | Physics 70.2k (Collision 26.8k of which Shape 17.6k, BroadPhase 3.8k; Constraints 14.3k; Body 6.3k; Vehicle 4.8k; SoftBody 4.6k; Character 3.6k; Hair 2.6k; Ragdoll 1.0k), Core 11.0k, Math 10.8k, Geometry 7.5k, Compute 4.6k, ObjectStream 3.4k, Renderer 2.2k, AABBTree 1.5k, Shaders 1.1k, Skeleton 1.0k, TriangleSplitter 0.4k. ≈113k lines total |
| Tests | `UnitTests/` 592 `TEST_CASE`s (doctest), helpers in `UnitTestFramework.h`, `PhysicsTestContext`, `Layers.h`, `Logging*Listener.h` |
| Entry points / public API | `RegisterTypes`/`Factory`, `PhysicsSystem` (`Init`, `Update`, `GetBodyInterface`, queries), `BodyCreationSettings`, `*ShapeSettings::Create`, `*ConstraintSettings::Create`, `CharacterVirtual`, `Ragdoll`, `VehicleConstraint`, `SoftBodyCreationSettings` |
| Global/mutable state | `Factory::sInstance`, allocator hooks (`Allocate`/`Free`), `Trace`/`AssertFailed` hooks, profiler, collision dispatch tables (`CollisionDispatch::sCollideShape[][]`), `Shape::sRegistry` |
| Concurrency | `JobSystem` (thread pool, barriers, job dependencies), lock-free `LockFreeHashMap`, atomics everywhere in `PhysicsSystem::Update`, `BodyLockInterface` (`MutexArray`), `PhysicsLock` |
| Error model | asserts (`JPH_ASSERT`, debug only), `Result<T>` for settings→object creation, `EPhysicsUpdateError` flags, no exceptions |
| Numeric semantics | `float` default, `JPH_DOUBLE_PRECISION` switches positions to `double` (`Real`, `RVec3`, `RMat44`). Own trig (`Vec4::SinCos`, `ATan`, ...) — must be ported, never use `std` trig (libm differs per platform). `Vec3` keeps Z duplicated in W. FMA only when not `CROSS_PLATFORM_DETERMINISTIC`. Denormal flushing (`FPFlushDenormals`), FP exceptions in debug |
| Determinism-relevant | own `QuickSort`/`InsertionSort` (unstable, order matters), `HashTable`/`UnorderedMap` (iteration order), body/constraint ordering, island building, `DeterminismLog` |
| Dependencies | none external (STL only) |
| Risky constructs | intrusive refcounting `Ref`/`RefConst`, custom RTTI (`JPH_DECLARE_RTTI_*`) + `ObjectStream` reflection, class hierarchies (`Shape`, `Constraint`, `ConstraintPart`), raw `Body*` pointers with `BodyID` (index+sequence), double dispatch tables, `TempAllocator` stack allocator, SIMD intrinsics (SSE/AVX/NEON/RVV/WASM), unions/type punning in math, `memcpy`-based `StreamIn/Out` |

## 3. Rust design

### Crate layout (workspace)

| Crate | Ports | Depends on |
|---|---|---|
| `crates/jolt-core` | `Jolt/Core` (crate root, one module per header) + `Jolt/Math` (`math::*`) | — |
| `crates/jolt-geometry` | `Jolt/Geometry` (crate root), `Jolt/AABBTree` (`aabb_tree`), `Jolt/TriangleSplitter` (`triangle_splitter`) | core |
| `crates/jolt-physics` | `Jolt/Physics` (modules follow subdirectories: `body`, `collision::shape`, `constraints`, ...), `Jolt/Skeleton` (`skeleton`), `Jolt/Renderer` (`debug_renderer`, feature) | core, geometry |
| `crates/jolt` | facade: `pub use jolt_core::*`, `geometry`, `physics`; examples (`HelloWorld`), benches (`PerformanceTest`) | all |
| `crates/jolt-test-utils` | `UnitTestFramework.h` (`check_approx_equal_*`), later `PhysicsTestContext`, `Layers`, logging listeners | as needed |

Core and Math share a crate because they include each other (`Core/HashTable.h` uses `BVec16`; Math uses `Core.h`,
`HashCombine.h`). Split further only if compile times demand it.

### Cargo features (mirror CMake options, forwarded down the crate graph)

`double-precision` (`DOUBLE_PRECISION`), `cross-platform-deterministic` (`CROSS_PLATFORM_DETERMINISTIC`),
`debug-renderer` (`JPH_DEBUG_RENDERER`), `profiler` (`JPH_PROFILE_ENABLED`). Asserts: `debug_assert!` (=`JPH_ASSERT`).

### Key decisions

| C++ | Rust | Why |
|---|---|---|
| `Vec3`/`Vec4`/`UVec4`/`Mat44`/`Quat` (SIMD classes) | `#[repr(C, align(16))]` `Copy` structs over `[f32; 4]`; scalar implementation first, `std::arch` SSE/NEON backends later behind `cfg`, required to give bit-identical results | correctness first; IEEE add/mul/div/sqrt are exact so SIMD and scalar agree |
| FMA (`USE_FMADD`) | never fused; Rust does not contract `a*b+c` | matches the `CROSS_PLATFORM_DETERMINISTIC` oracle |
| own trig / `Sqrt` | ported 1:1 (`Trigonometry.h`, `Vec4::SinCos`, ...) | platform libm would break determinism |
| `Real`, `RVec3`, `RMat44` | type aliases switched by `double-precision` feature | same as C++ |
| `Ref<T>` / `RefConst<T>` (intrusive refcount) | `Arc<T>` / `Arc<dyn Shape>`; shapes immutable after creation | shared, thread-safe ownership |
| `Shape` hierarchy + `EShapeType`/`EShapeSubType` | `trait Shape: Send + Sync` + `sub_type()`; downcast via `as_any()` | users can add custom shapes (C++ allows it) |
| `CollisionDispatch` 2D function tables | table of fn pointers indexed by `(EShapeSubType, EShapeSubType)`, filled at registration | keeps double dispatch + extensibility |
| `Body*` + `BodyID` (index + sequence) | `BodyManager` owns `Vec<Option<Box<Body>>>`/arena, `BodyID` exactly as C++ (same bit layout) | generational-index pattern, deterministic |
| `Constraint` hierarchy | `trait Constraint`, `Arc<dyn Constraint>` in `ConstraintManager`; `ConstraintPart`s are plain structs | |
| listeners (`ContactListener`, `BodyActivationListener`, ...) | traits with default methods, registered as `Arc<dyn Trait + Send + Sync>` | called from worker threads |
| `TempAllocator` | not ported at first: plain `Vec`s; revisit with a bump allocator if profiling says so | → DEVIATIONS |
| `Array`, `StaticArray`, `UnorderedMap/Set`, STL allocators | `Vec`, `arrayvec`/`smallvec` or own `StaticArray`; **`HashTable` ported** (iteration order is observable) | determinism |
| `QuickSort`, `InsertionSort`, `BinaryHeap` | ported 1:1, never `sort_unstable` as replacement | element order affects simulation |
| `JobSystem` | trait `JobSystem`; `JobSystemSingleThreaded` first, then thread pool port; `PhysicsSystem::Update` single-threaded until M5 | ship a correct sequential engine first |
| `PhysicsLock`, `BodyLockInterface`, `MutexArray` | `std::sync`/`parking_lot` mutexes + RAII guards | |
| RTTI, `Factory`, `ObjectStream` | dropped; `serde` feature for settings serialization; `StreamIn/Out` + `SaveBinaryState` ported for binary state | → DEVIATIONS |
| `Result<T>` | `Result<T, JoltError>` (message string like C++), `thiserror` if useful | |
| `JPH_ASSERT` | `debug_assert!` | |
| `Trace`, `AssertFailed` hooks | `log`-style callback in `jolt-core` (or `tracing` later) | |

### Naming

`snake_case` methods, C++ prefixes dropped: `GetX()` → `x()`, `SetX()` → `set_x()`, `sReplicate()` → `replicate()`,
`inValue`/`outValue`/`mValue` → `value`, `IsClose` → `is_close`. Out-parameters become return values (tuples/structs).
Types keep their C++ names (`Vec3`, `BodyID`, `PhysicsSystem`), except enums, which drop the `E` prefix
(`EMotionType` → `MotionType`). Record non-obvious renames below.

| C++ | Rust |
|---|---|
| `EMotionType` etc. | `MotionType` (enum `E` prefix dropped) |
| `Vec3::sReplicate(v)` | `Vec3::replicate(v)` |
| `GetX()/SetX()` | `x()/set_x()` |
| `Hash<T>` functor | trait `JoltHash::jolt_hash()` (avoids clash with `std::hash::Hash`) |
| `HashCombineArgs(...)`, `JPH_MAKE_HASHABLE(T, ...)` | `hash_combine_args!(...)`, `make_hashable!(T, field, ...)` |
| `HashBytes(p, n, seed)` / `HashString(s, seed)` | `hash_bytes(&[u8])` + `hash_bytes_with_seed`, `hash_string` + `hash_string_with_seed` |
| `Color::sRed`, `Color(uint32)`, `Color(c, alpha)`, `GetUInt32()`, `sGetDistinctColor` | `Color::RED`, `Color::from_u32`, `Color::with_alpha`, `uint32()`, `distinct_color` |
| `StringFormat(fmt, ...)` | `string_format!` (Rust `format!` syntax) |
| `StringToVector(s, out, delim, clear)` / `VectorToString(v, out, delim)` | `string_to_vector(s, delim) -> Vec<String>` / `vector_to_string(v, delim) -> String` |
| `Result<T>` | `jolt_core::Result<T>` = `std::result::Result<T, JoltError>`; empty state = `Option<Result<T>>` |
| `JPH_SCOPE_EXIT(f)` | `scope_exit!(f)` |
| `Math.h` free functions | `jolt_core::math::*` (in `math/mod.rs`, no `math::math`) |
| `JPH_PI`, `cLargeFloat` | `math::JPH_PI`, `math::LARGE_FLOAT` |
| `std::min` / `std::max` (via `Core.h`) | `math::min` / `math::max` (C++ semantics: `b < a ? b : a`, no NaN special case like `f32::min`) |
| `Sqrt(float)` / `Sqrt(double)` overloads | `math::sqrt<T: Float>` (trait `Float` for `f32`/`f64`) |
| `GetNextPowerOf2`, `IsPowerOf2`, `CountBits` | `get_next_power_of_2`, `is_power_of_2`, `count_bits` |
| `BitCast<uint32>(f)` / `BitCast<float>(u)` | `f.to_bits()` / `f32::from_bits(u)` |
| `FindRoot(a, b, c, outX1, outX2) -> int` | `find_root(a, b, c) -> (i32, x1, x2)` |
| `HalfFloatConversion::FromFloat<ROUND_TO_NEAREST>(v)` | `half_float::from_float(v, RoundingMode::RoundToNearest)` (template arg -> argument) |
| `ACosApproximate` | `acos_approximate` (C++ `ACos` -> `acos`, `ATan2` -> `atan2`, ...) |
| `InsertionSort(b, e[, cmp])`, `QuickSort(b, e[, cmp])` | `insertion_sort(&mut [T])` / `insertion_sort_by(&mut [T], less)`, `quick_sort` / `quick_sort_by` (`less(a, b) -> bool`, not `Ordering`) |
| `BinaryHeapPush(b, e, pred)` / `BinaryHeapPop(b, e, pred)` | `binary_heap_push(&mut [T], pred)` / `binary_heap_pop(&mut [T], pred)` |
| `HashTable` / `UnorderedMap` / `UnorderedSet` template params `Hash`, `KeyEqual` | trait `KeyHasher<K>` (default `JoltHasher` = `JoltHash`), always `PartialEq`; `UnorderedMap<K, V, H>` / `UnorderedSet<K, H>` are type aliases of `HashTable` |
| `HashTable::size()` / `empty()` / `ClearAndKeepMemory()` | `len()` / `is_empty()` / `clear_and_keep_memory()` |
| `HashTable::find(k)` returning an iterator, `erase(iterator)` | `find(&k) -> Option<&KV>`, `find_mut`, `find_index(&k) -> Option<u32>` (the iterator's bucket index) + `get_at(i)`, `erase_at(i)`; `erase(&k) -> u32` |
| `HashTable::insert(v)` -> `pair<iterator, bool>` | `insert(v) -> (&mut KV, bool)` |
| `begin()`/`end()` iteration | `iter()` / `iter_mut()` (bucket order) |
| `UnorderedMap::operator [](k)` / `try_emplace(k, args...)` | `get_or_insert_default(k) -> &mut V` / `try_emplace(k, v) -> (&mut (K, V), bool)`; extra `get` / `get_mut` |
| `StaticArray<T, N>::push_back` / `emplace_back` / `pop_back` / `erase(it)` / `erase(b, e)` / `GetHash()` | `push` / `push` / `pop() -> Option<T>` / `erase(index)` / `erase_range(range)` / `JoltHash` |
| `BVec16::sLoadByte16`, `sEquals`, `GetTrues()` | `BVec16::load_byte16(&[u8])`, `equals`, `trues()` |
| `LinearCurve::GetValue(x)` / `GetMinX()` / `mPoints` | `value(x)` / `min_x()` / `points` (`LinearCurve::Point` is `linear_curve::Point`) |
| `UnitTestRandom` (`std::mt19937`) | `jolt_test_utils::UnitTestRandom` (`next_u32()`, `shuffle()`) |

## 4. Slices and status

Leaves first; each slice = Rust code + its ported tests, green before moving on. Milestones: **M1** core+math,
**M2** geometry, **M3** shapes + collision queries, **M4** single-threaded rigid body simulation (HelloWorld runs),
**M5** multithreading, **M6** determinism hashes match, **M7** character/ragdoll/vehicle/soft body.

| # | Slice | C++ sources | Depends on | Tests | Ported | Diff-tested | Notes |
|---|---|---|---|---|---|---|---|
| 1 | Core basics: `HashCombine`, `Color`, `StringTools`, `Result`, `NonCopyable`, `ScopeExit` | Core/* | — | ☑ | ☑ | ☐ | 14/14 tests. `NonCopyable` N/A (Rust moves); `Color::to_vec4` deferred to slice 4 |
| 2 | Sorting & containers: `InsertionSort`, `QuickSort`, `BinaryHeap`, `StaticArray`, `HashTable`, `UnorderedMap/Set`, `LinearCurve` | Core/* | 1 | ☑ | ☑ | ☐ | 27/27 tests. Scalar `BVec16` subset (`math::bvec16`) pulled forward for `HashTable`, rest in slice 4; `StaticArray` over `arrayvec`; `LinearCurve` binary state waits for slice 11. `ArrayTest`/`STLLocalAllocatorTest` N/A (Vec) |
| 3 | Math scalar: `Math.h`, `HalfFloat`, `Float2/3/4`, `Double3`, `Trigonometry`, `FindRoot` | Math/* | 1 | ☑ | ☑ | ☐ | 13/14 tests (`TestHalfFloatToFloat` + half->float checks of INF/NaN tests wait for slice 4). `Math.h` lives in `math` (mod.rs); `hash_table` now calls `count_trailing_zeros`/`count_leading_zeros`/`get_next_power_of_2`. Deferred to slice 4 (need `Vec4`/`UVec4`): `HalfFloatConversion::ToFloat`/`ToFloatFallback`, `Sin`/`Cos`/`Tan`/`ASin`/`ACos`/`ATan`/`ATan2` (only `ACosApproximate` ported) |
| 4 | Vectors: `Vec3`, `Vec4`, `UVec4`, `BVec16`, `DVec3`, `Swizzle` | Math/* | 3 | ☐ | ☐ | ☐ | W = Z invariant in `Vec3` |
| 5 | Matrices: `Mat44`, `DMat44`, `Quat`, `Matrix`, `Vector`, `DynMatrix`, `GaussianElimination`, `EigenValueSymmetric`, `Real` | Math/* | 4 | ☐ | ☐ | ☐ | **M1** |
| 6 | Geometry primitives: `AABox`, `OrientedBox`, `Plane`, `Sphere`, `Ellipse`, `Triangle`, `IndexedTriangle`, `Indexify`, `MortonCode`, `AABox4` | Geometry/* | 5 | ☐ | ☐ | ☐ | |
| 7 | Ray tests & closest points: `RayAABox`, `RayTriangle`, `RaySphere`, `RayCylinder`, `RayCapsule`, `ClosestPoint` | Geometry/* | 6 | ☐ | ☐ | ☐ | |
| 8 | GJK/EPA: `ConvexSupport`, `GJKClosestPoint`, `EPAPenetrationDepth`, `EPAConvexHullBuilder` | Geometry/* | 7 | ☐ | ☐ | ☐ | |
| 9 | Hulls: `ConvexHullBuilder`, `ConvexHullBuilder2D` | Geometry/* | 6 | ☐ | ☐ | ☐ | **M2** |
| 10 | `TriangleSplitter`, `AABBTree` builder + codecs | TriangleSplitter, AABBTree | 6 | ☐ | ☐ | ☐ | |
| 11 | Streams & state: `StreamIn/Out`, `StreamUtils`, `StateRecorder` | Core, Physics | 1 | ☐ | ☐ | ☐ | needed by `SaveBinaryState` |
| 12 | Collision basics: `SubShapeID`, `PhysicsMaterial`, `CollisionGroup`/`GroupFilter`, layers & filters, collectors, `Shape` trait, `ShapeCast`, `RayCast`, `CollisionDispatch` | Physics/Collision | 5–10 | ☐ | ☐ | ☐ | |
| 13 | Convex shapes: Sphere, Box, Capsule, TaperedCapsule, Cylinder, TaperedCylinder, ConvexHull, Triangle, Plane, Empty | Collision/Shape | 12 | ☐ | ☐ | ☐ | |
| 14 | Decorated & compound: Scaled, RotatedTranslated, OffsetCenterOfMass, StaticCompound, MutableCompound | Collision/Shape | 13 | ☐ | ☐ | ☐ | |
| 15 | Mesh & HeightField shapes, `ActiveEdges`, `CollideConvexVsTriangles`, `CastConvexVsTriangles` | Collision/Shape | 10, 13 | ☐ | ☐ | ☐ | **M3** |
| 16 | Broad phase: `QuadTree`, `BroadPhaseQuadTree`, `BroadPhaseBruteForce`, layer interfaces | Collision/BroadPhase | 12 | ☐ | ☐ | ☐ | |
| 17 | Bodies: `BodyID`, `Body`, `MotionProperties`, `BodyCreationSettings`, `BodyManager`, `BodyInterface`, locks | Physics/Body | 12, 16 | ☐ | ☐ | ☐ | |
| 18 | Narrow phase queries: `NarrowPhaseQuery`, `TransformedShape`, `InternalEdgeRemovingCollector`, `EstimateCollisionResponse` | Physics/Collision | 15, 17 | ☐ | ☐ | ☐ | |
| 19 | Solver: `ConstraintPart/*`, `ContactConstraintManager`, `IslandBuilder`, `LargeIslandSplitter` | Physics/Constraints, Physics | 17 | ☐ | ☐ | ☐ | |
| 20 | `PhysicsSystem::Update` single-threaded (`JobSystemSingleThreaded`), step listeners, sensors, CCD (`MotionQualityLinearCast`) | Physics | 18, 19 | ☐ | ☐ | ☐ | **M4**, port `HelloWorld` |
| 21 | Constraints: Fixed, Point, Distance, Hinge, Slider, Cone, SwingTwist, SixDOF, Path, Gear, RackAndPinion, Pulley | Physics/Constraints | 20 | ☐ | ☐ | ☐ | |
| 22 | Job system: `JobSystemThreadPool`, `JobSystemWithBarrier`, `Semaphore`, `LockFreeHashMap`, `FixedSizeFreeList`, parallel `Update` | Core, Physics | 20 | ☐ | ☐ | ☐ | **M5**, run tests under `--release` + loom/miri where possible |
| 23 | `PerformanceTest` port + determinism hashes | PerformanceTest | 21, 22 | ☐ | ☐ | ☐ | **M6** |
| 24 | Skeleton & Ragdoll | Skeleton, Physics/Ragdoll | 21 | ☐ | ☐ | ☐ | |
| 25 | Character, CharacterVirtual | Physics/Character | 20 | ☐ | ☐ | ☐ | |
| 26 | Vehicles | Physics/Vehicle | 21 | ☐ | ☐ | ☐ | |
| 27 | Soft bodies | Physics/SoftBody | 20 | ☐ | ☐ | ☐ | **M7** |
| 28 | Debug renderer trait | Renderer | 20 | ☐ | ☐ | ☐ | feature `debug-renderer` |
| — | Compute, Hair, ObjectStream | | | — | — | — | out of scope (see DEVIATIONS) |

## 5. Verification plan

- **Unit tests**: each C++ test file becomes a Rust file (`crates/<crate>/tests/<name>.rs` for public API, or an inline
  `#[cfg(test)] mod tests` for internals) whose header contains `//! Port of UnitTests/<Dir>/<File>.cpp`; each
  `TEST_CASE("TestFooBar")` becomes `#[test] fn test_foo_bar()` (`python tools/port/test_status.py --name TestFooBar`).
  Tests that do not apply in Rust stay unported and are listed in DEVIATIONS with the reason.
- **Precision**: run tests also with `--features double-precision` (CI does), and in `--release` for overflow/opt bugs.
- **C++ oracle (differential)**: dev machine toolchain is VS 2026 at `D:\Programs\VSC18` (not on PATH; run
  `VC\Auxiliary\Build\vcvars64.bat`). Build with `Build\cmake_vs2026_cl_cross_platform_deterministic.bat`, then
  `UnitTests` / `PerformanceTest`. Golden data for
  tricky slices (GJK/EPA, hull builder, AABB tree, solver) can be dumped from small C++ drivers into
  `crates/*/tests/fixtures/` and compared bit-exactly.
- **End-to-end**: `PerformanceTest -validate_hash` hashes from `.github/workflows/determinism_check.yml` (float and
  double) are the M6 acceptance criterion.
- **Extra**: `proptest` for math identities, `cargo miri test` for crates with `unsafe`, `criterion` benches vs C++
  `PerformanceTest` numbers (measured, never estimated).

## 6. Deviations

See [`DEVIATIONS.md`](DEVIATIONS.md).

## 7. Risks and open questions

- Multithreaded `PhysicsSystem::Update` is the hardest part to express safely (shared mutable bodies accessed by many
  jobs with ordering guaranteed by job dependencies). Plan: port sequential first, then introduce interior mutability
  per body (`UnsafeCell` behind a narrow, documented API) mirroring the C++ ownership rules.
- Bit-identical results require exact operation order everywhere (including SIMD horizontal sums such as the fused
  Jacobian dot products from upstream #2151). Keep the C++ evaluation order; never "simplify" float expressions.
- Upstream moves fast (several commits per week): re-sync regularly using the base commit in §1.
- Open: public API style — keep C++-like settings structs (`BodyCreationSettings`) or add builders? Default: keep
  settings structs with `Default`, add builders later.
