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
| `StringFormat` | `printf`-style format string | `string_format!` with Rust `format!` syntax (still trimmed to 1023 bytes) | no varargs in Rust | planned |
| `ConvertToString` | `std::ostream <<` formatting | `Display` formatting (floats can format differently, e.g. `1e+20` vs `100000000000000000000`) | idiomatic; debug output only | planned |
| `StringToVector` | appends to an out vector, optional clear | returns a new `Vec` | out-params become return values | planned |
| `Result<T>` | own class with an empty (`Invalid`) state | `std::result::Result<T, JoltError>`, empty state as `Option` | idiomatic | planned |
| `Color` | default constructor leaves it uninitialized | `Default` = all zero | Rust has no uninitialized values | planned |
| `NonCopyable` | base class deleting copy | not ported: Rust types are move-only unless `Clone`/`Copy` | language feature | planned |
| `QuickSort` | copies the pivot value | tracks the index of the pivot element through swaps (same comparisons, same result), so `T` needs no `Clone` | no copy bound on sorted types | planned |
| `HashTable` storage | raw allocation of `KeyValue[mMaxSize]` + control bytes, placement new | `Vec<Option<KV>>` + `Vec<u8>` control bytes; bucket layout, probing, growth, tombstones, rehash and iteration order unchanged | no `unsafe` | planned |
| `HashTable` API | STL-like iterators, `KeyEqual` template parameter, moved-from table is empty | Rust iterators, bucket indices (`find_index`/`erase_at`), key equality is `PartialEq`, moves via `std::mem::take` | idiomatic | planned |
| `HashTable` grow overflow | `JPH_ASSERT` and returns `false` (then constructs at an uninitialized index) | panics | avoid an invalid state; needs > 2^31 buckets | planned |
| `HashTable::rehash` | on a never-allocated table writes through a null control pointer | no-op | memory safety | planned |
| `UnorderedMap::try_emplace` | constructs the value only when inserted | value is passed (and dropped if not inserted) | no variadic in-place construction | planned |
| `StaticArray` | own union buffer, `uint` sizes | wrapper around `arrayvec::ArrayVec`, `usize` sizes, derefs to a slice | no `unsafe` | planned |
| `BVec16` | SSE / NEON / scalar implementations | scalar implementation only, subset needed by `HashTable` (rest in slice 4) | all implementations give the same `GetTrues()` | planned |
| `LinearCurve` | `ObjectStream` serializable, `SaveBinaryState` / `RestoreBinaryState` | serialization not ported; binary state follows with `StreamIn/Out` (slice 11) | reflection not ported; streams not yet ported | planned |
| `UnitTestRandom` in tests | `std::shuffle` (implementation defined) | Fisher-Yates over the ported `mt19937` | `std::shuffle` differs between standard libraries anyway | planned |

## Unported tests

| C++ test | Reason |
|---|---|
| `UnitTests/Core/ArrayTest.cpp` (all) | `Array<T>` is `Vec<T>` (see Containers) |
| `UnitTests/Core/STLLocalAllocatorTest.cpp` (all) | allocators not ported (see Memory) |

## Bugs found in the original

| Location | Description | Ported as-is? |
|---|---|---|
| `Core/HashCombine.h` `HashString` | `uint64(*c)` on a `char`: sign-extends bytes >= 0x80 where `char` is signed (x86, MSVC) but not where it is unsigned (ARM Linux), so non-ASCII strings hash differently per platform and differently from `HashBytes` | yes, using the signed (x86) behavior |
| `Core/StringTools.cpp` `StringReplace` | never terminates for an empty search string | yes, plus a `debug_assert!` |
