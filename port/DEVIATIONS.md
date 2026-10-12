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
| `BVec16` | SSE / NEON / scalar implementations; the scalar `sEquals` sets only the highest bit of an equal byte | scalar implementation only; `equals` sets all bits of an equal byte like `_mm_cmpeq_epi8` (SSE) | all implementations give the same `GetTrues()` / `TestAnyTrue()` / `TestAllTrue()` | planned |
| `LinearCurve` | `ObjectStream` serializable, `SaveBinaryState` / `RestoreBinaryState` | serialization not ported; binary state follows with `StreamIn/Out` (slice 11) | reflection not ported; streams not yet ported | planned |
| `UnitTestRandom` in tests | `std::shuffle` (implementation defined) | Fisher-Yates over the ported `mt19937` | `std::shuffle` differs between standard libraries anyway | planned |
| `Float2`/`Float3`/`Float4`/`Double3` | default constructor leaves them uninitialized; `operator []` takes `int`, only `JPH_ASSERT`s the range | `Default` = all zero; `Index<usize>` panics when out of range | no uninitialized values; memory safety | planned |
| `MulRSqrtApproximate` | `_mm_rsqrt_ss` / `vrsqrte_f32` / `vfrsqrt7` + Newton-Raphson unless `CROSS_PLATFORM_DETERMINISTIC` | always `inA / Sqrt(inB)` (the deterministic path) | determinism; scalar implementation first | planned |
| `Sqrt` | `_mm_sqrt_ss` / `vsqrt_f32` / `fsqrt.s` | `f32::sqrt` / `f64::sqrt` | IEEE square root is correctly rounded, same bits | planned |
| `CountTrailingZeros` / `CountLeadingZeros` / `CountBits` | per-compiler intrinsics | `u32::trailing_zeros` / `leading_zeros` / `count_ones` | identical results (32 for 0) | planned |
| `HalfFloatConversion::FromFloat` | F16C `_mm_cvtps_ph` when `JPH_USE_F16C` | always `FromFloatFallback` | same bits (verified against `_mm_cvtps_ph` in `test_float_to_half_float`) | planned |
| `HalfFloatConversion` rounding mode | template parameter `int RoundingMode` | `RoundingMode` enum argument | no stable const generics over enums | planned |
| `FindRoot` | out-params untouched when there is no root | returns `(count, x1, x2)` with zeros when there is no root | out-params become return values | planned |
| `GetNextPowerOf2` | undefined behavior (`1 << 32`) for values > 2^31 | panics in debug builds (shift overflow), wraps in release | Rust shift semantics | planned |
| `Vec3`/`Vec4`/`UVec4`/`DVec3` implementation | SSE / AVX / NEON / RVV / scalar implementations selected at compile time | one scalar implementation (no `std::arch` backends yet); IEEE add/sub/mul/div/sqrt give the same bits as SIMD and horizontal sums keep the vectorized bracketing | correctness first, SIMD later behind `cfg` with bit-identical results | planned |
| Vector edge cases where the C++ scalar fallback differs from SSE | `std::min`/`std::max` (`b < a ? b : a`); `abs`; `uint32(float)` in `ToInt` (UB for negative / large values); `(float)uint32` in `UVec4::ToFloat`; `DVec3::Dot` summed as `0 + x + y + z` | follows the x86-64 SSE/AVX build (the platform of the cross platform deterministic reference): `min`/`max` like `_mm_min_ps`/`_mm_max_ps` (`a < b ? a : b`, differs for NaN and +0/-0); `abs` = `max(0 - v, v)` (keeps -0); `to_int` like `_mm_cvttps_epi32` (signed truncation, out of range / NaN -> 0x80000000); `UVec4::to_float` converts signed (`_mm_cvtepi32_ps`); `DVec3::dot` = `(x + y) + z` like all SIMD paths | bit-identical with the reference build; only NaN, +0/-0 and out of range inputs are affected (NEON differs from SSE there too) | planned |
| `Vec3` / `DVec3` W component | W = Z only maintained with `JPH_FLOATING_POINT_EXCEPTIONS_ENABLED` (otherwise e.g. `sLoadFloat3Unsafe` loads garbage into W) | W = Z always; `load_float3_unsafe` / `load_double3_unsafe` read only 3 values | no reads past the end of a `Float3`; X, Y, Z unaffected | planned |
| `Vec4::Tan` / `Vec4::ATan` | add `FLT_MIN` to the divisor when `JPH_FLOATING_POINT_EXCEPTIONS_ENABLED` (MSVC Debug/Release builds) | never added (like GCC/Clang and MSVC Distribution builds) | Rust has no FP exceptions; only changes results for divisors below ~1e-31 | planned |
| `sGatherFloat4<Scale>` / `sGatherInt4<Scale>` | any byte `Scale`, raw pointer base | `SCALE` must be a multiple of 4, base is a slice (bounds checked) | memory safety | planned |
| `HalfFloatConversion::ToFloat` | F16C `_mm_cvtph_ps` when `JPH_USE_F16C` | always `ToFloatFallback` | same bits (verified against `_mm_cvtph_ps` in `test_half_float_to_float`) | planned |
| `Vec3::sRandom` / `std::uniform_real_distribution` in tests | template over any UniformRandomBitGenerator; the distribution algorithm is implementation defined | trait `RandomGenerator`; tests use `UnitTestRandom::uniform_f32` (libstdc++ algorithm) and draw function arguments left to right | Rust has no std random distributions | planned |

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
| `Math/FindRoot.h` `FindRoot<double>` | calls `DifferenceOfProducts`, which only has a `float` overload, so the discriminant of a `double` equation is computed in `float` precision | yes (`Float::difference_of_products` for `f64` narrows to `f32`) |
| `UnitTests/Math/MathTests.cpp` `TestIsPowerOf2` | loop checks `IsPowerOf2(uint64(1) << 1)` instead of `<< i` | yes |
