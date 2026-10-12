//! Port of `Jolt/Math`: vectors, matrices, quaternions and numeric helpers.
//!
//! One module per C++ header (`Vec3.h` + `Vec3.inl` -> `vec3`). The free functions of `Jolt/Math/Math.h` live directly
//! in this module (a `math::math` module would only add noise).

use std::ops::{BitAnd, Mul, Sub};

pub mod bvec16;
pub mod double3;
pub mod dvec3;
pub mod find_root;
pub mod float2;
pub mod float3;
pub mod float4;
pub mod half_float;
pub mod swizzle;
pub mod trigonometry;
pub mod uvec4;
pub mod vec3;
pub mod vec4;

pub use bvec16::BVec16;
pub use double3::Double3;
pub use dvec3::DVec3;
pub use find_root::find_root;
pub use float2::Float2;
pub use float3::{Float3, VertexList};
pub use float4::Float4;
pub use half_float::HalfFloat;
pub use swizzle::{SWIZZLE_UNUSED, SWIZZLE_W, SWIZZLE_X, SWIZZLE_Y, SWIZZLE_Z};
pub use trigonometry::{acos, acos_approximate, asin, atan, atan2, cos, sin, tan};
pub use uvec4::UVec4;
pub use vec3::Vec3;
pub use vec4::Vec4;

/// The constant \f$\pi\f$
/// (C++ `3.14159265358979323846f`, the same `f32` as `std::f32::consts::PI`)
pub const JPH_PI: f32 = std::f32::consts::PI;

/// A large floating point value which, when squared, is still much smaller than FLT_MAX
pub const LARGE_FLOAT: f32 = 1.0e15;

/// Floating point types (`float` and `double`) used by the templated helpers of `Math.h`
pub trait Float: Copy + PartialOrd + Signed {
    /// Take the square root of a value (`Sqrt`)
    fn sqrt(self) -> Self;

    /// `DifferenceOfProducts` as called from a template with this type: there is only a `float` overload in C++, so
    /// `double` arguments are narrowed to `float` (and the result widened again).
    fn difference_of_products(a: Self, b: Self, c: Self, d: Self) -> Self;
}

impl Float for f32 {
    #[inline]
    fn sqrt(self) -> Self {
        // IEEE square root is correctly rounded, so this matches _mm_sqrt_ss / vsqrt_f32 / fsqrt.s
        f32::sqrt(self)
    }

    #[inline]
    fn difference_of_products(a: Self, b: Self, c: Self, d: Self) -> Self {
        difference_of_products(a, b, c, d)
    }
}

impl Float for f64 {
    #[inline]
    fn sqrt(self) -> Self {
        // IEEE square root is correctly rounded, so this matches _mm_sqrt_sd / vsqrt_f64 / fsqrt.d
        f64::sqrt(self)
    }

    #[inline]
    fn difference_of_products(a: Self, b: Self, c: Self, d: Self) -> Self {
        difference_of_products(a as f32, b as f32, c as f32, d as f32) as f64
    }
}

/// Number types that have the constants needed by [`sign`]
pub trait Signed: Copy + PartialOrd {
    /// `T(0)`
    const ZERO: Self;
    /// `T(1)`
    const ONE: Self;
    /// `T(-1)`
    const MINUS_ONE: Self;
}

macro_rules! impl_signed {
    ($($type:ty: $zero:literal, $one:literal),*) => {
        $(impl Signed for $type {
            const ZERO: Self = $zero;
            const ONE: Self = $one;
            const MINUS_ONE: Self = -$one;
        })*
    };
}

impl_signed!(f32: 0.0, 1.0, f64: 0.0, 1.0, i8: 0, 1, i16: 0, 1, i32: 0, 1, i64: 0, 1, isize: 0, 1);

/// Port of `std::min` (as used through `using std::min` in `Core.h`): returns `a` unless `b < a`. Unlike `f32::min`
/// this does not special case NaN, which keeps the C++ semantics.
#[inline]
pub fn min<T: PartialOrd>(a: T, b: T) -> T {
    if b < a { b } else { a }
}

/// Port of `std::max` (as used through `using std::max` in `Core.h`): returns `a` unless `a < b`. Unlike `f32::max`
/// this does not special case NaN, which keeps the C++ semantics.
#[inline]
pub fn max<T: PartialOrd>(a: T, b: T) -> T {
    if a < b { b } else { a }
}

/// Convert a value from degrees to radians
#[inline]
pub const fn degrees_to_radians(v: f32) -> f32 {
    v * (JPH_PI / 180.0)
}

/// Convert a value from radians to degrees
#[inline]
pub const fn radians_to_degrees(v: f32) -> f32 {
    v * (180.0 / JPH_PI)
}

/// Convert angle in radians to the range \f$[-\pi, \pi]\f$
#[inline]
pub fn center_angle_around_zero(mut v: f32) -> f32 {
    // `while { body; condition } {}` is a do-while loop
    if v < -JPH_PI {
        while {
            v += 2.0 * JPH_PI;
            v < -JPH_PI
        } {}
    } else if v > JPH_PI {
        while {
            v -= 2.0 * JPH_PI;
            v > JPH_PI
        } {}
    }
    debug_assert!((-JPH_PI..=JPH_PI).contains(&v));
    v
}

/// Calculates inA * inB - inC * inD with higher accuracy when fused multiply add instructions are available.
/// If inA * inB and inC * inD are large, the subtraction can cause a large loss of precision when the result is small.
/// See: https://pharr.org/matt/blog/2019/11/03/difference-of-floats (or search for Kahan's algorithm)
///
/// The port never uses fused multiply add (see `port/DEVIATIONS.md`), so this is always the plain expression.
#[inline]
pub fn difference_of_products(a: f32, b: f32, c: f32, d: f32) -> f32 {
    a * b - c * d
}

/// Clamp a value between two values
#[inline]
pub fn clamp<T: PartialOrd>(v: T, min_value: T, max_value: T) -> T {
    min(max(v, min_value), max_value)
}

/// Square a value
#[inline]
pub fn square<T: Copy + Mul<Output = T>>(v: T) -> T {
    v * v
}

/// Take the square root of a float or double value
#[inline]
pub fn sqrt<T: Float>(v: T) -> T {
    v.sqrt()
}

/// Returns \f$inV^3\f$.
#[inline]
pub fn cubed<T: Copy + Mul<Output = T>>(v: T) -> T {
    v * v * v
}

/// Calculates approximate inA / Sqrt(inB).
/// Approximate reciprocal square roots are not bit identical between architectures,
/// so the fast paths are only used when cross platform determinism is not required.
/// Note that inB must be bigger or equal than FLT_MIN, otherwise the result can be Inf.
///
/// The port only has the exact path (`inA / Sqrt(inB)`) that the C++ cross platform deterministic build uses.
#[inline]
pub fn mul_rsqrt_approximate(a: f32, b: f32) -> f32 {
    debug_assert!(b >= f32::MIN_POSITIVE);

    a / sqrt(b)
}

/// Get the sign of a value
#[inline]
pub fn sign<T: Signed>(v: T) -> T {
    if v < T::ZERO { T::MINUS_ONE } else { T::ONE }
}

/// Check if inV is a power of 2
#[inline]
pub fn is_power_of_2<T>(v: T) -> bool
where
    T: Copy + PartialOrd + From<u8> + BitAnd<Output = T> + Sub<Output = T>,
{
    v > T::from(0) && (v & (v - T::from(1))) == T::from(0)
}

/// Integer types that can be aligned with [`align_up`] / [`is_aligned`] (C++ converts them through `uint64`)
pub trait AlignInt: Copy {
    /// `uint64(inV)`
    fn to_u64(self) -> u64;
    /// `T(inV)`
    fn from_u64(v: u64) -> Self;
}

macro_rules! impl_align_int {
    ($($type:ty),*) => {
        $(impl AlignInt for $type {
            #[inline]
            fn to_u64(self) -> u64 {
                self as u64
            }

            #[inline]
            fn from_u64(v: u64) -> Self {
                v as Self
            }
        })*
    };
}

impl_align_int!(u8, u16, u32, u64, usize, i8, i16, i32, i64, isize);

/// Align inV up to the next inAlignment bytes
#[inline]
pub fn align_up<T: AlignInt>(v: T, alignment: u64) -> T {
    debug_assert!(is_power_of_2(alignment));
    T::from_u64((v.to_u64().wrapping_add(alignment - 1)) & !(alignment - 1))
}

/// Check if inV is inAlignment aligned
#[inline]
pub fn is_aligned<T: AlignInt>(v: T, alignment: u64) -> bool {
    debug_assert!(is_power_of_2(alignment));
    (v.to_u64() & (alignment - 1)) == 0
}

/// Compute number of trailing zero bits (how many low bits are zero)
///
/// Every C++ implementation (`tzcnt`, `_BitScanForward`, `__builtin_ctz`) returns 32 for 0, like `u32::trailing_zeros`.
#[inline]
pub const fn count_trailing_zeros(value: u32) -> u32 {
    value.trailing_zeros()
}

/// Compute the number of leading zero bits (how many high bits are zero)
///
/// Every C++ implementation (`lzcnt`, `_BitScanReverse`, `__builtin_clz`) returns 32 for 0, like `u32::leading_zeros`.
#[inline]
pub const fn count_leading_zeros(value: u32) -> u32 {
    value.leading_zeros()
}

/// Count the number of 1 bits in a value
#[inline]
pub const fn count_bits(value: u32) -> u32 {
    value.count_ones()
}

/// Get the next higher power of 2 of a value, or the value itself if the value is already a power of 2
#[inline]
pub const fn get_next_power_of_2(value: u32) -> u32 {
    if value <= 1 { 1 } else { 1u32 << (32 - count_leading_zeros(value - 1)) }
}
