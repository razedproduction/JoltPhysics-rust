//! Port of `Jolt/Math/Vec4.h` + `Vec4.inl`.
//!
//! Scalar implementation. Where the C++ scalar fallback and the x86-64 SSE implementation (the platform of the
//! cross platform deterministic reference build) differ, this follows SSE; see `port/DEVIATIONS.md`. Negation is
//! `0 - v` like in the C++ cross platform deterministic build.

// Keep the literals of the C++ source (e.g. 0.70710678f, 0.6366197723675814f) instead of the std constants
#![allow(clippy::approx_constant)]

use std::fmt;
use std::ops::{Add, AddAssign, Div, DivAssign, Index, IndexMut, Mul, MulAssign, Neg, Sub, SubAssign};

use super::swizzle::{SWIZZLE_UNUSED, SWIZZLE_W, SWIZZLE_X, SWIZZLE_Y, SWIZZLE_Z};
use super::{Float4, JPH_PI, UVec4, Vec3, count_trailing_zeros};

/// `_mm_min_ps` semantics: returns `a` if `a < b`, else `b` (differs from `std::min` for NaN and +0 / -0)
#[inline]
pub(crate) fn simd_min<T: PartialOrd>(a: T, b: T) -> T {
    if a < b { a } else { b }
}

/// `_mm_max_ps` semantics: returns `a` if `a > b`, else `b` (differs from `std::max` for NaN and +0 / -0)
#[inline]
pub(crate) fn simd_max<T: PartialOrd>(a: T, b: T) -> T {
    if a > b { a } else { b }
}

/// `_mm_cvttps_epi32` semantics: truncate to a signed int, values out of range (and NaN) become 0x80000000
#[inline]
pub(crate) fn simd_float_to_int(v: f32) -> u32 {
    if (-2147483648.0..2147483648.0).contains(&v) { v as i32 as u32 } else { 0x80000000 }
}

/// `_mm_max_ps(_mm_sub_ps(_mm_setzero_ps(), v), v)`: absolute value as the SSE implementation computes it (keeps -0)
#[inline]
pub(crate) fn simd_abs(v: f32) -> f32 {
    simd_max(0.0 - v, v)
}

/// 4 component vector of floats
#[derive(Clone, Copy, Debug, Default)]
#[repr(C, align(16))]
pub struct Vec4 {
    pub(crate) f32s: [f32; 4],
}

impl Vec4 {
    /// Create a vector from 4 components
    #[inline]
    pub const fn new(x: f32, y: f32, z: f32, w: f32) -> Self {
        Self { f32s: [x, y, z, w] }
    }

    /// Create a vector from an array of 4 components
    #[inline]
    pub const fn from_array(v: [f32; 4]) -> Self {
        Self { f32s: v }
    }

    /// Get the 4 components as an array
    #[inline]
    pub const fn to_array(self) -> [f32; 4] {
        self.f32s
    }

    /// Construct from a Vec3 (C++ `explicit Vec4(Vec3Arg)`). WARNING: W component undefined in C++, here it is the Z
    /// component (Vec3 keeps W = Z).
    #[inline]
    pub const fn from_vec3(v: Vec3) -> Self {
        Self { f32s: v.f32s }
    }

    /// Construct from a Vec3 and a W component
    #[inline]
    pub const fn from_vec3_w(v: Vec3, w: f32) -> Self {
        Self::new(v.f32s[0], v.f32s[1], v.f32s[2], w)
    }

    /// Swizzle the elements in inV
    #[inline]
    pub fn swizzle<const X: u32, const Y: u32, const Z: u32, const W: u32>(self) -> Self {
        const {
            assert!(X <= 3, "SwizzleX template parameter out of range");
            assert!(Y <= 3, "SwizzleY template parameter out of range");
            assert!(Z <= 3, "SwizzleZ template parameter out of range");
            assert!(W <= 3, "SwizzleW template parameter out of range");
        }

        Self::new(self.f32s[X as usize], self.f32s[Y as usize], self.f32s[Z as usize], self.f32s[W as usize])
    }

    /// Vector with all zeros
    #[inline]
    pub const fn zero() -> Self {
        Self::new(0.0, 0.0, 0.0, 0.0)
    }

    /// Vector with all ones
    #[inline]
    pub const fn one() -> Self {
        Self::replicate(1.0)
    }

    /// Vector with all NaN's
    #[inline]
    pub const fn nan() -> Self {
        Self::replicate(f32::NAN)
    }

    /// Replicate inV across all components
    #[inline]
    pub const fn replicate(v: f32) -> Self {
        Self::new(v, v, v, v)
    }

    /// Load 4 floats from memory
    #[inline]
    pub const fn load_float4(v: &Float4) -> Self {
        Self::new(v.x, v.y, v.z, v.w)
    }

    /// Load 4 floats from memory, 16 bytes aligned
    #[inline]
    pub const fn load_float4_aligned(v: &Float4) -> Self {
        Self::new(v.x, v.y, v.z, v.w)
    }

    /// Gather 4 floats from memory at inBase + inOffsets[i] * Scale
    ///
    /// `SCALE` is in bytes like in C++ and must be a multiple of 4 (the size of a `f32`).
    #[inline]
    pub fn gather_float4<const SCALE: usize>(base: &[f32], offsets: UVec4) -> Self {
        const { assert!(SCALE % 4 == 0, "Scale must be a multiple of sizeof(float)") };
        let get = |offset: u32| base[offset as usize * SCALE / 4];
        Self::new(get(offsets.x()), get(offsets.y()), get(offsets.z()), get(offsets.w()))
    }

    /// Return the minimum value of each of the components
    #[inline]
    pub fn min(v1: Self, v2: Self) -> Self {
        Self::from_array(std::array::from_fn(|i| simd_min(v1.f32s[i], v2.f32s[i])))
    }

    /// Return the maximum of each of the components
    #[inline]
    pub fn max(v1: Self, v2: Self) -> Self {
        Self::from_array(std::array::from_fn(|i| simd_max(v1.f32s[i], v2.f32s[i])))
    }

    /// Clamp a vector between min and max (component wise)
    #[inline]
    pub fn clamp(v: Self, min: Self, max: Self) -> Self {
        Self::max(Self::min(v, max), min)
    }

    #[inline]
    fn compare(v1: Self, v2: Self, cmp: impl Fn(f32, f32) -> bool) -> UVec4 {
        UVec4::from_array(std::array::from_fn(|i| if cmp(v1.f32s[i], v2.f32s[i]) { 0xffffffff } else { 0 }))
    }

    /// Equals (component wise)
    #[inline]
    pub fn equals(v1: Self, v2: Self) -> UVec4 {
        Self::compare(v1, v2, |a, b| a == b)
    }

    /// Less than (component wise)
    #[inline]
    pub fn less(v1: Self, v2: Self) -> UVec4 {
        Self::compare(v1, v2, |a, b| a < b)
    }

    /// Less than or equal (component wise)
    #[inline]
    pub fn less_or_equal(v1: Self, v2: Self) -> UVec4 {
        Self::compare(v1, v2, |a, b| a <= b)
    }

    /// Greater than (component wise)
    #[inline]
    pub fn greater(v1: Self, v2: Self) -> UVec4 {
        Self::compare(v1, v2, |a, b| a > b)
    }

    /// Greater than or equal (component wise)
    #[inline]
    pub fn greater_or_equal(v1: Self, v2: Self) -> UVec4 {
        Self::compare(v1, v2, |a, b| a >= b)
    }

    /// Calculates inMul1 * inMul2 + inAdd
    ///
    /// Never fused (see `port/DEVIATIONS.md`).
    #[inline]
    pub fn fused_multiply_add(mul1: Self, mul2: Self, add: Self) -> Self {
        mul1 * mul2 + add
    }

    /// Component wise select, returns inNotSet when highest bit of inControl = 0 and inSet when highest bit of inControl = 1
    #[inline]
    pub fn select(not_set: Self, set: Self, control: UVec4) -> Self {
        Self::from_array(std::array::from_fn(|i| {
            if (control.u32s[i] & 0x80000000) != 0 { set.f32s[i] } else { not_set.f32s[i] }
        }))
    }

    /// Logical or (component wise)
    #[inline]
    pub fn or(v1: Self, v2: Self) -> Self {
        UVec4::or(v1.reinterpret_as_int(), v2.reinterpret_as_int()).reinterpret_as_float()
    }

    /// Logical xor (component wise)
    #[inline]
    pub fn xor(v1: Self, v2: Self) -> Self {
        UVec4::xor(v1.reinterpret_as_int(), v2.reinterpret_as_int()).reinterpret_as_float()
    }

    /// Logical and (component wise)
    #[inline]
    pub fn and(v1: Self, v2: Self) -> Self {
        UVec4::and(v1.reinterpret_as_int(), v2.reinterpret_as_int()).reinterpret_as_float()
    }

    /// Sort the four elements of ioValue and sort ioIndex at the same time.
    /// Based on a sorting network: http://en.wikipedia.org/wiki/Sorting_network
    #[inline]
    pub fn sort4(value: &mut Self, index: &mut UVec4) {
        // Pass 1, test 1st vs 3rd, 2nd vs 4th
        let v1 = value.swizzle::<SWIZZLE_Z, SWIZZLE_W, SWIZZLE_X, SWIZZLE_Y>();
        let i1 = index.swizzle::<SWIZZLE_Z, SWIZZLE_W, SWIZZLE_X, SWIZZLE_Y>();
        let c1 = Self::less(*value, v1).swizzle::<SWIZZLE_Z, SWIZZLE_W, SWIZZLE_Z, SWIZZLE_W>();
        *value = Self::select(*value, v1, c1);
        *index = UVec4::select(*index, i1, c1);

        // Pass 2, test 1st vs 2nd, 3rd vs 4th
        let v2 = value.swizzle::<SWIZZLE_Y, SWIZZLE_X, SWIZZLE_W, SWIZZLE_Z>();
        let i2 = index.swizzle::<SWIZZLE_Y, SWIZZLE_X, SWIZZLE_W, SWIZZLE_Z>();
        let c2 = Self::less(*value, v2).swizzle::<SWIZZLE_Y, SWIZZLE_Y, SWIZZLE_W, SWIZZLE_W>();
        *value = Self::select(*value, v2, c2);
        *index = UVec4::select(*index, i2, c2);

        // Pass 3, test 2nd vs 3rd component
        let v3 = value.swizzle::<SWIZZLE_X, SWIZZLE_Z, SWIZZLE_Y, SWIZZLE_W>();
        let i3 = index.swizzle::<SWIZZLE_X, SWIZZLE_Z, SWIZZLE_Y, SWIZZLE_W>();
        let c3 = Self::less(*value, v3).swizzle::<SWIZZLE_X, SWIZZLE_Z, SWIZZLE_Z, SWIZZLE_W>();
        *value = Self::select(*value, v3, c3);
        *index = UVec4::select(*index, i3, c3);
    }

    /// Reverse sort the four elements of ioValue (highest first) and sort ioIndex at the same time.
    /// Based on a sorting network: http://en.wikipedia.org/wiki/Sorting_network
    #[inline]
    pub fn sort4_reverse(value: &mut Self, index: &mut UVec4) {
        // Pass 1, test 1st vs 3rd, 2nd vs 4th
        let v1 = value.swizzle::<SWIZZLE_Z, SWIZZLE_W, SWIZZLE_X, SWIZZLE_Y>();
        let i1 = index.swizzle::<SWIZZLE_Z, SWIZZLE_W, SWIZZLE_X, SWIZZLE_Y>();
        let c1 = Self::greater(*value, v1).swizzle::<SWIZZLE_Z, SWIZZLE_W, SWIZZLE_Z, SWIZZLE_W>();
        *value = Self::select(*value, v1, c1);
        *index = UVec4::select(*index, i1, c1);

        // Pass 2, test 1st vs 2nd, 3rd vs 4th
        let v2 = value.swizzle::<SWIZZLE_Y, SWIZZLE_X, SWIZZLE_W, SWIZZLE_Z>();
        let i2 = index.swizzle::<SWIZZLE_Y, SWIZZLE_X, SWIZZLE_W, SWIZZLE_Z>();
        let c2 = Self::greater(*value, v2).swizzle::<SWIZZLE_Y, SWIZZLE_Y, SWIZZLE_W, SWIZZLE_W>();
        *value = Self::select(*value, v2, c2);
        *index = UVec4::select(*index, i2, c2);

        // Pass 3, test 2nd vs 3rd component
        let v3 = value.swizzle::<SWIZZLE_X, SWIZZLE_Z, SWIZZLE_Y, SWIZZLE_W>();
        let i3 = index.swizzle::<SWIZZLE_X, SWIZZLE_Z, SWIZZLE_Y, SWIZZLE_W>();
        let c3 = Self::greater(*value, v3).swizzle::<SWIZZLE_X, SWIZZLE_Z, SWIZZLE_Z, SWIZZLE_W>();
        *value = Self::select(*value, v3, c3);
        *index = UVec4::select(*index, i3, c3);
    }

    /// Get individual components
    #[inline]
    pub const fn x(self) -> f32 {
        self.f32s[0]
    }

    #[inline]
    pub const fn y(self) -> f32 {
        self.f32s[1]
    }

    #[inline]
    pub const fn z(self) -> f32 {
        self.f32s[2]
    }

    #[inline]
    pub const fn w(self) -> f32 {
        self.f32s[3]
    }

    /// Set individual components
    #[inline]
    pub fn set_x(&mut self, x: f32) {
        self.f32s[0] = x;
    }

    #[inline]
    pub fn set_y(&mut self, y: f32) {
        self.f32s[1] = y;
    }

    #[inline]
    pub fn set_z(&mut self, z: f32) {
        self.f32s[2] = z;
    }

    #[inline]
    pub fn set_w(&mut self, w: f32) {
        self.f32s[3] = w;
    }

    /// Set all components
    #[inline]
    pub fn set(&mut self, x: f32, y: f32, z: f32, w: f32) {
        *self = Self::new(x, y, z, w);
    }

    /// Test if two vectors are close
    #[inline]
    pub fn is_close(self, v2: Self, max_dist_sq: f32) -> bool {
        (v2 - self).length_sq() <= max_dist_sq
    }

    /// Test if two vectors are close (with the default inMaxDistSq = 1.0e-12f)
    #[inline]
    pub fn is_close_default(self, v2: Self) -> bool {
        self.is_close(v2, 1.0e-12)
    }

    /// Test if vector is near zero
    #[inline]
    pub fn is_near_zero(self, max_dist_sq: f32) -> bool {
        self.length_sq() <= max_dist_sq
    }

    /// Test if vector is near zero (with the default inMaxDistSq = 1.0e-12f)
    #[inline]
    pub fn is_near_zero_default(self) -> bool {
        self.is_near_zero(1.0e-12)
    }

    /// Test if length^2 of this vector is within the range [1 - inTolerance, 1 + inTolerance]
    #[inline]
    pub fn is_normalized(self, tolerance: f32) -> bool {
        (self.length_sq() - 1.0).abs() <= tolerance
    }

    /// Test if length^2 of this vector is within the range [1 - inTolerance, 1 + inTolerance] (with the default
    /// inTolerance = 1.0e-6f)
    #[inline]
    pub fn is_normalized_default(self) -> bool {
        self.is_normalized(1.0e-6)
    }

    /// Test if vector contains NaN elements
    #[inline]
    pub fn is_nan(self) -> bool {
        self.f32s.iter().any(|v| v.is_nan())
    }

    /// Replicate the X component to all components
    #[inline]
    pub const fn splat_x(self) -> Self {
        Self::replicate(self.f32s[0])
    }

    /// Replicate the Y component to all components
    #[inline]
    pub const fn splat_y(self) -> Self {
        Self::replicate(self.f32s[1])
    }

    /// Replicate the Z component to all components
    #[inline]
    pub const fn splat_z(self) -> Self {
        Self::replicate(self.f32s[2])
    }

    /// Replicate the W component to all components
    #[inline]
    pub const fn splat_w(self) -> Self {
        Self::replicate(self.f32s[3])
    }

    /// Replicate the X component to all components
    #[inline]
    pub const fn splat_x3(self) -> Vec3 {
        Vec3::replicate(self.f32s[0])
    }

    /// Replicate the Y component to all components
    #[inline]
    pub const fn splat_y3(self) -> Vec3 {
        Vec3::replicate(self.f32s[1])
    }

    /// Replicate the Z component to all components
    #[inline]
    pub const fn splat_z3(self) -> Vec3 {
        Vec3::replicate(self.f32s[2])
    }

    /// Replicate the W component to all components
    #[inline]
    pub const fn splat_w3(self) -> Vec3 {
        Vec3::replicate(self.f32s[3])
    }

    /// Get index of component with lowest value
    #[inline]
    pub fn lowest_component_index(self) -> i32 {
        // Get the minimum value in all 4 components
        let mut value = Self::min(self, self.swizzle::<SWIZZLE_Y, SWIZZLE_X, SWIZZLE_W, SWIZZLE_Z>());
        value = Self::min(value, value.swizzle::<SWIZZLE_Z, SWIZZLE_W, SWIZZLE_X, SWIZZLE_Y>());

        // Compare with the original vector to find which component is equal to the minimum value
        count_trailing_zeros(Self::equals(self, value).trues() as u32) as i32
    }

    /// Get index of component with highest value
    #[inline]
    pub fn highest_component_index(self) -> i32 {
        // Get the maximum value in all 4 components
        let mut value = Self::max(self, self.swizzle::<SWIZZLE_Y, SWIZZLE_X, SWIZZLE_W, SWIZZLE_Z>());
        value = Self::max(value, value.swizzle::<SWIZZLE_Z, SWIZZLE_W, SWIZZLE_X, SWIZZLE_Y>());

        // Compare with the original vector to find which component is equal to the maximum value
        count_trailing_zeros(Self::equals(self, value).trues() as u32) as i32
    }

    /// Return the absolute value of each of the components
    #[inline]
    pub fn abs(self) -> Self {
        Self::from_array(self.f32s.map(simd_abs))
    }

    /// Reciprocal vector (1 / value) for each of the components
    #[inline]
    pub fn reciprocal(self) -> Self {
        Self::one() / self
    }

    /// Calculates inA * inB - inC * inD with more precision when FMA instructions are available. See DifferenceOfProducts.
    ///
    /// The port never uses FMA, so this is always the plain expression.
    #[inline]
    pub fn difference_of_products(a: Self, b: Self, c: Self, d: Self) -> Self {
        a * b - c * d
    }

    /// Dot product, returns the dot product in X, Y, Z and W components
    #[inline]
    pub fn dot_v(self, v2: Self) -> Self {
        Self::replicate(self.dot(v2))
    }

    /// Dot product
    #[inline]
    pub fn dot(self, v2: Self) -> f32 {
        (self * v2).reduce_sum()
    }

    /// Squared length of vector
    #[inline]
    pub fn length_sq(self) -> f32 {
        self.dot(self)
    }

    /// Length of vector
    #[inline]
    pub fn length(self) -> f32 {
        super::sqrt(self.length_sq())
    }

    /// Normalize vector
    #[inline]
    pub fn normalized(self) -> Self {
        self / self.length()
    }

    /// Store 4 floats to memory
    #[inline]
    pub const fn store_float4(self) -> Float4 {
        Float4::new(self.f32s[0], self.f32s[1], self.f32s[2], self.f32s[3])
    }

    /// Convert each component from a float to an int
    ///
    /// Like `_mm_cvttps_epi32` (the SSE implementation) values are truncated to signed integers, values that are out
    /// of range become 0x80000000.
    #[inline]
    pub fn to_int(self) -> UVec4 {
        UVec4::from_array(self.f32s.map(simd_float_to_int))
    }

    /// Reinterpret Vec4 as a UVec4 (doesn't change the bits)
    #[inline]
    pub fn reinterpret_as_int(self) -> UVec4 {
        UVec4::from_array(self.f32s.map(f32::to_bits))
    }

    /// Store if X is negative in bit 0, Y in bit 1, Z in bit 2 and W in bit 3
    #[inline]
    pub fn sign_bits(self) -> i32 {
        (if self.f32s[0].is_sign_negative() { 1 } else { 0 })
            | (if self.f32s[1].is_sign_negative() { 2 } else { 0 })
            | (if self.f32s[2].is_sign_negative() { 4 } else { 0 })
            | (if self.f32s[3].is_sign_negative() { 8 } else { 0 })
    }

    /// Get the minimum of X, Y, Z and W
    #[inline]
    pub fn reduce_min(self) -> f32 {
        let mut v = Self::min(self, self.swizzle::<SWIZZLE_Y, SWIZZLE_UNUSED, SWIZZLE_W, SWIZZLE_UNUSED>());
        v = Self::min(v, v.swizzle::<SWIZZLE_Z, SWIZZLE_UNUSED, SWIZZLE_UNUSED, SWIZZLE_UNUSED>());
        v.x()
    }

    /// Get the maximum of X, Y, Z and W
    #[inline]
    pub fn reduce_max(self) -> f32 {
        let mut v = Self::max(self, self.swizzle::<SWIZZLE_Y, SWIZZLE_UNUSED, SWIZZLE_W, SWIZZLE_UNUSED>());
        v = Self::max(v, v.swizzle::<SWIZZLE_Z, SWIZZLE_UNUSED, SWIZZLE_UNUSED, SWIZZLE_UNUSED>());
        v.x()
    }

    /// Sum X, Y, Z and W
    #[inline]
    pub fn reduce_sum(self) -> f32 {
        // Brackets placed so that the order is consistent with the vectorized version
        (self.f32s[0] + self.f32s[1]) + (self.f32s[2] + self.f32s[3])
    }

    /// Component wise square root
    #[inline]
    pub fn sqrt(self) -> Self {
        Self::from_array(self.f32s.map(super::sqrt))
    }

    /// Get vector that contains the sign of each element (returns 1.0f if positive, -1.0f if negative)
    #[inline]
    pub fn sign(self) -> Self {
        Self::from_array(self.f32s.map(|v| if v.is_sign_negative() { -1.0 } else { 1.0 }))
    }

    /// Flips the signs of the components, e.g. FlipSign<-1, 1, -1, 1>() will flip the signs of the X and Z components
    #[inline]
    pub fn flip_sign<const X: i32, const Y: i32, const Z: i32, const W: i32>(self) -> Self {
        const {
            assert!(X == 1 || X == -1, "X must be 1 or -1");
            assert!(Y == 1 || Y == -1, "Y must be 1 or -1");
            assert!(Z == 1 || Z == -1, "Z must be 1 or -1");
            assert!(W == 1 || W == -1, "W must be 1 or -1");
        }
        Self::xor(
            self,
            Self::new(
                if X > 0 { 0.0 } else { -0.0 },
                if Y > 0 { 0.0 } else { -0.0 },
                if Z > 0 { 0.0 } else { -0.0 },
                if W > 0 { 0.0 } else { -0.0 },
            ),
        )
    }

    /// Calculate the sine and cosine for each element of this vector (input in radians)
    ///
    /// Returns `(sin, cos)` (C++ out-parameters `outSin`, `outCos`).
    #[inline]
    pub fn sin_cos(self) -> (Self, Self) {
        // Implementation based on sinf.c from the cephes library, combines sinf and cosf in a single function, changes octants to quadrants and vectorizes it
        // Original implementation by Stephen L. Moshier (See: http://www.moshier.net/)

        // Make argument positive and remember sign for sin only since cos is symmetric around x (highest bit of a float is the sign bit)
        let mut sin_sign = UVec4::and(self.reinterpret_as_int(), UVec4::replicate(0x80000000));
        let mut x = Self::xor(self, sin_sign.reinterpret_as_float());

        // x / (PI / 2) rounded to nearest int gives us the quadrant closest to x
        let quadrant = (0.6366197723675814 * x + Self::replicate(0.5)).to_int();

        // Make x relative to the closest quadrant.
        // This does x = x - quadrant * PI / 2 using a two step Cody-Waite argument reduction.
        // This improves the accuracy of the result by avoiding loss of significant bits in the subtraction.
        // We start with x = x - quadrant * PI / 2, PI / 2 in hexadecimal notation is 0x3fc90fdb, we remove the lowest 16 bits to
        // get 0x3fc90000 (= 1.5703125) this means we can now multiply with a number of up to 2^16 without losing any bits.
        // This leaves us with: x = (x - quadrant * 1.5703125) - quadrant * (PI / 2 - 1.5703125).
        // PI / 2 - 1.5703125 in hexadecimal is 0x39fdaa22, stripping the lowest 12 bits we get 0x39fda000 (= 0.0004837512969970703125)
        // This leaves uw with: x = ((x - quadrant * 1.5703125) - quadrant * 0.0004837512969970703125) - quadrant * (PI / 2 - 1.5703125 - 0.0004837512969970703125)
        // See: https://stackoverflow.com/questions/42455143/sine-cosine-modular-extended-precision-arithmetic
        // After this we have x in the range [-PI / 4, PI / 4].
        let float_quadrant = quadrant.to_float();
        x = ((x - float_quadrant * 1.5703125) - float_quadrant * 0.0004837512969970703125)
            - float_quadrant * 7.549789948768648e-8;

        // Calculate x2 = x^2
        let x2 = x * x;

        // Taylor expansion:
        // Cos(x) = 1 - x^2/2! + x^4/4! - x^6/6! + x^8/8! + ... = (((x2/8!- 1/6!) * x2 + 1/4!) * x2 - 1/2!) * x2 + 1
        let taylor_cos = ((2.443315711809948e-5 * x2 - Self::replicate(1.388731625493765e-3)) * x2
            + Self::replicate(4.166664568298827e-2))
            * x2
            * x2
            - 0.5 * x2
            + Self::one();
        // Sin(x) = x - x^3/3! + x^5/5! - x^7/7! + ... = ((-x2/7! + 1/5!) * x2 - 1/3!) * x2 * x + x
        let taylor_sin = ((-1.9515295891e-4 * x2 + Self::replicate(8.3321608736e-3)) * x2
            - Self::replicate(1.6666654611e-1))
            * x2
            * x
            + x;

        // The lowest 2 bits of quadrant indicate the quadrant that we are in.
        // Let x be the original input value and x' our value that has been mapped to the range [-PI / 4, PI / 4].
        // since cos(x) = sin(x - PI / 2) and since we want to use the Taylor expansion as close as possible to 0,
        // we can alternate between using the Taylor expansion for sin and cos according to the following table:
        //
        // quadrant	 sin(x)		 cos(x)
        // XXX00b	 sin(x')	 cos(x')
        // XXX01b	 cos(x')	-sin(x')
        // XXX10b	-sin(x')	-cos(x')
        // XXX11b	-cos(x')	 sin(x')
        //
        // So: sin_sign = bit2, cos_sign = bit1 ^ bit2, bit1 determines if we use sin or cos Taylor expansion
        let bit1 = quadrant.logical_shift_left::<31>();
        let bit2 = UVec4::and(quadrant.logical_shift_left::<30>(), UVec4::replicate(0x80000000));

        // Select which one of the results is sin and which one is cos
        let s = Self::select(taylor_sin, taylor_cos, bit1);
        let c = Self::select(taylor_cos, taylor_sin, bit1);

        // Update the signs
        sin_sign = UVec4::xor(sin_sign, bit2);
        let cos_sign = UVec4::xor(bit1, bit2);

        // Correct the signs
        (Self::xor(s, sin_sign.reinterpret_as_float()), Self::xor(c, cos_sign.reinterpret_as_float()))
    }

    /// Calculate the tangent for each element of this vector (input in radians)
    #[inline]
    pub fn tan(self) -> Self {
        // Implementation based on tanf.c from the cephes library, see Vec4::SinCos for further details
        // Original implementation by Stephen L. Moshier (See: http://www.moshier.net/)

        // Make argument positive
        let tan_sign = UVec4::and(self.reinterpret_as_int(), UVec4::replicate(0x80000000));
        let mut x = Self::xor(self, tan_sign.reinterpret_as_float());

        // x / (PI / 2) rounded to nearest int gives us the quadrant closest to x
        let quadrant = (0.6366197723675814 * x + Self::replicate(0.5)).to_int();

        // Remap x to range [-PI / 4, PI / 4], see Vec4::SinCos
        let float_quadrant = quadrant.to_float();
        x = ((x - float_quadrant * 1.5703125) - float_quadrant * 0.0004837512969970703125)
            - float_quadrant * 7.549789948768648e-8;

        // Calculate x2 = x^2
        let x2 = x * x;

        // Roughly equivalent to the Taylor expansion:
        // Tan(x) = x + x^3/3 + 2*x^5/15 + 17*x^7/315 + 62*x^9/2835 + ...
        let mut tan = (((((9.38540185543e-3 * x2 + Self::replicate(3.11992232697e-3)) * x2
            + Self::replicate(2.44301354525e-2))
            * x2
            + Self::replicate(5.34112807005e-2))
            * x2
            + Self::replicate(1.33387994085e-1))
            * x2
            + Self::replicate(3.33331568548e-1))
            * x2
            * x
            + x;

        // For the 2nd and 4th quadrant we need to invert the value
        // (C++ adds FLT_MIN to tan when floating point exceptions are enabled to prevent a division by zero, this
        // port follows the build without floating point exceptions)
        let bit1 = quadrant.logical_shift_left::<31>();
        tan = Self::select(tan, Self::replicate(-1.0) / tan, bit1);

        // Put the sign back
        Self::xor(tan, tan_sign.reinterpret_as_float())
    }

    /// Calculate the arc sine for each element of this vector (returns value in the range [-PI / 2, PI / 2])
    /// Note that all input values will be clamped to the range [-1, 1] and this function will not return NaNs like std::asin
    #[inline]
    pub fn asin(self) -> Self {
        // Implementation based on asinf.c from the cephes library
        // Original implementation by Stephen L. Moshier (See: http://www.moshier.net/)

        // Make argument positive
        let asin_sign = UVec4::and(self.reinterpret_as_int(), UVec4::replicate(0x80000000));
        let mut a = Self::xor(self, asin_sign.reinterpret_as_float());

        // ASin is not defined outside the range [-1, 1] but it often happens that a value is slightly above 1 so we just clamp here
        a = Self::min(a, Self::one());

        // When |x| <= 0.5 we use the asin approximation as is
        let z1 = a * a;
        let x1 = a;

        // When |x| > 0.5 we use the identity asin(x) = PI / 2 - 2 * asin(sqrt((1 - x) / 2))
        let z2 = 0.5 * (Self::one() - a);
        let x2 = z2.sqrt();

        // Select which of the two situations we have
        let greater = Self::greater(a, Self::replicate(0.5));
        let mut z = Self::select(z1, z2, greater);
        let x = Self::select(x1, x2, greater);

        // Polynomial approximation of asin
        z = ((((4.2163199048e-2 * z + Self::replicate(2.4181311049e-2)) * z + Self::replicate(4.5470025998e-2)) * z
            + Self::replicate(7.4953002686e-2))
            * z
            + Self::replicate(1.6666752422e-1))
            * z
            * x
            + x;

        // If |x| > 0.5 we need to apply the remainder of the identity above
        z = Self::select(z, Self::replicate(0.5 * JPH_PI) - (z + z), greater);

        // Put the sign back
        Self::xor(z, asin_sign.reinterpret_as_float())
    }

    /// Calculate the arc cosine for each element of this vector (returns value in the range [0, PI])
    /// Note that all input values will be clamped to the range [-1, 1] and this function will not return NaNs like std::acos
    #[inline]
    pub fn acos(self) -> Self {
        // Not the most accurate, but simple
        Self::replicate(0.5 * JPH_PI) - self.asin()
    }

    /// Calculate the arc tangent for each element of this vector (returns value in the range [-PI / 2, PI / 2])
    #[inline]
    pub fn atan(self) -> Self {
        // Implementation based on atanf.c from the cephes library
        // Original implementation by Stephen L. Moshier (See: http://www.moshier.net/)

        // Make argument positive
        let atan_sign = UVec4::and(self.reinterpret_as_int(), UVec4::replicate(0x80000000));
        let mut x = Self::xor(self, atan_sign.reinterpret_as_float());
        let mut y = Self::zero();

        // If x > Tan(PI / 8)
        let greater1 = Self::greater(x, Self::replicate(0.4142135623730950));
        let x1 = (x - Self::one()) / (x + Self::one());

        // If x > Tan(3 * PI / 8)
        // (C++ adds FLT_MIN to x when floating point exceptions are enabled to prevent a division by zero, this port
        // follows the build without floating point exceptions)
        let greater2 = Self::greater(x, Self::replicate(2.414213562373095));
        let x2 = Self::replicate(-1.0) / x;

        // Apply first if
        x = Self::select(x, x1, greater1);
        y = Self::select(y, Self::replicate(0.25 * JPH_PI), greater1);

        // Apply second if
        x = Self::select(x, x2, greater2);
        y = Self::select(y, Self::replicate(0.5 * JPH_PI), greater2);

        // Polynomial approximation
        let z = x * x;
        y += (((8.05374449538e-2 * z - Self::replicate(1.38776856032e-1)) * z + Self::replicate(1.99777106478e-1)) * z
            - Self::replicate(3.33329491539e-1))
            * z
            * x
            + x;

        // Put the sign back
        Self::xor(y, atan_sign.reinterpret_as_float())
    }

    /// Calculate the arc tangent of y / x using the signs of the arguments to determine the correct quadrant (returns value in the range [-PI, PI])
    #[inline]
    pub fn atan2(y: Self, x: Self) -> Self {
        let sign_mask = UVec4::replicate(0x80000000);

        // Determine absolute value and sign of y
        let y_sign = UVec4::and(y.reinterpret_as_int(), sign_mask);
        let y_abs = Self::xor(y, y_sign.reinterpret_as_float());

        // Determine absolute value and sign of x
        let x_sign = UVec4::and(x.reinterpret_as_int(), sign_mask);
        let x_abs = Self::xor(x, x_sign.reinterpret_as_float());

        // Always divide smallest / largest to avoid dividing by zero
        let x_is_numerator = Self::less(x_abs, y_abs);
        let numerator = Self::select(y_abs, x_abs, x_is_numerator);
        let denominator = Self::select(x_abs, y_abs, x_is_numerator);
        let mut atan = (numerator / denominator).atan();

        // If we calculated x / y instead of y / x the result is PI / 2 - result (note that this is true because we know the result is positive because the input was positive)
        atan = Self::select(atan, Self::replicate(0.5 * JPH_PI) - atan, x_is_numerator);

        // Now we need to map to the correct quadrant
        // x_sign	y_sign	result
        // +1		+1		atan
        // -1		+1		-atan + PI
        // -1		-1		atan - PI
        // +1		-1		-atan
        // This can be written as: x_sign * y_sign * (atan - (x_sign < 0? PI : 0))
        atan -= Self::and(x_sign.arithmetic_shift_right::<31>().reinterpret_as_float(), Self::replicate(JPH_PI));
        atan = Self::xor(atan, UVec4::xor(x_sign, y_sign).reinterpret_as_float());
        atan
    }

    /// Compress a unit vector to a 32 bit value, precision is around 0.5 * 10^-3
    #[inline]
    pub fn compress_unit_vector(self) -> u32 {
        const ONE_OVER_SQRT2: f32 = 0.70710678;
        const NUM_BITS: u32 = 9;
        const MASK: u32 = (1 << NUM_BITS) - 1;
        const MAX_VALUE: u32 = MASK - 1; // Need odd number of buckets to quantize to or else we can't encode 0
        const SCALE: f32 = MAX_VALUE as f32 / (2.0 * ONE_OVER_SQRT2);

        // Store sign bit
        let mut v = self;
        let max_element = v.abs().highest_component_index() as u32;
        let mut value = 0u32;
        if v[max_element as usize] < 0.0 {
            value = 0x80000000;
            v = -v;
        }

        // Store highest component
        value |= max_element << 29;

        // Store the other three components in a compressed format
        let mut compressed = Self::clamp(
            (v + Self::replicate(ONE_OVER_SQRT2)) * SCALE + Self::replicate(0.5),
            Self::zero(),
            Self::replicate(MAX_VALUE as f32),
        )
        .to_int();
        match max_element {
            0 => compressed = compressed.swizzle::<SWIZZLE_Y, SWIZZLE_Z, SWIZZLE_W, SWIZZLE_UNUSED>(),
            1 => compressed = compressed.swizzle::<SWIZZLE_X, SWIZZLE_Z, SWIZZLE_W, SWIZZLE_UNUSED>(),
            2 => compressed = compressed.swizzle::<SWIZZLE_X, SWIZZLE_Y, SWIZZLE_W, SWIZZLE_UNUSED>(),
            _ => {}
        }

        value |= compressed.x();
        value |= compressed.y() << NUM_BITS;
        value |= compressed.z() << (2 * NUM_BITS);
        value
    }

    /// Decompress a unit vector from a 32 bit value
    #[inline]
    pub fn decompress_unit_vector(value: u32) -> Self {
        const ONE_OVER_SQRT2: f32 = 0.70710678;
        const NUM_BITS: u32 = 9;
        const MASK: u32 = (1 << NUM_BITS) - 1;
        const MAX_VALUE: u32 = MASK - 1; // Need odd number of buckets to quantize to or else we can't encode 0
        const HALF_MAX_VALUE: i32 = (MAX_VALUE >> 1) as i32;
        const SCALE: f32 = 2.0 * ONE_OVER_SQRT2 / MAX_VALUE as f32;

        // Restore three components
        let mut v = Self::new(
            ((value & MASK) as i32 - HALF_MAX_VALUE) as f32,
            (((value >> NUM_BITS) & MASK) as i32 - HALF_MAX_VALUE) as f32,
            (((value >> (2 * NUM_BITS)) as i32 & MASK as i32) as f32) - HALF_MAX_VALUE as f32,
            0.0,
        ) * SCALE;
        debug_assert!(v.w() == 0.0);

        // Restore the highest component
        v.set_w(super::sqrt(super::max(1.0 - v.length_sq(), 0.0)));

        // Extract sign
        if (value & 0x80000000) != 0 {
            v = -v;
        }

        // Swizzle the components in place
        match (value >> 29) & 3 {
            0 => v = v.swizzle::<SWIZZLE_W, SWIZZLE_X, SWIZZLE_Y, SWIZZLE_Z>(),
            1 => v = v.swizzle::<SWIZZLE_X, SWIZZLE_W, SWIZZLE_Y, SWIZZLE_Z>(),
            2 => v = v.swizzle::<SWIZZLE_X, SWIZZLE_Y, SWIZZLE_W, SWIZZLE_Z>(),
            _ => {}
        }

        v
    }
}

impl From<Vec3> for Vec4 {
    #[inline]
    fn from(v: Vec3) -> Self {
        Self::from_vec3(v)
    }
}

impl PartialEq for Vec4 {
    /// Comparison
    #[inline]
    fn eq(&self, v2: &Self) -> bool {
        Self::equals(*self, *v2).test_all_true()
    }
}

impl Index<usize> for Vec4 {
    type Output = f32;

    /// Get float component by index
    #[inline]
    fn index(&self, coordinate: usize) -> &f32 {
        debug_assert!(coordinate < 4);
        &self.f32s[coordinate]
    }
}

impl IndexMut<usize> for Vec4 {
    #[inline]
    fn index_mut(&mut self, coordinate: usize) -> &mut f32 {
        debug_assert!(coordinate < 4);
        &mut self.f32s[coordinate]
    }
}

impl Mul for Vec4 {
    type Output = Self;

    /// Multiply two float vectors (component wise)
    #[inline]
    fn mul(self, v2: Self) -> Self {
        Self::from_array(std::array::from_fn(|i| self.f32s[i] * v2.f32s[i]))
    }
}

impl Mul<f32> for Vec4 {
    type Output = Self;

    /// Multiply vector with float
    #[inline]
    fn mul(self, v2: f32) -> Self {
        Self::from_array(self.f32s.map(|v| v * v2))
    }
}

impl Mul<Vec4> for f32 {
    type Output = Vec4;

    /// Multiply vector with float
    #[inline]
    fn mul(self, v2: Vec4) -> Vec4 {
        Vec4::from_array(v2.f32s.map(|v| self * v))
    }
}

impl Div<f32> for Vec4 {
    type Output = Self;

    /// Divide vector by float
    #[inline]
    fn div(self, v2: f32) -> Self {
        Self::from_array(self.f32s.map(|v| v / v2))
    }
}

impl MulAssign<f32> for Vec4 {
    /// Multiply vector with float
    #[inline]
    fn mul_assign(&mut self, v2: f32) {
        *self = *self * v2;
    }
}

impl MulAssign for Vec4 {
    /// Multiply vector with vector
    #[inline]
    fn mul_assign(&mut self, v2: Self) {
        *self = *self * v2;
    }
}

impl DivAssign<f32> for Vec4 {
    /// Divide vector by float
    #[inline]
    fn div_assign(&mut self, v2: f32) {
        *self = *self / v2;
    }
}

impl Add for Vec4 {
    type Output = Self;

    /// Add two float vectors (component wise)
    #[inline]
    fn add(self, v2: Self) -> Self {
        Self::from_array(std::array::from_fn(|i| self.f32s[i] + v2.f32s[i]))
    }
}

impl AddAssign for Vec4 {
    /// Add two float vectors (component wise)
    #[inline]
    fn add_assign(&mut self, v2: Self) {
        *self = *self + v2;
    }
}

impl Neg for Vec4 {
    type Output = Self;

    /// Negate
    ///
    /// Computed as `0 - v` like the C++ cross platform deterministic build (so -0 is represented as 0).
    #[inline]
    fn neg(self) -> Self {
        Self::from_array(self.f32s.map(|v| 0.0 - v))
    }
}

impl Sub for Vec4 {
    type Output = Self;

    /// Subtract two float vectors (component wise)
    #[inline]
    fn sub(self, v2: Self) -> Self {
        Self::from_array(std::array::from_fn(|i| self.f32s[i] - v2.f32s[i]))
    }
}

impl SubAssign for Vec4 {
    /// Subtract two float vectors (component wise)
    #[inline]
    fn sub_assign(&mut self, v2: Self) {
        *self = *self - v2;
    }
}

impl Div for Vec4 {
    type Output = Self;

    /// Divide (component wise)
    #[inline]
    fn div(self, v2: Self) -> Self {
        Self::from_array(std::array::from_fn(|i| self.f32s[i] / v2.f32s[i]))
    }
}

impl fmt::Display for Vec4 {
    /// To String
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{}, {}, {}, {}", self.f32s[0], self.f32s[1], self.f32s[2], self.f32s[3])
    }
}
