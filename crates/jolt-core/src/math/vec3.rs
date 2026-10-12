//! Port of `Jolt/Math/Vec3.h` + `Vec3.inl` + `Vec3.cpp`.
//!
//! Scalar implementation. Where the C++ scalar fallback and the x86-64 SSE implementation (the platform of the
//! cross platform deterministic reference build) differ, this follows SSE; see `port/DEVIATIONS.md`. Negation is
//! `0 - v` and `ReduceSum` adds `+ 0.0` like in the C++ cross platform deterministic build.

// Keep the literals of the C++ source (e.g. 0.70710678f, 0.6366197723675814f) instead of the std constants
#![allow(clippy::approx_constant)]

use std::fmt;
use std::ops::{Add, AddAssign, Div, DivAssign, Index, Mul, MulAssign, Neg, Sub, SubAssign};
use std::sync::LazyLock;

use super::swizzle::{SWIZZLE_UNUSED, SWIZZLE_X, SWIZZLE_Y, SWIZZLE_Z};
use super::vec4::{simd_abs, simd_float_to_int, simd_max, simd_min};
use super::{Float3, JPH_PI, UVec4, Vec4, square};
use crate::hash_combine::JoltHash;
use crate::static_array::StaticArray;

/// A random number engine as used by [`Vec3::random`] (the C++ `Random` template parameter, a standard
/// UniformRandomBitGenerator such as `std::mt19937`)
pub trait RandomGenerator {
    /// Smallest value that [`generate`](Self::generate) returns (`Random::min()`)
    fn min_value(&self) -> u64;
    /// Largest value that [`generate`](Self::generate) returns (`Random::max()`)
    fn max_value(&self) -> u64;
    /// Generate the next random number (`Random::operator ()`)
    fn generate(&mut self) -> u64;
}

/// 3 component vector (stored as 4 vectors).
/// Note that we keep the 4th component the same as the 3rd component to avoid divisions by zero when JPH_FLOATING_POINT_EXCEPTIONS_ENABLED defined
///
/// The port always keeps W = Z (every constructor and setter replicates Z into W).
#[derive(Clone, Copy, Debug, Default)]
#[repr(C, align(16))]
pub struct Vec3 {
    pub(crate) f32s: [f32; 4],
}

impl Vec3 {
    /// Create a vector from 3 components
    #[inline]
    pub const fn new(x: f32, y: f32, z: f32) -> Self {
        Self { f32s: [x, y, z, z] }
    }

    /// Construct from a Vec4 (C++ `explicit Vec3(Vec4Arg)`), the W component is replaced by Z
    #[inline]
    pub const fn from_vec4(v: Vec4) -> Self {
        Self::new(v.f32s[0], v.f32s[1], v.f32s[2])
    }

    /// Load 3 floats from memory (C++ `explicit Vec3(const Float3 &)`)
    #[inline]
    pub const fn from_float3(v: &Float3) -> Self {
        Self::new(v.x, v.y, v.z)
    }

    /// Vector with all zeros
    #[inline]
    pub const fn zero() -> Self {
        Self::new(0.0, 0.0, 0.0)
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

    /// Vectors with the principal axis
    #[inline]
    pub const fn axis_x() -> Self {
        Self::new(1.0, 0.0, 0.0)
    }

    #[inline]
    pub const fn axis_y() -> Self {
        Self::new(0.0, 1.0, 0.0)
    }

    #[inline]
    pub const fn axis_z() -> Self {
        Self::new(0.0, 0.0, 1.0)
    }

    /// Replicate inV across all components
    #[inline]
    pub const fn replicate(v: f32) -> Self {
        Self::new(v, v, v)
    }

    /// Load 3 floats from memory (reads 32 bits extra which it doesn't use)
    ///
    /// The port only reads the 3 floats.
    #[inline]
    pub const fn load_float3_unsafe(v: &Float3) -> Self {
        Self::new(v.x, v.y, v.z)
    }

    /// Return the minimum value of each of the components
    #[inline]
    pub fn min(v1: Self, v2: Self) -> Self {
        Self::new(simd_min(v1.f32s[0], v2.f32s[0]), simd_min(v1.f32s[1], v2.f32s[1]), simd_min(v1.f32s[2], v2.f32s[2]))
    }

    /// Return the maximum of each of the components
    #[inline]
    pub fn max(v1: Self, v2: Self) -> Self {
        Self::new(simd_max(v1.f32s[0], v2.f32s[0]), simd_max(v1.f32s[1], v2.f32s[1]), simd_max(v1.f32s[2], v2.f32s[2]))
    }

    /// Clamp a vector between min and max (component wise)
    #[inline]
    pub fn clamp(v: Self, min: Self, max: Self) -> Self {
        Self::max(Self::min(v, max), min)
    }

    #[inline]
    fn compare(v1: Self, v2: Self, cmp: impl Fn(f32, f32) -> bool) -> UVec4 {
        let c = |i: usize| if cmp(v1.f32s[i], v2.f32s[i]) { 0xffffffff } else { 0 };
        let z = c(2);
        UVec4::new(c(0), c(1), z, z)
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
        let s = |i: usize| if (control.u32s[i] & 0x80000000) != 0 { set.f32s[i] } else { not_set.f32s[i] };
        Self::new(s(0), s(1), s(2))
    }

    /// Logical or (component wise)
    #[inline]
    pub fn or(v1: Self, v2: Self) -> Self {
        Self::from_vec4(UVec4::or(v1.reinterpret_as_int(), v2.reinterpret_as_int()).reinterpret_as_float())
    }

    /// Logical xor (component wise)
    #[inline]
    pub fn xor(v1: Self, v2: Self) -> Self {
        Self::from_vec4(UVec4::xor(v1.reinterpret_as_int(), v2.reinterpret_as_int()).reinterpret_as_float())
    }

    /// Logical and (component wise)
    #[inline]
    pub fn and(v1: Self, v2: Self) -> Self {
        Self::from_vec4(UVec4::and(v1.reinterpret_as_int(), v2.reinterpret_as_int()).reinterpret_as_float())
    }

    /// Get unit vector given spherical coordinates
    /// inTheta \f$\in [0, \pi]\f$ is angle between vector and z-axis
    /// inPhi \f$\in [0, 2 \pi]\f$ is the angle in the xy-plane starting from the x axis and rotating counter clockwise around the z-axis
    #[inline]
    pub fn unit_spherical(theta: f32, phi: f32) -> Self {
        let (s, c) = Vec4::new(theta, phi, 0.0, 0.0).sin_cos();
        Self::new(s.x() * c.y(), s.x() * s.y(), c.x())
    }

    /// A set of vectors uniformly spanning the surface of a unit sphere, usable for debug purposes
    pub fn unit_sphere() -> &'static StaticArray<Vec3, 1026> {
        static UNIT_SPHERE: LazyLock<StaticArray<Vec3, 1026>> = LazyLock::new(create_unit_sphere);
        &UNIT_SPHERE
    }

    /// Get random unit vector
    #[inline]
    pub fn random<R: RandomGenerator>(random: &mut R) -> Self {
        // Generating uniform unit random vectors in Rn - Andersen Ang
        // See: https://angms.science/doc/RM/randUnitVec.pdf
        let z = -1.0
            + 2.0 * (random.generate() - random.min_value()) as f32 / (random.max_value() - random.min_value()) as f32;
        let r = super::sqrt(1.0 - square(z));
        let theta = 2.0 * JPH_PI * (random.generate() - random.min_value()) as f32
            / (random.max_value() - random.min_value()) as f32;
        let (s, c) = Vec4::replicate(theta).sin_cos();
        Self::new(r * s.x(), r * c.x(), z)
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
        // Assure Z and W are the same
        self.f32s[2] = z;
        self.f32s[3] = z;
    }

    /// Set all components
    #[inline]
    pub fn set(&mut self, x: f32, y: f32, z: f32) {
        *self = Self::new(x, y, z);
    }

    /// Set float component by index
    #[inline]
    pub fn set_component(&mut self, coordinate: usize, value: f32) {
        assert!(coordinate < 3, "Vec3 coordinate {coordinate} out of range");
        self.f32s[coordinate] = value;
        self.f32s[3] = self.f32s[2]; // Assure Z and W are the same
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
        self.f32s[0].is_nan() || self.f32s[1].is_nan() || self.f32s[2].is_nan()
    }

    /// Swizzle the elements in inV
    #[inline]
    pub fn swizzle<const X: u32, const Y: u32, const Z: u32>(self) -> Self {
        const {
            assert!(X <= 3, "SwizzleX template parameter out of range");
            assert!(Y <= 3, "SwizzleY template parameter out of range");
            assert!(Z <= 3, "SwizzleZ template parameter out of range");
        }

        Self::new(self.f32s[X as usize], self.f32s[Y as usize], self.f32s[Z as usize])
    }

    /// Replicate the X component to all components
    #[inline]
    pub const fn splat_x(self) -> Vec4 {
        Vec4::replicate(self.f32s[0])
    }

    /// Replicate the Y component to all components
    #[inline]
    pub const fn splat_y(self) -> Vec4 {
        Vec4::replicate(self.f32s[1])
    }

    /// Replicate the Z component to all components
    #[inline]
    pub const fn splat_z(self) -> Vec4 {
        Vec4::replicate(self.f32s[2])
    }

    /// Get index of component with lowest value
    #[inline]
    pub fn lowest_component_index(self) -> i32 {
        if self.x() < self.y() {
            if self.z() < self.x() { 2 } else { 0 }
        } else if self.z() < self.y() {
            2
        } else {
            1
        }
    }

    /// Get index of component with highest value
    #[inline]
    pub fn highest_component_index(self) -> i32 {
        if self.x() > self.y() {
            if self.z() > self.x() { 2 } else { 0 }
        } else if self.z() > self.y() {
            2
        } else {
            1
        }
    }

    /// Return the absolute value of each of the components
    #[inline]
    pub fn abs(self) -> Self {
        Self::new(simd_abs(self.f32s[0]), simd_abs(self.f32s[1]), simd_abs(self.f32s[2]))
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

    /// Cross product
    #[inline]
    pub fn cross(self, v2: Self) -> Self {
        Self::new(
            self.f32s[1] * v2.f32s[2] - self.f32s[2] * v2.f32s[1],
            self.f32s[2] * v2.f32s[0] - self.f32s[0] * v2.f32s[2],
            self.f32s[0] * v2.f32s[1] - self.f32s[1] * v2.f32s[0],
        )
    }

    /// Cross product (more precise version when FMA is available)
    #[inline]
    pub fn cross_precise(self, v2: Self) -> Self {
        Self::difference_of_products(
            self,
            v2.swizzle::<SWIZZLE_Y, SWIZZLE_Z, SWIZZLE_X>(),
            self.swizzle::<SWIZZLE_Y, SWIZZLE_Z, SWIZZLE_X>(),
            v2,
        )
        .swizzle::<SWIZZLE_Y, SWIZZLE_Z, SWIZZLE_X>()
    }

    /// Dot product, returns the dot product in X, Y and Z components
    #[inline]
    pub fn dot_v(self, v2: Self) -> Self {
        Self::replicate(self.dot(v2))
    }

    /// Dot product, returns the dot product in X, Y, Z and W components
    #[inline]
    pub fn dot_v4(self, v2: Self) -> Vec4 {
        Vec4::replicate(self.dot(v2))
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

    /// Normalize vector or return inZeroValue if the length of the vector is zero
    #[inline]
    pub fn normalized_or(self, zero_value: Self) -> Self {
        let len_sq = self.length_sq();
        if len_sq <= f32::MIN_POSITIVE { zero_value } else { self / super::sqrt(len_sq) }
    }

    /// Store 3 floats to memory
    #[inline]
    pub const fn store_float3(self) -> Float3 {
        Float3::new(self.f32s[0], self.f32s[1], self.f32s[2])
    }

    /// Convert each component from a float to an int
    ///
    /// Like `_mm_cvttps_epi32` (the SSE implementation) values are truncated to signed integers, values that are out
    /// of range become 0x80000000.
    #[inline]
    pub fn to_int(self) -> UVec4 {
        UVec4::from_array(self.f32s.map(simd_float_to_int))
    }

    /// Reinterpret Vec3 as a UVec4 (doesn't change the bits)
    #[inline]
    pub fn reinterpret_as_int(self) -> UVec4 {
        UVec4::from_array(self.f32s.map(f32::to_bits))
    }

    /// Get the minimum of X, Y and Z
    #[inline]
    pub fn reduce_min(self) -> f32 {
        let mut v = Self::min(self, self.swizzle::<SWIZZLE_Y, SWIZZLE_UNUSED, SWIZZLE_Z>());
        v = Self::min(v, v.swizzle::<SWIZZLE_Z, SWIZZLE_UNUSED, SWIZZLE_UNUSED>());
        v.x()
    }

    /// Get the maximum of X, Y and Z
    #[inline]
    pub fn reduce_max(self) -> f32 {
        let mut v = Self::max(self, self.swizzle::<SWIZZLE_Y, SWIZZLE_UNUSED, SWIZZLE_Z>());
        v = Self::max(v, v.swizzle::<SWIZZLE_Z, SWIZZLE_UNUSED, SWIZZLE_UNUSED>());
        v.x()
    }

    /// Get the sum of X, Y and Z
    #[inline]
    pub fn reduce_sum(self) -> f32 {
        // Ensure that we handle -0.0f correctly when cross platform deterministic behavior is required.
        (self.f32s[0] + self.f32s[1]) + (self.f32s[2] + 0.0)
    }

    /// Component wise square root
    #[inline]
    pub fn sqrt(self) -> Self {
        Self::new(super::sqrt(self.f32s[0]), super::sqrt(self.f32s[1]), super::sqrt(self.f32s[2]))
    }

    /// Get normalized vector that is perpendicular to this vector
    #[inline]
    pub fn normalized_perpendicular(self) -> Self {
        let (x, y, z) = (self.f32s[0], self.f32s[1], self.f32s[2]);
        let (xx, yy, zz) = (x * x, y * y, z * z);
        let perp_x = Self::new(z, 0.0, 0.0 - x);
        let perp_y = Self::new(0.0, z, 0.0 - y);
        (if xx > yy { perp_x } else { perp_y }) / super::sqrt(simd_max(xx, yy) + zz)
    }

    /// Get vector that contains the sign of each element (returns 1.0f if positive, -1.0f if negative)
    #[inline]
    pub fn sign(self) -> Self {
        let s = |v: f32| if v.is_sign_negative() { -1.0 } else { 1.0 };
        Self::new(s(self.f32s[0]), s(self.f32s[1]), s(self.f32s[2]))
    }

    /// Flips the signs of the components, e.g. FlipSign<-1, 1, -1>() will flip the signs of the X and Z components
    #[inline]
    pub fn flip_sign<const X: i32, const Y: i32, const Z: i32>(self) -> Self {
        const {
            assert!(X == 1 || X == -1, "X must be 1 or -1");
            assert!(Y == 1 || Y == -1, "Y must be 1 or -1");
            assert!(Z == 1 || Z == -1, "Z must be 1 or -1");
        }
        Self::xor(
            self,
            Self::new(if X > 0 { 0.0 } else { -0.0 }, if Y > 0 { 0.0 } else { -0.0 }, if Z > 0 { 0.0 } else { -0.0 }),
        )
    }

    /// Compress a unit vector to a 32 bit value, precision is around 10^-4
    #[inline]
    pub fn compress_unit_vector(self) -> u32 {
        const ONE_OVER_SQRT2: f32 = 0.70710678;
        const NUM_BITS: u32 = 14;
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

        // Store the other two components in a compressed format
        let mut compressed = Self::clamp(
            (v + Self::replicate(ONE_OVER_SQRT2)) * SCALE + Self::replicate(0.5),
            Self::zero(),
            Self::replicate(MAX_VALUE as f32),
        )
        .to_int();
        match max_element {
            0 => compressed = compressed.swizzle::<SWIZZLE_Y, SWIZZLE_Z, SWIZZLE_UNUSED, SWIZZLE_UNUSED>(),
            1 => compressed = compressed.swizzle::<SWIZZLE_X, SWIZZLE_Z, SWIZZLE_UNUSED, SWIZZLE_UNUSED>(),
            _ => {}
        }

        value |= compressed.x();
        value |= compressed.y() << NUM_BITS;
        value
    }

    /// Decompress a unit vector from a 32 bit value
    #[inline]
    pub fn decompress_unit_vector(value: u32) -> Self {
        const ONE_OVER_SQRT2: f32 = 0.70710678;
        const NUM_BITS: u32 = 14;
        const MASK: u32 = (1 << NUM_BITS) - 1;
        const MAX_VALUE: u32 = MASK - 1; // Need odd number of buckets to quantize to or else we can't encode 0
        const HALF_MAX_VALUE: i32 = (MAX_VALUE >> 1) as i32;
        const SCALE: f32 = 2.0 * ONE_OVER_SQRT2 / MAX_VALUE as f32;

        // Restore two components
        let mut v = Self::new(
            ((value & MASK) as i32 - HALF_MAX_VALUE) as f32,
            (((value >> NUM_BITS) & MASK) as i32 - HALF_MAX_VALUE) as f32,
            0.0,
        ) * SCALE;
        debug_assert!(v.z() == 0.0);

        // Restore the highest component
        v.set_z(super::sqrt(super::max(1.0 - v.length_sq(), 0.0)));

        // Extract sign
        if (value & 0x80000000) != 0 {
            v = -v;
        }

        // Swizzle the components in place
        match (value >> 29) & 3 {
            0 => v = v.swizzle::<SWIZZLE_Z, SWIZZLE_X, SWIZZLE_Y>(),
            1 => v = v.swizzle::<SWIZZLE_X, SWIZZLE_Z, SWIZZLE_Y>(),
            _ => {}
        }

        v
    }
}

fn add_vertex(vertices: &mut StaticArray<Vec3, 1026>, vertex: Vec3) {
    let found = vertices.iter().any(|v| v.is_close(vertex, 1.0e-6));
    if !found {
        vertices.push(vertex);
    }
}

fn create_vertices(vertices: &mut StaticArray<Vec3, 1026>, dir1: Vec3, dir2: Vec3, dir3: Vec3, level: i32) {
    let center1 = (dir1 + dir2).normalized();
    let center2 = (dir2 + dir3).normalized();
    let center3 = (dir3 + dir1).normalized();

    add_vertex(vertices, center1);
    add_vertex(vertices, center2);
    add_vertex(vertices, center3);

    if level > 0 {
        let new_level = level - 1;
        create_vertices(vertices, dir1, center1, center3, new_level);
        create_vertices(vertices, center1, center2, center3, new_level);
        create_vertices(vertices, center1, dir2, center2, new_level);
        create_vertices(vertices, center3, center2, dir3, new_level);
    }
}

fn create_unit_sphere() -> StaticArray<Vec3, 1026> {
    const LEVEL: i32 = 3;

    let mut verts = StaticArray::default();

    // Add unit axis
    verts.push(Vec3::axis_x());
    verts.push(-Vec3::axis_x());
    verts.push(Vec3::axis_y());
    verts.push(-Vec3::axis_y());
    verts.push(Vec3::axis_z());
    verts.push(-Vec3::axis_z());

    // Subdivide
    create_vertices(&mut verts, Vec3::axis_x(), Vec3::axis_y(), Vec3::axis_z(), LEVEL);
    create_vertices(&mut verts, -Vec3::axis_x(), Vec3::axis_y(), Vec3::axis_z(), LEVEL);
    create_vertices(&mut verts, Vec3::axis_x(), -Vec3::axis_y(), Vec3::axis_z(), LEVEL);
    create_vertices(&mut verts, -Vec3::axis_x(), -Vec3::axis_y(), Vec3::axis_z(), LEVEL);
    create_vertices(&mut verts, Vec3::axis_x(), Vec3::axis_y(), -Vec3::axis_z(), LEVEL);
    create_vertices(&mut verts, -Vec3::axis_x(), Vec3::axis_y(), -Vec3::axis_z(), LEVEL);
    create_vertices(&mut verts, Vec3::axis_x(), -Vec3::axis_y(), -Vec3::axis_z(), LEVEL);
    create_vertices(&mut verts, -Vec3::axis_x(), -Vec3::axis_y(), -Vec3::axis_z(), LEVEL);

    verts
}

impl From<Vec4> for Vec3 {
    #[inline]
    fn from(v: Vec4) -> Self {
        Self::from_vec4(v)
    }
}

impl From<Float3> for Vec3 {
    #[inline]
    fn from(v: Float3) -> Self {
        Self::from_float3(&v)
    }
}

impl PartialEq for Vec3 {
    /// Comparison
    #[inline]
    fn eq(&self, v2: &Self) -> bool {
        Self::equals(*self, *v2).test_all_xyz_true()
    }
}

impl Index<usize> for Vec3 {
    type Output = f32;

    /// Get float component by index
    #[inline]
    fn index(&self, coordinate: usize) -> &f32 {
        assert!(coordinate < 3, "Vec3 coordinate {coordinate} out of range");
        &self.f32s[coordinate]
    }
}

impl Mul for Vec3 {
    type Output = Self;

    /// Multiply two float vectors (component wise)
    #[inline]
    fn mul(self, v2: Self) -> Self {
        Self::new(self.f32s[0] * v2.f32s[0], self.f32s[1] * v2.f32s[1], self.f32s[2] * v2.f32s[2])
    }
}

impl Mul<f32> for Vec3 {
    type Output = Self;

    /// Multiply vector with float
    #[inline]
    fn mul(self, v2: f32) -> Self {
        Self::new(self.f32s[0] * v2, self.f32s[1] * v2, self.f32s[2] * v2)
    }
}

impl Mul<Vec3> for f32 {
    type Output = Vec3;

    /// Multiply vector with float
    #[inline]
    fn mul(self, v2: Vec3) -> Vec3 {
        Vec3::new(self * v2.f32s[0], self * v2.f32s[1], self * v2.f32s[2])
    }
}

impl Div<f32> for Vec3 {
    type Output = Self;

    /// Divide vector by float
    #[inline]
    fn div(self, v2: f32) -> Self {
        Self::new(self.f32s[0] / v2, self.f32s[1] / v2, self.f32s[2] / v2)
    }
}

impl MulAssign<f32> for Vec3 {
    /// Multiply vector with float
    #[inline]
    fn mul_assign(&mut self, v2: f32) {
        *self = *self * v2;
    }
}

impl MulAssign for Vec3 {
    /// Multiply vector with vector
    #[inline]
    fn mul_assign(&mut self, v2: Self) {
        *self = *self * v2;
    }
}

impl DivAssign<f32> for Vec3 {
    /// Divide vector by float
    #[inline]
    fn div_assign(&mut self, v2: f32) {
        *self = *self / v2;
    }
}

impl Add for Vec3 {
    type Output = Self;

    /// Add two float vectors (component wise)
    #[inline]
    fn add(self, v2: Self) -> Self {
        Self::new(self.f32s[0] + v2.f32s[0], self.f32s[1] + v2.f32s[1], self.f32s[2] + v2.f32s[2])
    }
}

impl AddAssign for Vec3 {
    /// Add two float vectors (component wise)
    #[inline]
    fn add_assign(&mut self, v2: Self) {
        *self = *self + v2;
    }
}

impl Neg for Vec3 {
    type Output = Self;

    /// Negate
    ///
    /// Computed as `0 - v` like the C++ cross platform deterministic build (so -0 is represented as 0).
    #[inline]
    fn neg(self) -> Self {
        Self::new(0.0 - self.f32s[0], 0.0 - self.f32s[1], 0.0 - self.f32s[2])
    }
}

impl Sub for Vec3 {
    type Output = Self;

    /// Subtract two float vectors (component wise)
    #[inline]
    fn sub(self, v2: Self) -> Self {
        Self::new(self.f32s[0] - v2.f32s[0], self.f32s[1] - v2.f32s[1], self.f32s[2] - v2.f32s[2])
    }
}

impl SubAssign for Vec3 {
    /// Subtract two float vectors (component wise)
    #[inline]
    fn sub_assign(&mut self, v2: Self) {
        *self = *self - v2;
    }
}

impl Div for Vec3 {
    type Output = Self;

    /// Divide (component wise)
    #[inline]
    fn div(self, v2: Self) -> Self {
        Self::new(self.f32s[0] / v2.f32s[0], self.f32s[1] / v2.f32s[1], self.f32s[2] / v2.f32s[2])
    }
}

impl fmt::Display for Vec3 {
    /// To String
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{}, {}, {}", self.f32s[0], self.f32s[1], self.f32s[2])
    }
}

// Create a JoltHash for Vec3
impl JoltHash for Vec3 {
    fn jolt_hash(&self) -> u64 {
        crate::hash_combine_args!(self.x(), self.y(), self.z())
    }
}
