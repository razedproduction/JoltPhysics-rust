//! Port of `Jolt/Math/DVec3.h` + `DVec3.inl`.
//!
//! Scalar implementation. Where the C++ scalar fallback and the x86-64 SSE / AVX implementation (the platform of the
//! cross platform deterministic reference build) differ, this follows SSE / AVX; see `port/DEVIATIONS.md`. Negation
//! is `0 - v` like in the C++ cross platform deterministic build.

use std::fmt;
use std::ops::{Add, AddAssign, Div, DivAssign, Index, Mul, MulAssign, Neg, Sub, SubAssign};

use super::vec4::{simd_max, simd_min};
use super::{Double3, Vec3, Vec4};
use crate::hash_combine::JoltHash;

/// 3 component vector of doubles (stored as 4 vectors).
/// Note that we keep the 4th component the same as the 3rd component to avoid divisions by zero when JPH_FLOATING_POINT_EXCEPTIONS_ENABLED defined
///
/// The port always keeps W = Z (every constructor and setter replicates Z into W).
#[derive(Clone, Copy, Debug, Default)]
#[repr(C, align(32))]
pub struct DVec3 {
    pub(crate) f64s: [f64; 4],
}

impl DVec3 {
    /// Representations of true and false for boolean operations
    pub const TRUE: f64 = f64::from_bits(!0u64);
    pub const FALSE: f64 = 0.0;

    /// Create a vector from 3 components
    #[inline]
    pub const fn new(x: f64, y: f64, z: f64) -> Self {
        Self { f64s: [x, y, z, z] }
    }

    /// Convert a Vec3 to a DVec3 (C++ `explicit DVec3(Vec3Arg)`)
    #[inline]
    pub fn from_vec3(v: Vec3) -> Self {
        Self::new(v.x() as f64, v.y() as f64, v.z() as f64)
    }

    /// Convert the X, Y and Z components of a Vec4 to a DVec3 (C++ `explicit DVec3(Vec4Arg)`)
    #[inline]
    pub fn from_vec4(v: Vec4) -> Self {
        Self::from_vec3(Vec3::from_vec4(v))
    }

    /// Load 3 doubles from memory (C++ `explicit DVec3(const Double3 &)`)
    #[inline]
    pub const fn from_double3(v: &Double3) -> Self {
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
    pub const fn replicate(v: f64) -> Self {
        Self::new(v, v, v)
    }

    /// Vector with all NaN's
    #[inline]
    pub const fn nan() -> Self {
        Self::replicate(f64::NAN)
    }

    /// Load 3 doubles from memory (reads 64 bits extra which it doesn't use)
    ///
    /// The port only reads the 3 doubles.
    #[inline]
    pub const fn load_double3_unsafe(v: &Double3) -> Self {
        Self::new(v.x, v.y, v.z)
    }

    /// Store 3 doubles to memory
    #[inline]
    pub const fn store_double3(self) -> Double3 {
        Double3::new(self.f64s[0], self.f64s[1], self.f64s[2])
    }

    /// Convert to float vector 3 rounding to nearest (C++ `explicit operator Vec3()`)
    #[inline]
    pub fn to_vec3(self) -> Vec3 {
        Vec3::new(self.x() as f32, self.y() as f32, self.z() as f32)
    }

    /// Prepare to convert to float vector 3 rounding towards zero (returns DVec3 that can be converted to a Vec3 to get the rounding)
    #[inline]
    pub fn prepare_round_to_zero(self) -> Self {
        // Float has 23 bit mantissa, double 52 bit mantissa => we lose 29 bits when converting from double to float
        const DOUBLE_TO_FLOAT_MANTISSA_LOSS: u64 = (1u64 << 29) - 1;
        let f = |v: f64| f64::from_bits(v.to_bits() & !DOUBLE_TO_FLOAT_MANTISSA_LOSS);
        Self::new(f(self.f64s[0]), f(self.f64s[1]), f(self.f64s[2]))
    }

    /// Prepare to convert to float vector 3 rounding towards infinity (returns DVec3 that can be converted to a Vec3 to get the rounding)
    #[inline]
    pub fn prepare_round_to_inf(self) -> Self {
        // Float has 23 bit mantissa, double 52 bit mantissa => we lose 29 bits when converting from double to float
        const DOUBLE_TO_FLOAT_MANTISSA_LOSS: u64 = (1u64 << 29) - 1;
        let f = |v: f64| {
            let u = v.to_bits();
            f64::from_bits(if (u & DOUBLE_TO_FLOAT_MANTISSA_LOSS) == 0 { u } else { u | DOUBLE_TO_FLOAT_MANTISSA_LOSS })
        };
        Self::new(f(self.f64s[0]), f(self.f64s[1]), f(self.f64s[2]))
    }

    /// Convert to float vector 3 rounding down
    #[inline]
    pub fn to_vec3_round_down(self) -> Vec3 {
        let to_zero = self.prepare_round_to_zero();
        let to_inf = self.prepare_round_to_inf();
        Self::select(to_zero, to_inf, Self::less(self, Self::zero())).to_vec3()
    }

    /// Convert to float vector 3 rounding up
    #[inline]
    pub fn to_vec3_round_up(self) -> Vec3 {
        let to_zero = self.prepare_round_to_zero();
        let to_inf = self.prepare_round_to_inf();
        Self::select(to_inf, to_zero, Self::less(self, Self::zero())).to_vec3()
    }

    /// Return the minimum value of each of the components
    #[inline]
    pub fn min(v1: Self, v2: Self) -> Self {
        Self::new(simd_min(v1.f64s[0], v2.f64s[0]), simd_min(v1.f64s[1], v2.f64s[1]), simd_min(v1.f64s[2], v2.f64s[2]))
    }

    /// Return the maximum of each of the components
    #[inline]
    pub fn max(v1: Self, v2: Self) -> Self {
        Self::new(simd_max(v1.f64s[0], v2.f64s[0]), simd_max(v1.f64s[1], v2.f64s[1]), simd_max(v1.f64s[2], v2.f64s[2]))
    }

    /// Clamp a vector between min and max (component wise)
    #[inline]
    pub fn clamp(v: Self, min: Self, max: Self) -> Self {
        Self::max(Self::min(v, max), min)
    }

    #[inline]
    fn compare(v1: Self, v2: Self, cmp: impl Fn(f64, f64) -> bool) -> Self {
        let c = |i: usize| if cmp(v1.f64s[i], v2.f64s[i]) { Self::TRUE } else { Self::FALSE };
        Self::new(c(0), c(1), c(2))
    }

    /// Equals (component wise)
    #[inline]
    pub fn equals(v1: Self, v2: Self) -> Self {
        Self::compare(v1, v2, |a, b| a == b)
    }

    /// Less than (component wise)
    #[inline]
    pub fn less(v1: Self, v2: Self) -> Self {
        Self::compare(v1, v2, |a, b| a < b)
    }

    /// Less than or equal (component wise)
    #[inline]
    pub fn less_or_equal(v1: Self, v2: Self) -> Self {
        Self::compare(v1, v2, |a, b| a <= b)
    }

    /// Greater than (component wise)
    #[inline]
    pub fn greater(v1: Self, v2: Self) -> Self {
        Self::compare(v1, v2, |a, b| a > b)
    }

    /// Greater than or equal (component wise)
    #[inline]
    pub fn greater_or_equal(v1: Self, v2: Self) -> Self {
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
    pub fn select(not_set: Self, set: Self, control: Self) -> Self {
        let s = |i: usize| if (control.f64s[i].to_bits() & (1u64 << 63)) != 0 { set.f64s[i] } else { not_set.f64s[i] };
        Self::new(s(0), s(1), s(2))
    }

    #[inline]
    fn bit_op(v1: Self, v2: Self, op: impl Fn(u64, u64) -> u64) -> Self {
        let b = |i: usize| f64::from_bits(op(v1.f64s[i].to_bits(), v2.f64s[i].to_bits()));
        Self::new(b(0), b(1), b(2))
    }

    /// Logical or (component wise)
    #[inline]
    pub fn or(v1: Self, v2: Self) -> Self {
        Self::bit_op(v1, v2, |a, b| a | b)
    }

    /// Logical xor (component wise)
    #[inline]
    pub fn xor(v1: Self, v2: Self) -> Self {
        Self::bit_op(v1, v2, |a, b| a ^ b)
    }

    /// Logical and (component wise)
    #[inline]
    pub fn and(v1: Self, v2: Self) -> Self {
        Self::bit_op(v1, v2, |a, b| a & b)
    }

    /// Store if X is true in bit 0, Y in bit 1, Z in bit 2 and W in bit 3 (true is when highest bit of component is set)
    #[inline]
    pub fn trues(self) -> i32 {
        ((self.f64s[0].to_bits() >> 63) | ((self.f64s[1].to_bits() >> 63) << 1) | ((self.f64s[2].to_bits() >> 63) << 2))
            as i32
    }

    /// Test if any of the components are true (true is when highest bit of component is set)
    #[inline]
    pub fn test_any_true(self) -> bool {
        self.trues() != 0
    }

    /// Test if all components are true (true is when highest bit of component is set)
    #[inline]
    pub fn test_all_true(self) -> bool {
        self.trues() == 0x7
    }

    /// Get individual components
    #[inline]
    pub const fn x(self) -> f64 {
        self.f64s[0]
    }

    #[inline]
    pub const fn y(self) -> f64 {
        self.f64s[1]
    }

    #[inline]
    pub const fn z(self) -> f64 {
        self.f64s[2]
    }

    /// Set individual components
    #[inline]
    pub fn set_x(&mut self, x: f64) {
        self.f64s[0] = x;
    }

    #[inline]
    pub fn set_y(&mut self, y: f64) {
        self.f64s[1] = y;
    }

    #[inline]
    pub fn set_z(&mut self, z: f64) {
        // Assure Z and W are the same
        self.f64s[2] = z;
        self.f64s[3] = z;
    }

    /// Set all components
    #[inline]
    pub fn set(&mut self, x: f64, y: f64, z: f64) {
        *self = Self::new(x, y, z);
    }

    /// Set double component by index
    #[inline]
    pub fn set_component(&mut self, coordinate: usize, value: f64) {
        assert!(coordinate < 3, "DVec3 coordinate {coordinate} out of range");
        self.f64s[coordinate] = value;
        self.f64s[3] = self.f64s[2]; // Assure Z and W are the same
    }

    /// Test if two vectors are close
    #[inline]
    pub fn is_close(self, v2: Self, max_dist_sq: f64) -> bool {
        (v2 - self).length_sq() <= max_dist_sq
    }

    /// Test if two vectors are close (with the default inMaxDistSq = 1.0e-24)
    #[inline]
    pub fn is_close_default(self, v2: Self) -> bool {
        self.is_close(v2, 1.0e-24)
    }

    /// Test if vector is near zero
    #[inline]
    pub fn is_near_zero(self, max_dist_sq: f64) -> bool {
        self.length_sq() <= max_dist_sq
    }

    /// Test if vector is near zero (with the default inMaxDistSq = 1.0e-24)
    #[inline]
    pub fn is_near_zero_default(self) -> bool {
        self.is_near_zero(1.0e-24)
    }

    /// Test if vector is normalized
    #[inline]
    pub fn is_normalized(self, tolerance: f64) -> bool {
        (self.length_sq() - 1.0).abs() <= tolerance
    }

    /// Test if vector is normalized (with the default inTolerance = 1.0e-12)
    #[inline]
    pub fn is_normalized_default(self) -> bool {
        self.is_normalized(1.0e-12)
    }

    /// Test if vector contains NaN elements
    #[inline]
    pub fn is_nan(self) -> bool {
        self.f64s[0].is_nan() || self.f64s[1].is_nan() || self.f64s[2].is_nan()
    }

    /// Return the absolute value of each of the components
    #[inline]
    pub fn abs(self) -> Self {
        Self::new(simd_abs_f64(self.f64s[0]), simd_abs_f64(self.f64s[1]), simd_abs_f64(self.f64s[2]))
    }

    /// Reciprocal vector (1 / value) for each of the components
    #[inline]
    pub fn reciprocal(self) -> Self {
        Self::one() / self
    }

    /// Cross product
    #[inline]
    pub fn cross(self, v2: Self) -> Self {
        Self::new(
            self.f64s[1] * v2.f64s[2] - self.f64s[2] * v2.f64s[1],
            self.f64s[2] * v2.f64s[0] - self.f64s[0] * v2.f64s[2],
            self.f64s[0] * v2.f64s[1] - self.f64s[1] * v2.f64s[0],
        )
    }

    /// Dot product
    ///
    /// Summed as `(x + y) + z` like all SIMD implementations (the C++ scalar fallback starts from `0.0`).
    #[inline]
    pub fn dot(self, v2: Self) -> f64 {
        (self.f64s[0] * v2.f64s[0] + self.f64s[1] * v2.f64s[1]) + self.f64s[2] * v2.f64s[2]
    }

    /// Squared length of vector
    #[inline]
    pub fn length_sq(self) -> f64 {
        self.dot(self)
    }

    /// Length of vector
    #[inline]
    pub fn length(self) -> f64 {
        super::sqrt(self.dot(self))
    }

    /// Normalize vector
    #[inline]
    pub fn normalized(self) -> Self {
        self / self.length()
    }

    /// Component wise square root
    #[inline]
    pub fn sqrt(self) -> Self {
        Self::new(super::sqrt(self.f64s[0]), super::sqrt(self.f64s[1]), super::sqrt(self.f64s[2]))
    }

    /// Get vector that contains the sign of each element (returns 1.0 if positive, -1.0 if negative)
    #[inline]
    pub fn sign(self) -> Self {
        let s = |v: f64| if v.is_sign_negative() { -1.0 } else { 1.0 };
        Self::new(s(self.f64s[0]), s(self.f64s[1]), s(self.f64s[2]))
    }
}

/// `_mm256_max_pd(_mm256_sub_pd(_mm256_setzero_pd(), v), v)`: absolute value as the AVX implementation computes it
#[inline]
fn simd_abs_f64(v: f64) -> f64 {
    simd_max(0.0 - v, v)
}

impl From<Vec3> for DVec3 {
    #[inline]
    fn from(v: Vec3) -> Self {
        Self::from_vec3(v)
    }
}

impl From<Vec4> for DVec3 {
    #[inline]
    fn from(v: Vec4) -> Self {
        Self::from_vec4(v)
    }
}

impl From<Double3> for DVec3 {
    #[inline]
    fn from(v: Double3) -> Self {
        Self::from_double3(&v)
    }
}

impl From<DVec3> for Vec3 {
    #[inline]
    fn from(v: DVec3) -> Self {
        v.to_vec3()
    }
}

impl PartialEq for DVec3 {
    /// Comparison
    #[inline]
    fn eq(&self, v2: &Self) -> bool {
        Self::equals(*self, *v2).test_all_true()
    }
}

impl Index<usize> for DVec3 {
    type Output = f64;

    /// Get double component by index
    #[inline]
    fn index(&self, coordinate: usize) -> &f64 {
        assert!(coordinate < 3, "DVec3 coordinate {coordinate} out of range");
        &self.f64s[coordinate]
    }
}

impl Mul for DVec3 {
    type Output = Self;

    /// Multiply two double vectors (component wise)
    #[inline]
    fn mul(self, v2: Self) -> Self {
        Self::new(self.f64s[0] * v2.f64s[0], self.f64s[1] * v2.f64s[1], self.f64s[2] * v2.f64s[2])
    }
}

impl Mul<f64> for DVec3 {
    type Output = Self;

    /// Multiply vector with double
    #[inline]
    fn mul(self, v2: f64) -> Self {
        Self::new(self.f64s[0] * v2, self.f64s[1] * v2, self.f64s[2] * v2)
    }
}

impl Mul<DVec3> for f64 {
    type Output = DVec3;

    /// Multiply vector with double
    #[inline]
    fn mul(self, v2: DVec3) -> DVec3 {
        DVec3::new(self * v2.f64s[0], self * v2.f64s[1], self * v2.f64s[2])
    }
}

impl Div<f64> for DVec3 {
    type Output = Self;

    /// Divide vector by double
    #[inline]
    fn div(self, v2: f64) -> Self {
        Self::new(self.f64s[0] / v2, self.f64s[1] / v2, self.f64s[2] / v2)
    }
}

impl MulAssign<f64> for DVec3 {
    /// Multiply vector with double
    #[inline]
    fn mul_assign(&mut self, v2: f64) {
        *self = *self * v2;
    }
}

impl MulAssign for DVec3 {
    /// Multiply vector with vector
    #[inline]
    fn mul_assign(&mut self, v2: Self) {
        *self = *self * v2;
    }
}

impl DivAssign<f64> for DVec3 {
    /// Divide vector by double
    #[inline]
    fn div_assign(&mut self, v2: f64) {
        *self = *self / v2;
    }
}

impl Add<Vec3> for DVec3 {
    type Output = Self;

    /// Add two vectors (component wise)
    #[inline]
    fn add(self, v2: Vec3) -> Self {
        Self::new(self.f64s[0] + v2.x() as f64, self.f64s[1] + v2.y() as f64, self.f64s[2] + v2.z() as f64)
    }
}

impl Add for DVec3 {
    type Output = Self;

    /// Add two double vectors (component wise)
    #[inline]
    fn add(self, v2: Self) -> Self {
        Self::new(self.f64s[0] + v2.f64s[0], self.f64s[1] + v2.f64s[1], self.f64s[2] + v2.f64s[2])
    }
}

impl AddAssign<Vec3> for DVec3 {
    /// Add two vectors (component wise)
    #[inline]
    fn add_assign(&mut self, v2: Vec3) {
        *self = *self + v2;
    }
}

impl AddAssign for DVec3 {
    /// Add two double vectors (component wise)
    #[inline]
    fn add_assign(&mut self, v2: Self) {
        *self = *self + v2;
    }
}

impl Neg for DVec3 {
    type Output = Self;

    /// Negate
    ///
    /// Computed as `0 - v` like the C++ cross platform deterministic build (so -0 is represented as 0).
    #[inline]
    fn neg(self) -> Self {
        Self::new(0.0 - self.f64s[0], 0.0 - self.f64s[1], 0.0 - self.f64s[2])
    }
}

impl Sub<Vec3> for DVec3 {
    type Output = Self;

    /// Subtract two vectors (component wise)
    #[inline]
    fn sub(self, v2: Vec3) -> Self {
        Self::new(self.f64s[0] - v2.x() as f64, self.f64s[1] - v2.y() as f64, self.f64s[2] - v2.z() as f64)
    }
}

impl Sub for DVec3 {
    type Output = Self;

    /// Subtract two double vectors (component wise)
    #[inline]
    fn sub(self, v2: Self) -> Self {
        Self::new(self.f64s[0] - v2.f64s[0], self.f64s[1] - v2.f64s[1], self.f64s[2] - v2.f64s[2])
    }
}

impl SubAssign<Vec3> for DVec3 {
    /// Subtract two vectors (component wise)
    #[inline]
    fn sub_assign(&mut self, v2: Vec3) {
        *self = *self - v2;
    }
}

impl SubAssign for DVec3 {
    /// Subtract two double vectors (component wise)
    #[inline]
    fn sub_assign(&mut self, v2: Self) {
        *self = *self - v2;
    }
}

impl Div for DVec3 {
    type Output = Self;

    /// Divide (component wise)
    #[inline]
    fn div(self, v2: Self) -> Self {
        Self::new(self.f64s[0] / v2.f64s[0], self.f64s[1] / v2.f64s[1], self.f64s[2] / v2.f64s[2])
    }
}

impl fmt::Display for DVec3 {
    /// To String
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{}, {}, {}", self.f64s[0], self.f64s[1], self.f64s[2])
    }
}

// Create a JoltHash for DVec3
impl JoltHash for DVec3 {
    fn jolt_hash(&self) -> u64 {
        crate::hash_combine_args!(self.x(), self.y(), self.z())
    }
}
