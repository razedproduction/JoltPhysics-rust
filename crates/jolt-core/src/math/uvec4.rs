//! Port of `Jolt/Math/UVec4.h` + `UVec4.inl`.
//!
//! Scalar implementation. Where the C++ scalar fallback and the x86-64 SSE implementation (the platform of the
//! cross platform deterministic reference build) differ, this follows SSE; see `port/DEVIATIONS.md`.

use std::fmt;
use std::ops::{Add, AddAssign, Index, IndexMut, Mul, Sub, SubAssign};

use super::Vec4;
use super::swizzle::{SWIZZLE_W, SWIZZLE_X, SWIZZLE_Y, SWIZZLE_Z};

/// 4 component vector of unsigned 32 bit integers
#[derive(Clone, Copy, Debug, Default)]
#[repr(C, align(16))]
pub struct UVec4 {
    pub(crate) u32s: [u32; 4],
}

impl UVec4 {
    /// Create a vector from 4 integer components
    #[inline]
    pub const fn new(x: u32, y: u32, z: u32, w: u32) -> Self {
        Self { u32s: [x, y, z, w] }
    }

    /// Create a vector from an array of 4 integer components
    #[inline]
    pub const fn from_array(v: [u32; 4]) -> Self {
        Self { u32s: v }
    }

    /// Get the 4 components as an array
    #[inline]
    pub const fn to_array(self) -> [u32; 4] {
        self.u32s
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

        Self::new(self.u32s[X as usize], self.u32s[Y as usize], self.u32s[Z as usize], self.u32s[W as usize])
    }

    /// Vector with all zeros
    #[inline]
    pub const fn zero() -> Self {
        Self::new(0, 0, 0, 0)
    }

    /// Replicate int inV across all components
    #[inline]
    pub const fn replicate(v: u32) -> Self {
        Self::new(v, v, v, v)
    }

    /// Load 1 int from memory and place it in the X component, zeros Y, Z and W
    #[inline]
    pub fn load_int(v: &[u32]) -> Self {
        Self::new(v[0], 0, 0, 0)
    }

    /// Load 4 ints from memory
    #[inline]
    pub fn load_int4(v: &[u32]) -> Self {
        Self::new(v[0], v[1], v[2], v[3])
    }

    /// Load 4 ints from memory, aligned to 16 bytes
    #[inline]
    pub fn load_int4_aligned(v: &[u32]) -> Self {
        Self::new(v[0], v[1], v[2], v[3])
    }

    /// Gather 4 ints from memory at inBase + inOffsets[i] * Scale
    ///
    /// `SCALE` is in bytes like in C++ and must be a multiple of 4 (the size of an `u32`).
    #[inline]
    pub fn gather_int4<const SCALE: usize>(base: &[u32], offsets: UVec4) -> Self {
        const { assert!(SCALE % 4 == 0, "Scale must be a multiple of sizeof(uint32)") };
        let get = |offset: u32| base[offset as usize * SCALE / 4];
        Self::new(get(offsets.x()), get(offsets.y()), get(offsets.z()), get(offsets.w()))
    }

    /// Return the minimum value of each of the components
    #[inline]
    pub fn min(v1: Self, v2: Self) -> Self {
        Self::from_array(std::array::from_fn(|i| super::min(v1.u32s[i], v2.u32s[i])))
    }

    /// Return the maximum of each of the components
    #[inline]
    pub fn max(v1: Self, v2: Self) -> Self {
        Self::from_array(std::array::from_fn(|i| super::max(v1.u32s[i], v2.u32s[i])))
    }

    /// Equals (component wise)
    #[inline]
    pub fn equals(v1: Self, v2: Self) -> Self {
        Self::from_array(std::array::from_fn(|i| if v1.u32s[i] == v2.u32s[i] { 0xffffffff } else { 0 }))
    }

    /// Component wise select, returns inNotSet when highest bit of inControl = 0 and inSet when highest bit of inControl = 1
    #[inline]
    pub fn select(not_set: Self, set: Self, control: Self) -> Self {
        Self::from_array(std::array::from_fn(|i| {
            if (control.u32s[i] & 0x80000000) != 0 { set.u32s[i] } else { not_set.u32s[i] }
        }))
    }

    /// Logical or (component wise)
    #[inline]
    pub fn or(v1: Self, v2: Self) -> Self {
        Self::from_array(std::array::from_fn(|i| v1.u32s[i] | v2.u32s[i]))
    }

    /// Logical xor (component wise)
    #[inline]
    pub fn xor(v1: Self, v2: Self) -> Self {
        Self::from_array(std::array::from_fn(|i| v1.u32s[i] ^ v2.u32s[i]))
    }

    /// Logical and (component wise)
    #[inline]
    pub fn and(v1: Self, v2: Self) -> Self {
        Self::from_array(std::array::from_fn(|i| v1.u32s[i] & v2.u32s[i]))
    }

    /// Logical not (component wise)
    #[inline]
    #[allow(clippy::should_implement_trait)] // C++ `sNot`, a static function like `sAnd` / `sOr`
    pub fn not(v1: Self) -> Self {
        Self::from_array(std::array::from_fn(|i| !v1.u32s[i]))
    }

    /// Sorts the elements in inIndex so that the values that correspond to trues in inValue are the first elements.
    /// The remaining elements will be set to inValue.w.
    /// I.e. if inValue = (true, false, true, false) and inIndex = (1, 2, 3, 4) the function returns (1, 3, 4, 4).
    #[inline]
    pub fn sort4_true(value: Self, index: Self) -> Self {
        // If inValue.z is false then shift W to Z
        let mut v = Self::select(index.swizzle::<SWIZZLE_X, SWIZZLE_Y, SWIZZLE_W, SWIZZLE_W>(), index, value.splat_z());

        // If inValue.y is false then shift Z and further to Y and further
        v = Self::select(v.swizzle::<SWIZZLE_X, SWIZZLE_Z, SWIZZLE_W, SWIZZLE_W>(), v, value.splat_y());

        // If inValue.x is false then shift X and further to Y and further
        v = Self::select(v.swizzle::<SWIZZLE_Y, SWIZZLE_Z, SWIZZLE_W, SWIZZLE_W>(), v, value.splat_x());

        v
    }

    /// Get individual components
    #[inline]
    pub const fn x(self) -> u32 {
        self.u32s[0]
    }

    #[inline]
    pub const fn y(self) -> u32 {
        self.u32s[1]
    }

    #[inline]
    pub const fn z(self) -> u32 {
        self.u32s[2]
    }

    #[inline]
    pub const fn w(self) -> u32 {
        self.u32s[3]
    }

    /// Set individual components
    #[inline]
    pub fn set_x(&mut self, x: u32) {
        self.u32s[0] = x;
    }

    #[inline]
    pub fn set_y(&mut self, y: u32) {
        self.u32s[1] = y;
    }

    #[inline]
    pub fn set_z(&mut self, z: u32) {
        self.u32s[2] = z;
    }

    #[inline]
    pub fn set_w(&mut self, w: u32) {
        self.u32s[3] = w;
    }

    /// Replicate the X component to all components
    #[inline]
    pub const fn splat_x(self) -> Self {
        Self::replicate(self.u32s[0])
    }

    /// Replicate the Y component to all components
    #[inline]
    pub const fn splat_y(self) -> Self {
        Self::replicate(self.u32s[1])
    }

    /// Replicate the Z component to all components
    #[inline]
    pub const fn splat_z(self) -> Self {
        Self::replicate(self.u32s[2])
    }

    /// Replicate the W component to all components
    #[inline]
    pub const fn splat_w(self) -> Self {
        Self::replicate(self.u32s[3])
    }

    /// Convert each component from an int to a float
    ///
    /// Like `_mm_cvtepi32_ps` (the SSE implementation) the components are interpreted as signed integers.
    #[inline]
    pub fn to_float(self) -> Vec4 {
        Vec4::from_array(std::array::from_fn(|i| self.u32s[i] as i32 as f32))
    }

    /// Reinterpret UVec4 as a Vec4 (doesn't change the bits)
    #[inline]
    pub fn reinterpret_as_float(self) -> Vec4 {
        Vec4::from_array(std::array::from_fn(|i| f32::from_bits(self.u32s[i])))
    }

    /// Dot product, returns the dot product in X, Y, Z and W components
    #[inline]
    pub fn dot_v(self, v2: Self) -> Self {
        Self::replicate(self.dot(v2))
    }

    /// Dot product
    #[inline]
    pub fn dot(self, v2: Self) -> u32 {
        // Integer arithmetic wraps like in C++ and is associative, so the summation order does not matter
        let mul = self * v2;
        mul.u32s[0].wrapping_add(mul.u32s[1]).wrapping_add(mul.u32s[2]).wrapping_add(mul.u32s[3])
    }

    /// Store 4 ints to memory
    #[inline]
    pub const fn store_int4(self) -> [u32; 4] {
        self.u32s
    }

    /// Store 4 ints to memory, aligned to 16 bytes
    #[inline]
    pub const fn store_int4_aligned(self) -> [u32; 4] {
        self.u32s
    }

    /// Test if any of the components are true (true is when highest bit of component is set)
    #[inline]
    pub const fn test_any_true(self) -> bool {
        self.trues() != 0
    }

    /// Test if any of X, Y or Z components are true (true is when highest bit of component is set)
    #[inline]
    pub const fn test_any_xyz_true(self) -> bool {
        (self.trues() & 0b111) != 0
    }

    /// Test if all components are true (true is when highest bit of component is set)
    #[inline]
    pub const fn test_all_true(self) -> bool {
        self.trues() == 0b1111
    }

    /// Test if X, Y and Z components are true (true is when highest bit of component is set)
    #[inline]
    pub const fn test_all_xyz_true(self) -> bool {
        (self.trues() & 0b111) == 0b111
    }

    /// Count the number of components that are true (true is when highest bit of component is set)
    #[inline]
    pub const fn count_trues(self) -> i32 {
        ((self.u32s[0] >> 31) + (self.u32s[1] >> 31) + (self.u32s[2] >> 31) + (self.u32s[3] >> 31)) as i32
    }

    /// Store if X is true in bit 0, Y in bit 1, Z in bit 2 and W in bit 3 (true is when highest bit of component is set)
    #[inline]
    pub const fn trues(self) -> i32 {
        ((self.u32s[0] >> 31) | ((self.u32s[1] >> 31) << 1) | ((self.u32s[2] >> 31) << 2) | ((self.u32s[3] >> 31) << 3))
            as i32
    }

    /// Shift all components by Count bits to the left (filling with zeros from the left)
    #[inline]
    pub fn logical_shift_left<const COUNT: u32>(self) -> Self {
        const { assert!(COUNT <= 31, "Invalid shift") };
        Self::from_array(std::array::from_fn(|i| self.u32s[i] << COUNT))
    }

    /// Shift all components by Count bits to the right (filling with zeros from the right)
    #[inline]
    pub fn logical_shift_right<const COUNT: u32>(self) -> Self {
        const { assert!(COUNT <= 31, "Invalid shift") };
        Self::from_array(std::array::from_fn(|i| self.u32s[i] >> COUNT))
    }

    /// Shift all components by Count bits to the right (shifting in the value of the highest bit)
    #[inline]
    pub fn arithmetic_shift_right<const COUNT: u32>(self) -> Self {
        const { assert!(COUNT <= 31, "Invalid shift") };
        Self::from_array(std::array::from_fn(|i| ((self.u32s[i] as i32) >> COUNT) as u32))
    }

    /// Takes the lower 4 16 bits and expands them to X, Y, Z and W
    #[inline]
    pub const fn expand4_uint16_lo(self) -> Self {
        Self::new(
            self.u32s[0] & 0xffff,
            (self.u32s[0] >> 16) & 0xffff,
            self.u32s[1] & 0xffff,
            (self.u32s[1] >> 16) & 0xffff,
        )
    }

    /// Takes the upper 4 16 bits and expands them to X, Y, Z and W
    #[inline]
    pub const fn expand4_uint16_hi(self) -> Self {
        Self::new(
            self.u32s[2] & 0xffff,
            (self.u32s[2] >> 16) & 0xffff,
            self.u32s[3] & 0xffff,
            (self.u32s[3] >> 16) & 0xffff,
        )
    }

    #[inline]
    fn expand4_byte(v: u32) -> Self {
        Self::from_array(std::array::from_fn(|i| (v >> (i * 8)) & 0xff))
    }

    /// Takes byte 0 .. 3 and expands them to X, Y, Z and W
    #[inline]
    pub fn expand4_byte0(self) -> Self {
        Self::expand4_byte(self.u32s[0])
    }

    /// Takes byte 4 .. 7 and expands them to X, Y, Z and W
    #[inline]
    pub fn expand4_byte4(self) -> Self {
        Self::expand4_byte(self.u32s[1])
    }

    /// Takes byte 8 .. 11 and expands them to X, Y, Z and W
    #[inline]
    pub fn expand4_byte8(self) -> Self {
        Self::expand4_byte(self.u32s[2])
    }

    /// Takes byte 12 .. 15 and expands them to X, Y, Z and W
    #[inline]
    pub fn expand4_byte12(self) -> Self {
        Self::expand4_byte(self.u32s[3])
    }

    /// Shift vector components by 4 - Count floats to the left, so if Count = 1 the resulting vector is (W, 0, 0, 0), when Count = 3 the resulting vector is (Y, Z, W, 0)
    #[inline]
    pub fn shift_components4_minus(self, count: i32) -> Self {
        debug_assert!((0..=4).contains(&count));
        let count = count as usize;
        let mut result = Self::zero();
        for i in 0..count {
            result.u32s[i] = self.u32s[i + 4 - count];
        }
        result
    }
}

impl PartialEq for UVec4 {
    /// Comparison
    #[inline]
    fn eq(&self, v2: &Self) -> bool {
        Self::equals(*self, *v2).test_all_true()
    }
}

impl Eq for UVec4 {}

impl Index<usize> for UVec4 {
    type Output = u32;

    /// Get component by index
    #[inline]
    fn index(&self, coordinate: usize) -> &u32 {
        debug_assert!(coordinate < 4);
        &self.u32s[coordinate]
    }
}

impl IndexMut<usize> for UVec4 {
    #[inline]
    fn index_mut(&mut self, coordinate: usize) -> &mut u32 {
        debug_assert!(coordinate < 4);
        &mut self.u32s[coordinate]
    }
}

impl Mul for UVec4 {
    type Output = Self;

    /// Component wise multiplication of two integer vectors (stores low 32 bits of result only)
    #[inline]
    fn mul(self, v2: Self) -> Self {
        Self::from_array(std::array::from_fn(|i| self.u32s[i].wrapping_mul(v2.u32s[i])))
    }
}

impl Add for UVec4 {
    type Output = Self;

    /// Add two integer vectors (component wise)
    #[inline]
    fn add(self, v2: Self) -> Self {
        Self::from_array(std::array::from_fn(|i| self.u32s[i].wrapping_add(v2.u32s[i])))
    }
}

impl AddAssign for UVec4 {
    /// Add two integer vectors (component wise)
    #[inline]
    fn add_assign(&mut self, v2: Self) {
        *self = *self + v2;
    }
}

impl Sub for UVec4 {
    type Output = Self;

    /// Subtract two integer vectors (component wise)
    #[inline]
    fn sub(self, v2: Self) -> Self {
        Self::from_array(std::array::from_fn(|i| self.u32s[i].wrapping_sub(v2.u32s[i])))
    }
}

impl SubAssign for UVec4 {
    /// Subtract two integer vectors (component wise)
    #[inline]
    fn sub_assign(&mut self, v2: Self) {
        *self = *self - v2;
    }
}

impl fmt::Display for UVec4 {
    /// To String
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{}, {}, {}, {}", self.u32s[0], self.u32s[1], self.u32s[2], self.u32s[3])
    }
}
