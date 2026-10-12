//! Port of `Jolt/Math/BVec16.h` + `BVec16.inl`.
//!
//! Scalar implementation that works on two `u64`s like the C++ fallback. `sEquals` follows the SSE implementation
//! (`_mm_cmpeq_epi8`: 0xff for equal bytes, the C++ fallback only sets the highest bit); all implementations agree on
//! `GetTrues`, `TestAnyTrue` and `TestAllTrue`. See `port/DEVIATIONS.md`.

use std::fmt;
use std::ops::{Index, IndexMut};

/// A vector consisting of 16 bytes
#[derive(Clone, Copy, Debug, Default)]
#[repr(C, align(16))]
pub struct BVec16 {
    /// The 16 bytes (C++ union member `mU8`; `mU64[i]` are bytes `8 * i .. 8 * i + 8` in little endian order)
    u8s: [u8; 16],
}

impl BVec16 {
    /// Create a vector from 16 bytes
    #[inline]
    pub const fn new(
        b0: u8,
        b1: u8,
        b2: u8,
        b3: u8,
        b4: u8,
        b5: u8,
        b6: u8,
        b7: u8,
        b8: u8,
        b9: u8,
        b10: u8,
        b11: u8,
        b12: u8,
        b13: u8,
        b14: u8,
        b15: u8,
    ) -> Self {
        Self::from_bytes([b0, b1, b2, b3, b4, b5, b6, b7, b8, b9, b10, b11, b12, b13, b14, b15])
    }

    /// Create a vector from an array of 16 bytes
    #[inline]
    pub const fn from_bytes(bytes: [u8; 16]) -> Self {
        Self { u8s: bytes }
    }

    /// Create a vector from two uint64's
    #[inline]
    pub const fn from_u64(v0: u64, v1: u64) -> Self {
        let lo = v0.to_le_bytes();
        let hi = v1.to_le_bytes();
        let mut bytes = [0u8; 16];
        let mut i = 0;
        while i < 8 {
            bytes[i] = lo[i];
            bytes[i + 8] = hi[i];
            i += 1;
        }
        Self::from_bytes(bytes)
    }

    /// Get the 16 bytes (C++ union member `mU8`)
    #[inline]
    pub const fn to_bytes(self) -> [u8; 16] {
        self.u8s
    }

    /// Get the two uint64's (C++ union member `mU64`)
    #[inline]
    pub const fn to_u64(self) -> [u64; 2] {
        let mut lo = [0u8; 8];
        let mut hi = [0u8; 8];
        let mut i = 0;
        while i < 8 {
            lo[i] = self.u8s[i];
            hi[i] = self.u8s[i + 8];
            i += 1;
        }
        [u64::from_le_bytes(lo), u64::from_le_bytes(hi)]
    }

    /// Vector with all zeros
    #[inline]
    pub const fn zero() -> Self {
        Self::from_u64(0, 0)
    }

    /// Replicate int inV across all components
    #[inline]
    pub const fn replicate(v: u8) -> Self {
        let mut v = v as u64;
        v |= v << 8;
        v |= v << 16;
        v |= v << 32;
        Self::from_u64(v, v)
    }

    /// Load 16 bytes from memory (`v` must hold at least 16 bytes, only the first 16 are read)
    #[inline]
    pub fn load_byte16(v: &[u8]) -> Self {
        let bytes: [u8; 16] = v[..16].try_into().expect("slice has 16 bytes");
        Self::from_bytes(bytes)
    }

    /// Equals (component wise), highest bit of each component that is set is considered true
    ///
    /// Like `_mm_cmpeq_epi8` (the SSE implementation) equal components are 0xff.
    #[inline]
    pub const fn equals(v1: Self, v2: Self) -> Self {
        const fn equals(v1: u64, v2: u64) -> u64 {
            let mut r = v1 ^ !v2; // Bits that are equal are 1
            r &= r << 1; // Combine bit 0 through 1
            r &= r << 2; // Combine bit 0 through 3
            r &= r << 4; // Combine bit 0 through 7
            r &= 0x8080808080808080; // Keep only the highest bit of each byte
            (r >> 7) * 0xff // Set all bits of a true byte (no carries: each byte of r >> 7 is 0 or 1)
        }
        let a = v1.to_u64();
        let b = v2.to_u64();
        Self::from_u64(equals(a[0], b[0]), equals(a[1], b[1]))
    }

    /// Logical or (component wise)
    #[inline]
    pub const fn or(v1: Self, v2: Self) -> Self {
        let a = v1.to_u64();
        let b = v2.to_u64();
        Self::from_u64(a[0] | b[0], a[1] | b[1])
    }

    /// Logical xor (component wise)
    #[inline]
    pub const fn xor(v1: Self, v2: Self) -> Self {
        let a = v1.to_u64();
        let b = v2.to_u64();
        Self::from_u64(a[0] ^ b[0], a[1] ^ b[1])
    }

    /// Logical and (component wise)
    #[inline]
    pub const fn and(v1: Self, v2: Self) -> Self {
        let a = v1.to_u64();
        let b = v2.to_u64();
        Self::from_u64(a[0] & b[0], a[1] & b[1])
    }

    /// Logical not (component wise)
    #[inline]
    #[allow(clippy::should_implement_trait)] // C++ `sNot`, a static function like `sAnd` / `sOr`
    pub const fn not(v1: Self) -> Self {
        let a = v1.to_u64();
        Self::from_u64(!a[0], !a[1])
    }

    /// Test if any of the components are true (true is when highest bit of component is set)
    #[inline]
    pub const fn test_any_true(self) -> bool {
        let a = self.to_u64();
        ((a[0] | a[1]) & 0x8080808080808080) != 0
    }

    /// Test if all components are true (true is when highest bit of component is set)
    #[inline]
    pub const fn test_all_true(self) -> bool {
        let a = self.to_u64();
        ((a[0] & a[1]) & 0x8080808080808080) == 0x8080808080808080
    }

    /// Store if mU8[0] is true in bit 0, mU8[1] in bit 1, etc. (true is when highest bit of component is set)
    #[inline]
    pub const fn trues(self) -> i32 {
        let mut result = 0;
        let mut i = 0;
        while i < 16 {
            result |= ((self.u8s[i] >> 7) as i32) << i;
            i += 1;
        }
        result
    }
}

impl PartialEq for BVec16 {
    /// Comparison
    #[inline]
    fn eq(&self, v2: &Self) -> bool {
        Self::equals(*self, *v2).test_all_true()
    }
}

impl Eq for BVec16 {}

impl Index<usize> for BVec16 {
    type Output = u8;

    /// Get component by index
    #[inline]
    fn index(&self, coordinate: usize) -> &u8 {
        &self.u8s[coordinate]
    }
}

impl IndexMut<usize> for BVec16 {
    #[inline]
    fn index_mut(&mut self, coordinate: usize) -> &mut u8 {
        &mut self.u8s[coordinate]
    }
}

impl fmt::Display for BVec16 {
    /// To String
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        for (i, b) in self.u8s.iter().enumerate() {
            if i > 0 {
                f.write_str(", ")?;
            }
            write!(f, "{b}")?;
        }
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn equals_and_trues() {
        let mut bytes = [0u8; 16];
        bytes[0] = 0x81;
        bytes[5] = 0x81;
        bytes[15] = 0x81;
        let v = BVec16::from_bytes(bytes);
        assert_eq!(v.to_bytes(), bytes);
        assert_eq!(BVec16::equals(v, BVec16::replicate(0x81)).trues(), 0b1000_0000_0010_0001);
        assert_eq!(BVec16::equals(v, BVec16::zero()).trues(), 0b0111_1111_1101_1110);
        assert_eq!(BVec16::and(v, BVec16::replicate(0x80)).trues(), 0b1000_0000_0010_0001);
        assert_eq!(BVec16::load_byte16(&[0xff; 20]).trues(), 0xffff);
    }

    #[test]
    fn equals_sets_whole_byte() {
        let eq = BVec16::equals(BVec16::replicate(3), BVec16::from_u64(0x0303, 0));
        assert_eq!(eq.to_u64(), [0xffff, 0]);
    }
}
