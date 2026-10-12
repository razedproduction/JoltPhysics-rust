//! Port of `Jolt/Math/BVec16.h` + `BVec16.inl` (scalar implementation only).
//!
//! Slice 2 ports the subset needed by [`HashTable`](crate::hash_table::HashTable); the remaining operations
//! (`sOr`, `sXor`, `sNot`, `TestAnyTrue`, `TestAllTrue`, `operator ==`, indexing) and `BVec16Tests.cpp` follow in
//! slice 4 together with the SIMD backends. The scalar path is the C++ fallback for platforms without SSE/NEON; it
//! represents a "true" component by its highest bit only, exactly like the C++ fallback.

/// A vector consisting of 16 bytes
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
#[repr(C, align(16))]
pub struct BVec16 {
    /// The 16 bytes as two little endian `u64`s (C++ union member `mU64`; byte `i` is `mU8[i]`)
    u64s: [u64; 2],
}

impl BVec16 {
    /// Create a vector from 16 bytes
    #[inline]
    pub const fn from_bytes(bytes: [u8; 16]) -> Self {
        let mut lo = [0u8; 8];
        let mut hi = [0u8; 8];
        let mut i = 0;
        while i < 8 {
            lo[i] = bytes[i];
            hi[i] = bytes[i + 8];
            i += 1;
        }
        Self::from_u64(u64::from_le_bytes(lo), u64::from_le_bytes(hi))
    }

    /// Create a vector from two uint64's
    #[inline]
    pub const fn from_u64(v0: u64, v1: u64) -> Self {
        Self { u64s: [v0, v1] }
    }

    /// Vector with all zeros
    #[inline]
    pub const fn zero() -> Self {
        Self::from_u64(0, 0)
    }

    /// Replicate int `v` across all components
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
    #[inline]
    pub const fn equals(v1: Self, v2: Self) -> Self {
        const fn equals(v1: u64, v2: u64) -> u64 {
            let mut r = v1 ^ !v2; // Bits that are equal are 1
            r &= r << 1; // Combine bit 0 through 1
            r &= r << 2; // Combine bit 0 through 3
            r &= r << 4; // Combine bit 0 through 7
            r &= 0x8080808080808080; // Keep only the highest bit of each byte
            r
        }
        Self::from_u64(equals(v1.u64s[0], v2.u64s[0]), equals(v1.u64s[1], v2.u64s[1]))
    }

    /// Logical and (component wise)
    #[inline]
    pub const fn and(v1: Self, v2: Self) -> Self {
        Self::from_u64(v1.u64s[0] & v2.u64s[0], v1.u64s[1] & v2.u64s[1])
    }

    /// Get the 16 bytes (C++ union member `mU8`)
    #[inline]
    pub const fn to_bytes(self) -> [u8; 16] {
        let lo = self.u64s[0].to_le_bytes();
        let hi = self.u64s[1].to_le_bytes();
        let mut bytes = [0u8; 16];
        let mut i = 0;
        while i < 8 {
            bytes[i] = lo[i];
            bytes[i + 8] = hi[i];
            i += 1;
        }
        bytes
    }

    /// Store if mU8[0] is true in bit 0, mU8[1] in bit 1, etc. (true is when highest bit of component is set)
    #[inline]
    pub const fn trues(self) -> i32 {
        let bytes = self.to_bytes();
        let mut result = 0;
        let mut i = 0;
        while i < 16 {
            result |= ((bytes[i] >> 7) as i32) << i;
            i += 1;
        }
        result
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
}
