//! Port of `Jolt/Math/Double3.h`.

use std::ops::Index;

/// Class that holds 3 doubles. Used as a storage class. Convert to DVec3 for calculations.
///
/// C++ leaves a default constructed `Double3` uninitialized, [`Default`] zeroes it.
#[repr(C)]
#[derive(Clone, Copy, Debug, Default, PartialEq)]
pub struct Double3 {
    pub x: f64,
    pub y: f64,
    pub z: f64,
}

impl Double3 {
    /// Construct from components
    #[inline]
    pub const fn new(x: f64, y: f64, z: f64) -> Self {
        Self { x, y, z }
    }
}

impl Index<usize> for Double3 {
    type Output = f64;

    #[inline]
    fn index(&self, coordinate: usize) -> &f64 {
        match coordinate {
            0 => &self.x,
            1 => &self.y,
            2 => &self.z,
            _ => panic!("Double3 coordinate {coordinate} out of range"),
        }
    }
}

// Create a JoltHash for Double3
crate::make_hashable!(Double3, x, y, z);
