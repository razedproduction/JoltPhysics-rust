//! Port of `Jolt/Math/Float3.h`.

use std::ops::Index;

/// Class that holds 3 floats. Used as a storage class. Convert to Vec3 for calculations.
///
/// C++ leaves a default constructed `Float3` uninitialized, [`Default`] zeroes it.
#[repr(C)]
#[derive(Clone, Copy, Debug, Default, PartialEq)]
pub struct Float3 {
    pub x: f32,
    pub y: f32,
    pub z: f32,
}

impl Float3 {
    /// Construct from components
    #[inline]
    pub const fn new(x: f32, y: f32, z: f32) -> Self {
        Self { x, y, z }
    }
}

impl Index<usize> for Float3 {
    type Output = f32;

    #[inline]
    fn index(&self, coordinate: usize) -> &f32 {
        match coordinate {
            0 => &self.x,
            1 => &self.y,
            2 => &self.z,
            _ => panic!("Float3 coordinate {coordinate} out of range"),
        }
    }
}

pub type VertexList = Vec<Float3>;

// Create a JoltHash for Float3
crate::make_hashable!(Float3, x, y, z);
