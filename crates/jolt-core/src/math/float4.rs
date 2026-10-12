//! Port of `Jolt/Math/Float4.h`.

use std::ops::Index;

/// Class that holds 4 float values. Convert to Vec4 to perform calculations.
///
/// C++ leaves a default constructed `Float4` uninitialized, [`Default`] zeroes it.
#[repr(C)]
#[derive(Clone, Copy, Debug, Default, PartialEq)]
pub struct Float4 {
    pub x: f32,
    pub y: f32,
    pub z: f32,
    pub w: f32,
}

impl Float4 {
    /// Construct from components
    #[inline]
    pub const fn new(x: f32, y: f32, z: f32, w: f32) -> Self {
        Self { x, y, z, w }
    }
}

impl Index<usize> for Float4 {
    type Output = f32;

    #[inline]
    fn index(&self, coordinate: usize) -> &f32 {
        match coordinate {
            0 => &self.x,
            1 => &self.y,
            2 => &self.z,
            3 => &self.w,
            _ => panic!("Float4 coordinate {coordinate} out of range"),
        }
    }
}
