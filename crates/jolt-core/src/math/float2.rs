//! Port of `Jolt/Math/Float2.h`.

use std::fmt;

/// Class that holds 2 floats, used as a storage class mainly.
///
/// C++ leaves a default constructed `Float2` uninitialized, [`Default`] zeroes it.
#[repr(C)]
#[derive(Clone, Copy, Debug, Default, PartialEq)]
pub struct Float2 {
    pub x: f32,
    pub y: f32,
}

impl Float2 {
    /// Construct from components
    #[inline]
    pub const fn new(x: f32, y: f32) -> Self {
        Self { x, y }
    }
}

/// To String
impl fmt::Display for Float2 {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{}, {}", self.x, self.y)
    }
}
