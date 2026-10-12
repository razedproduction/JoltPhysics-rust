//! Port of `Jolt/Math/Swizzle.h`.
//!
//! Enum indicating which component to use when swizzling. C++ uses an unnamed enum as template argument; the swizzle
//! functions take `const u32` generic parameters, so these are plain `u32` constants.

/// Use the X component
pub const SWIZZLE_X: u32 = 0;
/// Use the Y component
pub const SWIZZLE_Y: u32 = 1;
/// Use the Z component
pub const SWIZZLE_Z: u32 = 2;
/// Use the W component
pub const SWIZZLE_W: u32 = 3;
/// We always use the Z component when we don't specifically want to initialize a value, this is consistent with what
/// is done in Vec3(x, y, z), Vec3(Float3 &) and Vec3::sLoadFloat3Unsafe
pub const SWIZZLE_UNUSED: u32 = 2;
