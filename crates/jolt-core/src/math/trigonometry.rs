//! Port of `Jolt/Math/Trigonometry.h`.
//!
//! Note that this file exists because std::sin etc. are not platform independent and will lead to non-deterministic
//! simulation.
//!
//! `Sin`, `Cos`, `Tan`, `ASin`, `ACos`, `ATan` and `ATan2` evaluate lane 0 of the `Vec4` implementations
//! (`Vec4::SinCos`, `Vec4::Tan`, ...) and are ported together with `Vec4` in slice 4.

use super::{JPH_PI, min, sqrt};

/// An approximation of ACos, max error is 4.2e-3 over the entire range [-1, 1], is approximately 2.5x faster than ACos
#[inline]
pub fn acos_approximate(x: f32) -> f32 {
    // See: https://www.johndcook.com/blog/2022/09/06/inverse-cosine-near-1/
    // See also: https://seblagarde.wordpress.com/2014/12/01/inverse-trigonometric-functions-gpu-optimization-for-amd-gcn-architecture/
    // Taylor of cos(x) = 1 - x^2 / 2 + ...
    // Substitute x = sqrt(2 y) we get: cos(sqrt(2 y)) = 1 - y
    // Substitute z = 1 - y we get: cos(sqrt(2 (1 - z))) = z <=> acos(z) = sqrt(2 (1 - z))
    // To avoid the discontinuity at 1, instead of using the Taylor expansion of acos(x) we use acos(x) / sqrt(2 (1 - x)) = 1 + (1 - x) / 12 + ...
    // Since the approximation was made at 1, it has quite a large error at 0 meaning that if we want to extend to the
    // range [-1, 1] by mirroring the range [0, 1], the value at 0+ is not the same as 0-.
    // So we observe that the form of the Taylor expansion is f(x) = sqrt(1 - x) * (a + b x) and we fit the function so that f(0) = pi / 2
    // this gives us a = pi / 2. f(1) = 0 regardless of b. We search for a constant b that minimizes the error in the range [0, 1].
    let abs_x = min(x.abs(), 1.0); // Ensure that we don't get a value larger than 1
    let val = sqrt(1.0 - abs_x) * (JPH_PI / 2.0 - 0.175394 * abs_x);

    // Our approximation is valid in the range [0, 1], extend it to the range [-1, 1]
    if x < 0.0 { JPH_PI - val } else { val }
}
