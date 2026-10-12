//! Shared helpers for porting `UnitTests/` (port of `UnitTestFramework.h`, `PhysicsTestContext`,
//! `Layers.h` and the `Logging*Listener.h` helpers). Grows together with the crates it tests.

/// Port of `CHECK_APPROX_EQUAL(float, float, tolerance)`: `|rhs - lhs| <= tolerance`.
#[track_caller]
pub fn check_approx_equal_f32(lhs: f32, rhs: f32, tolerance: f32) {
    assert!((rhs - lhs).abs() <= tolerance, "{lhs} != {rhs} (tolerance {tolerance})");
}

/// Port of `CHECK_APPROX_EQUAL(double, double, tolerance)`: `|rhs - lhs| <= tolerance`.
#[track_caller]
pub fn check_approx_equal_f64(lhs: f64, rhs: f64, tolerance: f64) {
    assert!((rhs - lhs).abs() <= tolerance, "{lhs} != {rhs} (tolerance {tolerance})");
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn approx_equal_accepts_within_tolerance() {
        check_approx_equal_f32(1.0, 1.0 + 0.5e-6, 1.0e-6);
        check_approx_equal_f64(1.0, 1.0 + 0.5e-6, 1.0e-6);
    }

    #[test]
    #[should_panic]
    fn approx_equal_rejects_outside_tolerance() {
        check_approx_equal_f32(1.0, 1.1, 1.0e-6);
    }
}
