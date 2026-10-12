//! Port of UnitTests/Math/TrigonometryTests.cpp

use jolt_core::math::{JPH_PI, Vec4, acos, acos_approximate, asin, atan, atan2, cos, sin, tan};
use jolt_test_utils::check_approx_equal_f32;

#[test]
fn test_a_cos_approximate() {
    // Check error over entire range [-1, 1]
    for i in -1000..=1000 {
        let x = i as f32 / 1000.0;
        let acos1 = x.acos();
        let acos2 = acos_approximate(x);
        check_approx_equal_f32(acos1, acos2, 4.3e-3);
    }

    // Check edge cases for exact matches
    assert_eq!(acos_approximate(1.0), 0.0);
    assert_eq!(acos_approximate(1.0e-12), JPH_PI / 2.0);
    assert_eq!(acos_approximate(-1.0e-12), JPH_PI / 2.0);
    assert_eq!(acos_approximate(-1.0), JPH_PI);
}

#[test]
fn test_scalar_functions_use_vec4_lane0() {
    // Not in the C++ tests: the scalar functions evaluate lane 0 of the Vec4 implementations
    for i in -100..=100 {
        let x = i as f32 * 0.0731;
        let (s, c) = Vec4::replicate(x).sin_cos();
        assert_eq!(sin(x).to_bits(), s.x().to_bits());
        assert_eq!(cos(x).to_bits(), c.x().to_bits());
        assert_eq!(tan(x).to_bits(), Vec4::replicate(x).tan().x().to_bits());
        assert_eq!(asin(x).to_bits(), Vec4::replicate(x).asin().x().to_bits());
        assert_eq!(acos(x).to_bits(), Vec4::replicate(x).acos().x().to_bits());
        assert_eq!(atan(x).to_bits(), Vec4::replicate(x).atan().x().to_bits());
        assert_eq!(atan2(x, 0.5).to_bits(), Vec4::atan2(Vec4::replicate(x), Vec4::replicate(0.5)).x().to_bits());
        check_approx_equal_f32(sin(x), x.sin(), 1.0e-6);
    }
}
