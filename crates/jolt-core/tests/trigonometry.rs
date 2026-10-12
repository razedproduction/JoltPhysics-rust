//! Port of UnitTests/Math/TrigonometryTests.cpp

use jolt_core::math::{JPH_PI, acos_approximate};
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
