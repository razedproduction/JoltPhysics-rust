//! Port of UnitTests/Core/LinearCurveTest.cpp

use jolt_core::LinearCurve;
use jolt_test_utils::check_approx_equal;

#[test]
fn test0_point_curve() {
    let curve = LinearCurve::default();

    assert_eq!(curve.value(1.0), 0.0);
}

#[test]
fn test1_point_curve() {
    let mut curve = LinearCurve::default();
    curve.add_point(1.0, 20.0);

    assert_eq!(curve.value(0.9), 20.0);
    assert_eq!(curve.value(1.0), 20.0);
    assert_eq!(curve.value(1.1), 20.0);
}

#[test]
fn test2_point_curve() {
    let mut curve = LinearCurve::default();
    curve.add_point(-1.0, 40.0);
    curve.add_point(-3.0, 20.0);
    curve.sort();

    check_approx_equal(curve.value(-3.1), 20.0);
    check_approx_equal(curve.value(-3.0), 20.0);
    check_approx_equal(curve.value(-2.0), 30.0);
    check_approx_equal(curve.value(-1.0), 40.0);
    check_approx_equal(curve.value(-0.9), 40.0);
}

#[test]
fn test3_point_curve() {
    let mut curve = LinearCurve::default();
    curve.add_point(1.0, 20.0);
    curve.add_point(5.0, 60.0);
    curve.add_point(3.0, 40.0);
    curve.sort();

    check_approx_equal(curve.value(0.9), 20.0);
    check_approx_equal(curve.value(1.0), 20.0);
    check_approx_equal(curve.value(2.0), 30.0);
    check_approx_equal(curve.value(3.0), 40.0);
    check_approx_equal(curve.value(4.0), 50.0);
    check_approx_equal(curve.value(5.0), 60.0);
    check_approx_equal(curve.value(5.1), 60.0);
}
