//! Port of UnitTests/Math/DVec3Tests.cpp
//!
//! The `JPH_CROSS_PLATFORM_DETERMINISTIC` sections are always ported (the port behaves like that build).

// The expected values are written like in the C++ test (e.g. `1 * 4 + 7`)
#![allow(clippy::identity_op)]

use jolt_core::math::*;

#[test]
fn test_d_vec3_zero() {
    let v = DVec3::zero();

    assert_eq!(v.x(), 0.0);
    assert_eq!(v.y(), 0.0);
    assert_eq!(v.z(), 0.0);
}

#[test]
fn test_d_vec3_axis() {
    assert_eq!(DVec3::axis_x(), DVec3::new(1.0, 0.0, 0.0));
    assert_eq!(DVec3::axis_y(), DVec3::new(0.0, 1.0, 0.0));
    assert_eq!(DVec3::axis_z(), DVec3::new(0.0, 0.0, 1.0));
}

#[test]
fn test_vec3_na_n() {
    let mut v = DVec3::nan();

    assert!(v.x().is_nan());
    assert!(v.y().is_nan());
    assert!(v.z().is_nan());
    assert!(v.is_nan());

    v.set_component(0, 0.0);
    assert!(v.is_nan());
    v.set_component(1, 0.0);
    assert!(v.is_nan());
    v.set_component(2, 0.0);
    assert!(!v.is_nan());
}

#[test]
fn test_d_vec3_construct_components() {
    let mut v = DVec3::new(1.0, 2.0, 3.0);

    // Test component access
    assert_eq!(v.x(), 1.0);
    assert_eq!(v.y(), 2.0);
    assert_eq!(v.z(), 3.0);

    // Test component access by [] operators
    assert_eq!(v[0], 1.0);
    assert_eq!(v[1], 2.0);
    assert_eq!(v[2], 3.0);

    // Test == and != operators
    assert!(v == DVec3::new(1.0, 2.0, 3.0));
    assert!(v != DVec3::new(1.0, 2.0, 4.0));

    // Set the components
    v.set_component(0, 4.0);
    v.set_component(1, 5.0);
    v.set_component(2, 6.0);
    assert_eq!(v, DVec3::new(4.0, 5.0, 6.0));

    // Set the components again
    v.set_x(7.0);
    v.set_y(8.0);
    v.set_z(9.0);
    assert_eq!(v, DVec3::new(7.0, 8.0, 9.0));

    // Set all components
    v.set(10.0, 11.0, 12.0);
    assert_eq!(v, DVec3::new(10.0, 11.0, 12.0));
}

#[test]
fn test_vec4_to_d_vec3() {
    assert_eq!(DVec3::from_vec4(Vec4::new(1.0, 3.0, 5.0, 7.0)), DVec3::new(1.0, 3.0, 5.0));
}

#[test]
fn test_d_vec3_replicate() {
    assert_eq!(DVec3::replicate(2.0), DVec3::new(2.0, 2.0, 2.0));
}

#[test]
fn test_d_vec3_to_vec3() {
    assert_eq!(DVec3::new(1.0, 3.0, 5.0).to_vec3(), Vec3::new(1.0, 3.0, 5.0));

    // Check rounding up and down
    let above_2 = f64::from_bits(0x4000000000000001); // 0x1.0000000000001p1
    let above_2_f = f32::from_bits(0x40000001); // 0x1.000002p1f
    assert_eq!(DVec3::new(2.0, above_2, -above_2).to_vec3_round_up(), Vec3::new(2.0, above_2_f, -2.0));
    assert_eq!(DVec3::new(2.0, above_2, -above_2).to_vec3_round_down(), Vec3::new(2.0, 2.0, -above_2_f));
}

#[test]
fn test_vec3_min_max() {
    let v1 = DVec3::new(1.0, 5.0, 3.0);
    let v2 = DVec3::new(4.0, 2.0, 6.0);

    assert_eq!(DVec3::min(v1, v2), DVec3::new(1.0, 2.0, 3.0));
    assert_eq!(DVec3::max(v1, v2), DVec3::new(4.0, 5.0, 6.0));
}

#[test]
fn test_d_vec3_clamp() {
    let v1 = DVec3::new(1.0, 2.0, 3.0);
    let v2 = DVec3::new(4.0, 5.0, 6.0);
    let v = DVec3::new(-1.0, 3.0, 7.0);

    assert_eq!(DVec3::clamp(v, v1, v2), DVec3::new(1.0, 3.0, 6.0));
}

#[test]
fn test_d_vec3_trues() {
    for bits in 0..8i32 {
        let c = |i: i32| if bits & (1 << i) != 0 { DVec3::TRUE } else { DVec3::FALSE };
        let v = DVec3::new(c(0), c(1), c(2));
        assert_eq!(v.trues(), bits);
        assert_eq!(v.test_any_true(), bits != 0);
        assert_eq!(v.test_all_true(), bits == 0b111);
    }
}

#[test]
fn test_d_vec3_comparisons() {
    // Can't directly check if equal to (true, false, true) because true = -NaN and -NaN != -NaN
    assert_eq!(DVec3::equals(DVec3::new(1.0, 2.0, 3.0), DVec3::new(1.0, 4.0, 3.0)).trues(), 0b101);
    assert_eq!(DVec3::less(DVec3::new(1.0, 2.0, 4.0), DVec3::new(1.0, 4.0, 3.0)).trues(), 0b010);
    assert_eq!(DVec3::less_or_equal(DVec3::new(1.0, 2.0, 4.0), DVec3::new(1.0, 4.0, 3.0)).trues(), 0b011);
    assert_eq!(DVec3::greater(DVec3::new(1.0, 2.0, 4.0), DVec3::new(1.0, 4.0, 3.0)).trues(), 0b100);
    assert_eq!(DVec3::greater_or_equal(DVec3::new(1.0, 2.0, 4.0), DVec3::new(1.0, 4.0, 3.0)).trues(), 0b101);
}

#[test]
fn test_d_vec3_fma() {
    assert_eq!(
        DVec3::fused_multiply_add(DVec3::new(1.0, 2.0, 3.0), DVec3::new(4.0, 5.0, 6.0), DVec3::new(7.0, 8.0, 9.0)),
        DVec3::new((1 * 4 + 7) as f64, (2 * 5 + 8) as f64, (3 * 6 + 9) as f64)
    );
}

#[test]
fn test_d_vec3_select() {
    let true2 = f64::from_bits(1u64 << 63);
    let false2 = f64::from_bits(!0u64 >> 1);
    let a = DVec3::new(1.0, 2.0, 3.0);
    let b = DVec3::new(4.0, 5.0, 6.0);
    assert_eq!(DVec3::select(a, b, DVec3::new(DVec3::TRUE, DVec3::FALSE, DVec3::TRUE)), DVec3::new(4.0, 2.0, 6.0));
    assert_eq!(DVec3::select(a, b, DVec3::new(DVec3::FALSE, DVec3::TRUE, DVec3::FALSE)), DVec3::new(1.0, 5.0, 3.0));
    assert_eq!(DVec3::select(a, b, DVec3::new(true2, false2, true2)), DVec3::new(4.0, 2.0, 6.0));
    assert_eq!(DVec3::select(a, b, DVec3::new(false2, true2, false2)), DVec3::new(1.0, 5.0, 3.0));
}

#[test]
fn test_d_vec3_bit_ops() {
    let d = |bits: u64| f64::from_bits(bits);

    // Test all bit permutations
    let v1 = DVec3::new(d(0b0011), d(0b00110), d(0b001100));
    let v2 = DVec3::new(d(0b0101), d(0b01010), d(0b010100));

    assert_eq!(DVec3::or(v1, v2), DVec3::new(d(0b0111), d(0b01110), d(0b011100)));
    assert_eq!(DVec3::xor(v1, v2), DVec3::new(d(0b0110), d(0b01100), d(0b011000)));
    assert_eq!(DVec3::and(v1, v2), DVec3::new(d(0b0001), d(0b00010), d(0b000100)));
}

#[test]
fn test_d_vec3_close() {
    assert!(DVec3::new(1.0, 2.0, 3.0).is_close(DVec3::new(1.001, 2.001, 3.001), 1.0e-4));
    assert!(!DVec3::new(1.0, 2.0, 3.0).is_close(DVec3::new(1.001, 2.001, 3.001), 1.0e-6));

    assert!(DVec3::new(1.001, 0.0, 0.0).is_normalized(1.0e-2));
    assert!(!DVec3::new(0.0, 1.001, 0.0).is_normalized(1.0e-4));

    assert!(DVec3::new(-1.0e-7, 1.0e-7, 1.0e-8).is_near_zero(1.0e-12));
    assert!(!DVec3::new(-1.0e-7, 1.0e-7, -1.0e-5).is_near_zero(1.0e-12));
}

#[test]
fn test_d_vec3_operators() {
    assert_eq!(-DVec3::new(1.0, 2.0, 3.0), DVec3::new(-1.0, -2.0, -3.0));

    let neg_zero = -DVec3::zero();
    assert_eq!(neg_zero, DVec3::zero());

    // When cross platform deterministic, we want to make sure that -0 is represented as 0
    assert_eq!(neg_zero.x().to_bits(), 0);
    assert_eq!(neg_zero.y().to_bits(), 0);
    assert_eq!(neg_zero.z().to_bits(), 0);

    assert_eq!(DVec3::new(1.0, 2.0, 3.0) + Vec3::new(4.0, 5.0, 6.0), DVec3::new(5.0, 7.0, 9.0));
    assert_eq!(DVec3::new(1.0, 2.0, 3.0) - Vec3::new(6.0, 5.0, 4.0), DVec3::new(-5.0, -3.0, -1.0));

    assert_eq!(DVec3::new(1.0, 2.0, 3.0) + DVec3::new(4.0, 5.0, 6.0), DVec3::new(5.0, 7.0, 9.0));
    assert_eq!(DVec3::new(1.0, 2.0, 3.0) - DVec3::new(6.0, 5.0, 4.0), DVec3::new(-5.0, -3.0, -1.0));

    assert_eq!(DVec3::new(1.0, 2.0, 3.0) * DVec3::new(4.0, 5.0, 6.0), DVec3::new(4.0, 10.0, 18.0));
    assert_eq!(DVec3::new(1.0, 2.0, 3.0) * 2.0, DVec3::new(2.0, 4.0, 6.0));
    assert_eq!(4.0 * DVec3::new(1.0, 2.0, 3.0), DVec3::new(4.0, 8.0, 12.0));

    assert_eq!(DVec3::new(1.0, 2.0, 3.0) / 2.0, DVec3::new(0.5, 1.0, 1.5));
    assert_eq!(DVec3::new(1.0, 2.0, 3.0) / DVec3::new(2.0, 8.0, 24.0), DVec3::new(0.5, 0.25, 0.125));

    let mut v = DVec3::new(1.0, 2.0, 3.0);
    v *= DVec3::new(4.0, 5.0, 6.0);
    assert_eq!(v, DVec3::new(4.0, 10.0, 18.0));
    v *= 2.0;
    assert_eq!(v, DVec3::new(8.0, 20.0, 36.0));
    v /= 2.0;
    assert_eq!(v, DVec3::new(4.0, 10.0, 18.0));
    v += DVec3::new(1.0, 2.0, 3.0);
    assert_eq!(v, DVec3::new(5.0, 12.0, 21.0));
    v -= DVec3::new(1.0, 2.0, 3.0);
    assert_eq!(v, DVec3::new(4.0, 10.0, 18.0));
    v += Vec3::new(1.0, 2.0, 3.0);
    assert_eq!(v, DVec3::new(5.0, 12.0, 21.0));
    v -= Vec3::new(1.0, 2.0, 3.0);
    assert_eq!(v, DVec3::new(4.0, 10.0, 18.0));

    assert_eq!(DVec3::new(2.0, 4.0, 8.0).reciprocal(), DVec3::new(0.5, 0.25, 0.125));
}

#[test]
fn test_d_vec3_abs() {
    assert_eq!(DVec3::new(1.0, -2.0, 3.0).abs(), DVec3::new(1.0, 2.0, 3.0));
    assert_eq!(DVec3::new(-1.0, 2.0, -3.0).abs(), DVec3::new(1.0, 2.0, 3.0));
}

#[test]
fn test_d_vec3_dot() {
    assert_eq!(DVec3::new(2.0, 3.0, 4.0).dot(DVec3::new(5.0, 6.0, 7.0)), (2 * 5 + 3 * 6 + 4 * 7) as f64);
}

#[test]
fn test_d_vec3_length() {
    assert_eq!(DVec3::new(2.0, 3.0, 4.0).length_sq(), (4 + 9 + 16) as f64);
    assert_eq!(DVec3::new(2.0, 3.0, 4.0).length(), sqrt((4 + 9 + 16) as f64));
}

#[test]
fn test_d_vec3_sqrt() {
    assert_eq!(DVec3::new(13.0, 15.0, 17.0).sqrt(), DVec3::new(sqrt(13.0), sqrt(15.0), sqrt(17.0)));
}

#[test]
fn test_d_vec3_equals() {
    assert!(!(DVec3::new(13.0, 15.0, 17.0) == DVec3::new(13.0, 15.0, 19.0)));
    assert!(DVec3::new(13.0, 15.0, 17.0) == DVec3::new(13.0, 15.0, 17.0));
    assert!(DVec3::new(13.0, 15.0, 17.0) != DVec3::new(13.0, 15.0, 19.0));
}

#[test]
fn test_d_vec3_load_store_double3_unsafe() {
    let d3 = Double3::new(1.0, 2.0, 3.0);
    let v = DVec3::load_double3_unsafe(&d3);
    let v2 = DVec3::new(1.0, 2.0, 3.0);
    assert_eq!(v, v2);

    let d3_out = DVec3::new(1.0, 2.0, 3.0).store_double3();
    assert_eq!(d3, d3_out);
}

#[test]
fn test_d_vec3_store_double3_edge_cases() {
    // Negative, zero, and large values to stress SIMD lane handling
    {
        let v = DVec3::new(-1.23456789, 0.0, 1.23456789e10);
        let out = v.store_double3();
        assert_eq!(out.x, v.x());
        assert_eq!(out.y, v.y());
        assert_eq!(out.z, v.z());
    }

    // Negative zero can expose SIMD mask bugs
    {
        let v = DVec3::new(-0.0, -0.0, -0.0);
        let out = v.store_double3();
        assert_eq!(out.x.to_bits(), v.x().to_bits());
        assert_eq!(out.y.to_bits(), v.y().to_bits());
        assert_eq!(out.z.to_bits(), v.z().to_bits());
    }
}

#[test]
fn test_d_vec3_cross() {
    assert_eq!(DVec3::new(1.0, 0.0, 0.0).cross(DVec3::new(0.0, 1.0, 0.0)), DVec3::new(0.0, 0.0, 1.0));
    assert_eq!(DVec3::new(0.0, 1.0, 0.0).cross(DVec3::new(1.0, 0.0, 0.0)), DVec3::new(0.0, 0.0, -1.0));
    assert_eq!(DVec3::new(0.0, 1.0, 0.0).cross(DVec3::new(0.0, 0.0, 1.0)), DVec3::new(1.0, 0.0, 0.0));
    assert_eq!(DVec3::new(0.0, 0.0, 1.0).cross(DVec3::new(0.0, 1.0, 0.0)), DVec3::new(-1.0, 0.0, 0.0));
    assert_eq!(DVec3::new(0.0, 0.0, 1.0).cross(DVec3::new(1.0, 0.0, 0.0)), DVec3::new(0.0, 1.0, 0.0));
    assert_eq!(DVec3::new(1.0, 0.0, 0.0).cross(DVec3::new(0.0, 0.0, 1.0)), DVec3::new(0.0, -1.0, 0.0));
}

#[test]
fn test_d_vec3_normalize() {
    assert_eq!(DVec3::new(3.0, 2.0, 1.0).normalized(), DVec3::new(3.0, 2.0, 1.0) / sqrt(9.0f64 + 4.0 + 1.0));
}

#[test]
fn test_d_vec3_sign() {
    assert_eq!(DVec3::new(1.2345, -6.7891, 0.0).sign(), DVec3::new(1.0, -1.0, 1.0));
    assert_eq!(DVec3::new(0.0, 2.3456, -7.8912).sign(), DVec3::new(1.0, 1.0, -1.0));
    assert_eq!(DVec3::new(-0.0, 0.0, -0.0).sign(), DVec3::new(-1.0, 1.0, -1.0));
    assert_eq!(DVec3::new(1.0, -1.0, 1.0).sign(), DVec3::new(1.0, -1.0, 1.0));
    let true_min = f64::from_bits(1); // DBL_TRUE_MIN
    assert_eq!(DVec3::new(true_min, -true_min, true_min).sign(), DVec3::new(1.0, -1.0, 1.0)); // Denormal number
    assert_eq!(DVec3::new(f64::INFINITY, -f64::INFINITY, f64::INFINITY).sign(), DVec3::new(1.0, -1.0, 1.0));
}

#[test]
fn test_d_vec3_convert_to_string() {
    let v = DVec3::new(1.0, 2.0, 3.0);
    assert_eq!(v.to_string(), "1, 2, 3");
}
