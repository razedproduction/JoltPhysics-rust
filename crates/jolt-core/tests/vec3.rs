//! Port of UnitTests/Math/Vec3Tests.cpp
//!
//! The `JPH_CROSS_PLATFORM_DETERMINISTIC` sections are always ported (the port behaves like that build), the
//! `JPH_USE_FMADD` branch is not (the port never uses FMA). `TestVec3SyncW` is only compiled with floating point
//! exceptions in C++, but the port always keeps W = Z, so it always runs.

// The expected values are written like in the C++ test (e.g. `1 * 4 + 7`)
#![allow(clippy::identity_op)]

use jolt_core::math::*;
use jolt_test_utils::{UnitTestRandom, check_approx_equal};

#[test]
fn test_vec3_construct_components() {
    let mut v = Vec3::new(1.0, 2.0, 3.0);

    // Test component access
    assert_eq!(v.x(), 1.0);
    assert_eq!(v.y(), 2.0);
    assert_eq!(v.z(), 3.0);

    // Test component access by [] operators
    assert_eq!(v[0], 1.0);
    assert_eq!(v[1], 2.0);
    assert_eq!(v[2], 3.0);

    // Test == and != operators
    assert!(v == Vec3::new(1.0, 2.0, 3.0));
    assert!(v != Vec3::new(1.0, 2.0, 4.0));

    // Set the components
    v.set_component(0, 4.0);
    v.set_component(1, 5.0);
    v.set_component(2, 6.0);
    assert_eq!(v, Vec3::new(4.0, 5.0, 6.0));

    // Set the components
    v.set_x(7.0);
    v.set_y(8.0);
    v.set_z(9.0);
    assert_eq!(v, Vec3::new(7.0, 8.0, 9.0));

    // Set all components
    v.set(10.0, 11.0, 12.0);
    assert_eq!(v, Vec3::new(10.0, 11.0, 12.0));
}

#[test]
fn test_vec3_load_store_float3() {
    let f3 = Float3::new(1.0, 2.0, 3.0);
    assert_eq!(Vec3::from_float3(&f3), Vec3::new(1.0, 2.0, 3.0));
    assert_eq!(Vec3::load_float3_unsafe(&f3), Vec3::new(1.0, 2.0, 3.0));

    let f3_out = Vec3::new(1.0, 2.0, 3.0).store_float3();
    assert_eq!(f3, f3_out);
}

#[test]
fn test_vec3_construct_vec4() {
    let v4 = Vec4::new(1.0, 2.0, 3.0, 4.0);
    assert_eq!(Vec3::from_vec4(v4), Vec3::new(1.0, 2.0, 3.0));
}

#[test]
fn test_vec3_zero() {
    let v = Vec3::zero();

    assert_eq!(v.x(), 0.0);
    assert_eq!(v.y(), 0.0);
    assert_eq!(v.z(), 0.0);
}

#[test]
fn test_vec3_na_n() {
    let mut v = Vec3::nan();

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
fn test_vec3_replicate() {
    assert_eq!(Vec3::replicate(2.0), Vec3::new(2.0, 2.0, 2.0));
}

#[test]
fn test_vec3_min_max() {
    let v1 = Vec3::new(1.0, 5.0, 3.0);
    let v2 = Vec3::new(4.0, 2.0, 6.0);
    let v3 = Vec3::new(6.0, 4.0, 2.0);

    assert_eq!(Vec3::min(v1, v2), Vec3::new(1.0, 2.0, 3.0));
    assert_eq!(Vec3::max(v1, v2), Vec3::new(4.0, 5.0, 6.0));

    assert_eq!(v1.reduce_min(), 1.0);
    assert_eq!(v1.reduce_max(), 5.0);
    assert_eq!(v2.reduce_min(), 2.0);
    assert_eq!(v2.reduce_max(), 6.0);

    assert_eq!(v1.lowest_component_index(), 0);
    assert_eq!(v1.highest_component_index(), 1);
    assert_eq!(v2.lowest_component_index(), 1);
    assert_eq!(v2.highest_component_index(), 2);
    assert_eq!(v3.lowest_component_index(), 2);
    assert_eq!(v3.highest_component_index(), 0);
}

#[test]
fn test_vec3_clamp() {
    let v1 = Vec3::new(1.0, 2.0, 3.0);
    let v2 = Vec3::new(4.0, 5.0, 6.0);
    let v = Vec3::new(-1.0, 3.0, 7.0);

    assert_eq!(Vec3::clamp(v, v1, v2), Vec3::new(1.0, 3.0, 6.0));
}

#[test]
fn test_vec3_comparisons() {
    assert_eq!(
        Vec3::equals(Vec3::new(1.0, 2.0, 3.0), Vec3::new(1.0, 4.0, 3.0)),
        UVec4::new(0xffffffff, 0, 0xffffffff, 0xffffffff)
    ); // W is always Z for comparisons
    assert_eq!(Vec3::less(Vec3::new(1.0, 2.0, 4.0), Vec3::new(1.0, 4.0, 3.0)), UVec4::new(0, 0xffffffff, 0, 0));
    assert_eq!(
        Vec3::less_or_equal(Vec3::new(1.0, 2.0, 4.0), Vec3::new(1.0, 4.0, 3.0)),
        UVec4::new(0xffffffff, 0xffffffff, 0, 0)
    );
    assert_eq!(
        Vec3::greater(Vec3::new(1.0, 2.0, 4.0), Vec3::new(1.0, 4.0, 3.0)),
        UVec4::new(0, 0, 0xffffffff, 0xffffffff)
    );
    assert_eq!(
        Vec3::greater_or_equal(Vec3::new(1.0, 2.0, 4.0), Vec3::new(1.0, 4.0, 3.0)),
        UVec4::new(0xffffffff, 0, 0xffffffff, 0xffffffff)
    );
}

#[test]
fn test_vec3_fma() {
    assert_eq!(
        Vec3::fused_multiply_add(Vec3::new(1.0, 2.0, 3.0), Vec3::new(4.0, 5.0, 6.0), Vec3::new(7.0, 8.0, 9.0)),
        Vec3::new((1 * 4 + 7) as f32, (2 * 5 + 8) as f32, (3 * 6 + 9) as f32)
    );
}

#[test]
fn test_vec3_select() {
    let a = Vec3::new(1.0, 2.0, 3.0);
    let b = Vec3::new(4.0, 5.0, 6.0);
    assert_eq!(Vec3::select(a, b, UVec4::new(0x80000000, 0, 0x80000000, 0)), Vec3::new(4.0, 2.0, 6.0));
    assert_eq!(Vec3::select(a, b, UVec4::new(0, 0x80000000, 0, 0x80000000)), Vec3::new(1.0, 5.0, 3.0));
    assert_eq!(
        Vec3::select(a, b, UVec4::new(0xffffffff, 0x7fffffff, 0xffffffff, 0x7fffffff)),
        Vec3::new(4.0, 2.0, 6.0)
    );
    assert_eq!(
        Vec3::select(a, b, UVec4::new(0x7fffffff, 0xffffffff, 0x7fffffff, 0xffffffff)),
        Vec3::new(1.0, 5.0, 3.0)
    );
}

#[test]
fn test_vec3_bit_ops() {
    // Test all bit permutations
    let v1 = Vec3::from_vec4(UVec4::new(0b0011, 0b00110, 0b001100, 0).reinterpret_as_float());
    let v2 = Vec3::from_vec4(UVec4::new(0b0101, 0b01010, 0b010100, 0).reinterpret_as_float());

    assert_eq!(Vec3::or(v1, v2), Vec3::from_vec4(UVec4::new(0b0111, 0b01110, 0b011100, 0).reinterpret_as_float()));
    assert_eq!(Vec3::xor(v1, v2), Vec3::from_vec4(UVec4::new(0b0110, 0b01100, 0b011000, 0).reinterpret_as_float()));
    assert_eq!(Vec3::and(v1, v2), Vec3::from_vec4(UVec4::new(0b0001, 0b00010, 0b000100, 0).reinterpret_as_float()));
}

#[test]
fn test_vec3_close() {
    assert!(Vec3::new(1.0, 2.0, 3.0).is_close(Vec3::new(1.001, 2.001, 3.001), 1.0e-4));
    assert!(!Vec3::new(1.0, 2.0, 3.0).is_close(Vec3::new(1.001, 2.001, 3.001), 1.0e-6));

    assert!(Vec3::new(1.001, 0.0, 0.0).is_normalized(1.0e-2));
    assert!(!Vec3::new(0.0, 1.001, 0.0).is_normalized(1.0e-4));

    assert!(Vec3::new(-1.0e-7, 1.0e-7, 1.0e-8).is_near_zero_default());
    assert!(!Vec3::new(-1.0e-7, 1.0e-7, -1.0e-5).is_near_zero_default());
}

#[test]
fn test_vec3_operators() {
    assert_eq!(-Vec3::new(1.0, 2.0, 3.0), Vec3::new(-1.0, -2.0, -3.0));

    let neg_zero = -Vec3::zero();
    assert_eq!(neg_zero, Vec3::zero());

    // When cross platform deterministic, we want to make sure that -0 is represented as 0
    let neg_zero_bin = neg_zero.reinterpret_as_int();
    assert_eq!(neg_zero_bin.x(), 0);
    assert_eq!(neg_zero_bin.y(), 0);
    assert_eq!(neg_zero_bin.z(), 0);

    assert_eq!(Vec3::new(1.0, 2.0, 3.0) + Vec3::new(4.0, 5.0, 6.0), Vec3::new(5.0, 7.0, 9.0));
    assert_eq!(Vec3::new(1.0, 2.0, 3.0) - Vec3::new(6.0, 5.0, 4.0), Vec3::new(-5.0, -3.0, -1.0));

    assert_eq!(Vec3::new(1.0, 2.0, 3.0) * Vec3::new(4.0, 5.0, 6.0), Vec3::new(4.0, 10.0, 18.0));
    assert_eq!(Vec3::new(1.0, 2.0, 3.0) * 2.0, Vec3::new(2.0, 4.0, 6.0));
    assert_eq!(4.0 * Vec3::new(1.0, 2.0, 3.0), Vec3::new(4.0, 8.0, 12.0));

    assert_eq!(Vec3::new(1.0, 2.0, 3.0) / 2.0, Vec3::new(0.5, 1.0, 1.5));
    assert_eq!(Vec3::new(1.0, 2.0, 3.0) / Vec3::new(2.0, 8.0, 24.0), Vec3::new(0.5, 0.25, 0.125));

    let mut v = Vec3::new(1.0, 2.0, 3.0);
    v *= Vec3::new(4.0, 5.0, 6.0);
    assert_eq!(v, Vec3::new(4.0, 10.0, 18.0));
    v *= 2.0;
    assert_eq!(v, Vec3::new(8.0, 20.0, 36.0));
    v /= 2.0;
    assert_eq!(v, Vec3::new(4.0, 10.0, 18.0));
    v += Vec3::new(1.0, 2.0, 3.0);
    assert_eq!(v, Vec3::new(5.0, 12.0, 21.0));
    v -= Vec3::new(1.0, 2.0, 3.0);
    assert_eq!(v, Vec3::new(4.0, 10.0, 18.0));

    assert_eq!(Vec3::new(2.0, 4.0, 8.0).reciprocal(), Vec3::new(0.5, 0.25, 0.125));
}

#[test]
fn test_vec3_swizzle() {
    let v = Vec3::new(1.0, 2.0, 3.0);

    assert_eq!(v.splat_x(), Vec4::replicate(1.0));
    assert_eq!(v.splat_y(), Vec4::replicate(2.0));
    assert_eq!(v.splat_z(), Vec4::replicate(3.0));

    assert_eq!(v.swizzle::<SWIZZLE_X, SWIZZLE_X, SWIZZLE_X>(), Vec3::new(1.0, 1.0, 1.0));
    assert_eq!(v.swizzle::<SWIZZLE_X, SWIZZLE_X, SWIZZLE_Y>(), Vec3::new(1.0, 1.0, 2.0));
    assert_eq!(v.swizzle::<SWIZZLE_X, SWIZZLE_X, SWIZZLE_Z>(), Vec3::new(1.0, 1.0, 3.0));
    assert_eq!(v.swizzle::<SWIZZLE_X, SWIZZLE_Y, SWIZZLE_X>(), Vec3::new(1.0, 2.0, 1.0));
    assert_eq!(v.swizzle::<SWIZZLE_X, SWIZZLE_Y, SWIZZLE_Y>(), Vec3::new(1.0, 2.0, 2.0));
    assert_eq!(v.swizzle::<SWIZZLE_X, SWIZZLE_Y, SWIZZLE_Z>(), Vec3::new(1.0, 2.0, 3.0));
    assert_eq!(v.swizzle::<SWIZZLE_X, SWIZZLE_Z, SWIZZLE_X>(), Vec3::new(1.0, 3.0, 1.0));
    assert_eq!(v.swizzle::<SWIZZLE_X, SWIZZLE_Z, SWIZZLE_Y>(), Vec3::new(1.0, 3.0, 2.0));
    assert_eq!(v.swizzle::<SWIZZLE_X, SWIZZLE_Z, SWIZZLE_Z>(), Vec3::new(1.0, 3.0, 3.0));

    assert_eq!(v.swizzle::<SWIZZLE_Y, SWIZZLE_X, SWIZZLE_X>(), Vec3::new(2.0, 1.0, 1.0));
    assert_eq!(v.swizzle::<SWIZZLE_Y, SWIZZLE_X, SWIZZLE_Y>(), Vec3::new(2.0, 1.0, 2.0));
    assert_eq!(v.swizzle::<SWIZZLE_Y, SWIZZLE_X, SWIZZLE_Z>(), Vec3::new(2.0, 1.0, 3.0));
    assert_eq!(v.swizzle::<SWIZZLE_Y, SWIZZLE_Y, SWIZZLE_X>(), Vec3::new(2.0, 2.0, 1.0));
    assert_eq!(v.swizzle::<SWIZZLE_Y, SWIZZLE_Y, SWIZZLE_Y>(), Vec3::new(2.0, 2.0, 2.0));
    assert_eq!(v.swizzle::<SWIZZLE_Y, SWIZZLE_Y, SWIZZLE_Z>(), Vec3::new(2.0, 2.0, 3.0));
    assert_eq!(v.swizzle::<SWIZZLE_Y, SWIZZLE_Z, SWIZZLE_X>(), Vec3::new(2.0, 3.0, 1.0));
    assert_eq!(v.swizzle::<SWIZZLE_Y, SWIZZLE_Z, SWIZZLE_Y>(), Vec3::new(2.0, 3.0, 2.0));
    assert_eq!(v.swizzle::<SWIZZLE_Y, SWIZZLE_Z, SWIZZLE_Z>(), Vec3::new(2.0, 3.0, 3.0));

    assert_eq!(v.swizzle::<SWIZZLE_Z, SWIZZLE_X, SWIZZLE_X>(), Vec3::new(3.0, 1.0, 1.0));
    assert_eq!(v.swizzle::<SWIZZLE_Z, SWIZZLE_X, SWIZZLE_Y>(), Vec3::new(3.0, 1.0, 2.0));
    assert_eq!(v.swizzle::<SWIZZLE_Z, SWIZZLE_X, SWIZZLE_Z>(), Vec3::new(3.0, 1.0, 3.0));
    assert_eq!(v.swizzle::<SWIZZLE_Z, SWIZZLE_Y, SWIZZLE_X>(), Vec3::new(3.0, 2.0, 1.0));
    assert_eq!(v.swizzle::<SWIZZLE_Z, SWIZZLE_Y, SWIZZLE_Y>(), Vec3::new(3.0, 2.0, 2.0));
    assert_eq!(v.swizzle::<SWIZZLE_Z, SWIZZLE_Y, SWIZZLE_Z>(), Vec3::new(3.0, 2.0, 3.0));
    assert_eq!(v.swizzle::<SWIZZLE_Z, SWIZZLE_Z, SWIZZLE_X>(), Vec3::new(3.0, 3.0, 1.0));
    assert_eq!(v.swizzle::<SWIZZLE_Z, SWIZZLE_Z, SWIZZLE_Y>(), Vec3::new(3.0, 3.0, 2.0));
    assert_eq!(v.swizzle::<SWIZZLE_Z, SWIZZLE_Z, SWIZZLE_Z>(), Vec3::new(3.0, 3.0, 3.0));
}

#[test]
fn test_vec3_abs() {
    assert_eq!(Vec3::new(1.0, -2.0, 3.0).abs(), Vec3::new(1.0, 2.0, 3.0));
    assert_eq!(Vec3::new(-1.0, 2.0, -3.0).abs(), Vec3::new(1.0, 2.0, 3.0));
}

/// All combinations of +0 and -0 for 3 components
fn signed_zeros3() -> impl Iterator<Item = Vec3> {
    (0..8).map(|i: u32| {
        let z = |bit: u32| if i & (4 >> bit) != 0 { 0.0 } else { -0.0 };
        Vec3::new(z(0), z(1), z(2))
    })
}

#[test]
fn test_vec3_reduce_sum() {
    assert_eq!(Vec3::new(1.0, 10.0, 100.0).reduce_sum(), 111.0);

    // Test handling of -0.0f
    for v in signed_zeros3() {
        assert_eq!(v.reduce_sum().to_bits(), 0, "{v:?}");
    }
}

#[test]
fn test_vec3_dot() {
    assert_eq!(Vec3::new(1.0, 2.0, 3.0).dot(Vec3::new(4.0, 5.0, 6.0)), (1 * 4 + 2 * 5 + 3 * 6) as f32);
    assert_eq!(
        Vec3::new(1.0, 2.0, 3.0).dot_v(Vec3::new(4.0, 5.0, 6.0)),
        Vec3::replicate((1 * 4 + 2 * 5 + 3 * 6) as f32)
    );
    assert_eq!(
        Vec3::new(1.0, 2.0, 3.0).dot_v4(Vec3::new(4.0, 5.0, 6.0)),
        Vec4::replicate((1 * 4 + 2 * 5 + 3 * 6) as f32)
    );

    // Test handling of -0.0f
    for v in signed_zeros3() {
        assert_eq!(v.dot(Vec3::replicate(1.0)).to_bits(), 0, "{v:?}");
    }
}

#[test]
fn test_vec3_length() {
    assert_eq!(Vec3::new(1.0, 2.0, 3.0).length_sq(), (1 + 4 + 9) as f32);
    assert_eq!(Vec3::new(1.0, 2.0, 3.0).length(), sqrt((1 + 4 + 9) as f32));
}

#[test]
fn test_vec3_sqrt() {
    check_approx_equal(Vec3::new(13.0, 15.0, 17.0).sqrt(), Vec3::new(sqrt(13.0), sqrt(15.0), sqrt(17.0)));
}

#[test]
fn test_vec3_cross() {
    assert_eq!(Vec3::new(1.0, 0.0, 0.0).cross(Vec3::new(0.0, 1.0, 0.0)), Vec3::new(0.0, 0.0, 1.0));
    assert_eq!(Vec3::new(0.0, 1.0, 0.0).cross(Vec3::new(1.0, 0.0, 0.0)), Vec3::new(0.0, 0.0, -1.0));
    assert_eq!(Vec3::new(0.0, 1.0, 0.0).cross(Vec3::new(0.0, 0.0, 1.0)), Vec3::new(1.0, 0.0, 0.0));
    assert_eq!(Vec3::new(0.0, 0.0, 1.0).cross(Vec3::new(0.0, 1.0, 0.0)), Vec3::new(-1.0, 0.0, 0.0));
    assert_eq!(Vec3::new(0.0, 0.0, 1.0).cross(Vec3::new(1.0, 0.0, 0.0)), Vec3::new(0.0, 1.0, 0.0));
    assert_eq!(Vec3::new(1.0, 0.0, 0.0).cross(Vec3::new(0.0, 0.0, 1.0)), Vec3::new(0.0, -1.0, 0.0));
}

#[test]
fn test_vec3_normalize() {
    assert_eq!(Vec3::new(3.0, 2.0, 1.0).normalized(), Vec3::new(3.0, 2.0, 1.0) / sqrt(9.0f32 + 4.0 + 1.0));
    assert_eq!(
        Vec3::new(3.0, 2.0, 1.0).normalized_or(Vec3::new(1.0, 2.0, 3.0)),
        Vec3::new(3.0, 2.0, 1.0) / sqrt(9.0f32 + 4.0 + 1.0)
    );
    assert_eq!(Vec3::zero().normalized_or(Vec3::new(1.0, 2.0, 3.0)), Vec3::new(1.0, 2.0, 3.0));
    assert_eq!(
        Vec3::new(0.999 * sqrt(f32::MIN_POSITIVE), 0.0, 0.0).normalized_or(Vec3::new(1.0, 2.0, 3.0)),
        Vec3::new(1.0, 2.0, 3.0)
    ); // A vector that has a squared length that is denormal should also be treated as zero
    check_approx_equal(
        Vec3::new(1.001 * sqrt(f32::MIN_POSITIVE), 0.0, 0.0).normalized_or(Vec3::new(1.0, 2.0, 3.0)),
        Vec3::new(1.0, 0.0, 0.0),
    ); // A value that is just above being denormal should work normally
}

#[test]
fn test_vec3_cast() {
    assert!(UVec4::equals(Vec3::new(1.0, 2.0, 3.0).to_int(), UVec4::new(1, 2, 3, 0)).test_all_xyz_true());
    assert!(
        UVec4::equals(Vec3::new(1.0, 2.0, 3.0).reinterpret_as_int(), UVec4::new(0x3f800000, 0x40000000, 0x40400000, 0))
            .test_all_xyz_true()
    );
}

#[test]
fn test_vec3_normalized_perpendicular() {
    let mut random = UnitTestRandom::default();
    for _ in 0..100 {
        let mut v = Vec3::random(&mut random);
        assert!(v.is_normalized_default());
        v *= random.uniform_f32(1.0, 10.0);

        let p = v.normalized_perpendicular();

        assert!(p.is_normalized_default());
        assert!(v.dot(p).abs() < 1.0e-6);
    }

    // Check treatment of -0.0f
    let check = |v: Vec3, expected: Vec3| {
        assert!(
            UVec4::equals(v.normalized_perpendicular().reinterpret_as_int(), expected.reinterpret_as_int())
                .test_all_xyz_true(),
            "{v:?}"
        );
    };
    check(Vec3::new(0.0, 0.0, 2.0), Vec3::new(0.0, 1.0, 0.0));
    check(Vec3::new(0.0, -0.0, 2.0), Vec3::new(0.0, 1.0, 0.0));
    check(Vec3::new(-0.0, 0.0, 2.0), Vec3::new(0.0, 1.0, 0.0));
    check(Vec3::new(-0.0, -0.0, 2.0), Vec3::new(0.0, 1.0, 0.0));
    check(Vec3::new(0.0, 2.0, 0.0), Vec3::new(0.0, 0.0, -1.0));
    check(Vec3::new(0.0, 2.0, -0.0), Vec3::new(0.0, -0.0, -1.0));
    check(Vec3::new(-0.0, 2.0, 0.0), Vec3::new(0.0, 0.0, -1.0));
    check(Vec3::new(-0.0, 2.0, -0.0), Vec3::new(0.0, -0.0, -1.0));
    check(Vec3::new(2.0, 0.0, 0.0), Vec3::new(0.0, 0.0, -1.0));
    check(Vec3::new(2.0, 0.0, -0.0), Vec3::new(-0.0, 0.0, -1.0));
    check(Vec3::new(2.0, -0.0, 0.0), Vec3::new(0.0, 0.0, -1.0));
    check(Vec3::new(2.0, -0.0, -0.0), Vec3::new(-0.0, 0.0, -1.0));
}

#[test]
fn test_vec3_sign() {
    assert_eq!(Vec3::new(1.2345, -6.7891, 0.0).sign(), Vec3::new(1.0, -1.0, 1.0));
    assert_eq!(Vec3::new(0.0, 2.3456, -7.8912).sign(), Vec3::new(1.0, 1.0, -1.0));
    assert_eq!(Vec3::new(-0.0, 0.0, -0.0).sign(), Vec3::new(-1.0, 1.0, -1.0));
    assert_eq!(Vec3::new(1.0, -1.0, 1.0).sign(), Vec3::new(1.0, -1.0, 1.0));
    let true_min = f32::from_bits(1); // FLT_TRUE_MIN
    assert_eq!(Vec3::new(true_min, -true_min, true_min).sign(), Vec3::new(1.0, -1.0, 1.0)); // Denormal number
    assert_eq!(Vec3::new(f32::INFINITY, -f32::INFINITY, f32::INFINITY).sign(), Vec3::new(1.0, -1.0, 1.0));
}

#[test]
fn test_vec3_flip_sign() {
    let v = Vec3::new(1.0, 2.0, 3.0);
    assert_eq!(v.flip_sign::<-1, 1, 1>(), Vec3::new(-1.0, 2.0, 3.0));
    assert_eq!(v.flip_sign::<1, -1, 1>(), Vec3::new(1.0, -2.0, 3.0));
    assert_eq!(v.flip_sign::<1, 1, -1>(), Vec3::new(1.0, 2.0, -3.0));
}

#[test]
fn test_vec3_sync_w() {
    {
        // Check that W equals Z
        let v = Vec3::new(1.0, 2.0, 3.0);
        assert_eq!(Vec4::from_vec3(v), Vec4::new(1.0, 2.0, 3.0, 3.0));
    }

    {
        // Check that setting individual components syncs W and Z
        let mut v = Vec3::default();
        v.set_component(2, 3.0);
        v.set_component(1, 2.0);
        v.set_component(0, 1.0);
        assert_eq!(v, Vec3::new(1.0, 2.0, 3.0));
        assert_eq!(Vec4::from_vec3(v), Vec4::new(1.0, 2.0, 3.0, 3.0));
    }

    {
        // Check that W and Z are still synced after a simple addition
        assert_eq!(Vec4::from_vec3(Vec3::new(1.0, 2.0, 3.0) + Vec3::new(4.0, 5.0, 6.0)), Vec4::new(5.0, 7.0, 9.0, 9.0));
    }

    {
        // Test that casting a Vec4 to Vec3 syncs W and Z
        assert_eq!(Vec4::from_vec3(Vec3::from_vec4(Vec4::new(1.0, 2.0, 3.0, 4.0))), Vec4::new(1.0, 2.0, 3.0, 3.0));
    }

    {
        // Test that loading from Float3 syncs W and Z
        assert_eq!(Vec4::from_vec3(Vec3::from_float3(&Float3::new(1.0, 2.0, 3.0))), Vec4::new(1.0, 2.0, 3.0, 3.0));
    }

    {
        // Test that loading unsafe from Float3 syncs W and Z
        assert_eq!(
            Vec4::from_vec3(Vec3::load_float3_unsafe(&Float3::new(1.0, 2.0, 3.0))),
            Vec4::new(1.0, 2.0, 3.0, 3.0)
        );
    }

    {
        // Test swizzle syncs W and Z
        assert_eq!(
            Vec4::from_vec3(Vec3::new(1.0, 2.0, 3.0).swizzle::<SWIZZLE_Z, SWIZZLE_Y, SWIZZLE_X>()),
            Vec4::new(3.0, 2.0, 1.0, 1.0)
        );
    }

    {
        // Test cross product syncs W and Z
        assert_eq!(
            Vec4::from_vec3(Vec3::new(1.0, 0.0, 0.0).cross(Vec3::new(0.0, 1.0, 0.0))),
            Vec4::new(0.0, 0.0, 1.0, 1.0)
        );
        assert_eq!(
            Vec4::from_vec3(Vec3::new(0.0, 1.0, 0.0).cross(Vec3::new(0.0, 0.0, 1.0))),
            Vec4::new(1.0, 0.0, 0.0, 0.0)
        );
    }
}

#[test]
fn test_vec3_convert_to_string() {
    let v = Vec3::new(1.0, 2.0, 3.0);
    assert_eq!(v.to_string(), "1, 2, 3");
}

#[test]
fn test_vec3_compress_unit_vector() {
    // We want these to be preserved exactly
    assert_eq!(Vec3::decompress_unit_vector(Vec3::axis_x().compress_unit_vector()), Vec3::axis_x());
    assert_eq!(Vec3::decompress_unit_vector(Vec3::axis_y().compress_unit_vector()), Vec3::axis_y());
    assert_eq!(Vec3::decompress_unit_vector(Vec3::axis_z().compress_unit_vector()), Vec3::axis_z());
    assert_eq!(Vec3::decompress_unit_vector((-Vec3::axis_x()).compress_unit_vector()), -Vec3::axis_x());
    assert_eq!(Vec3::decompress_unit_vector((-Vec3::axis_y()).compress_unit_vector()), -Vec3::axis_y());
    assert_eq!(Vec3::decompress_unit_vector((-Vec3::axis_z()).compress_unit_vector()), -Vec3::axis_z());

    let mut random = UnitTestRandom::default();
    for _ in 0..1000 {
        let v = Vec3::random(&mut random);
        let compressed = v.compress_unit_vector();
        let decompressed = Vec3::decompress_unit_vector(compressed);
        let diff = (decompressed - v).length();
        assert!(diff < 1.0e-4, "{v:?} {decompressed:?} {diff}");
    }
}

#[test]
fn test_difference_of_products() {
    let a = Vec3::new(33962.035, 33962.0351, 33962.0352);
    let b = Vec3::new(-30438.8, -30438.801, -30438.802);
    let c = Vec3::new(41563.4, 41563.401, 41563.402);
    let d = Vec3::new(-24871.969, -24871.970, -24871.971);
    let result = Vec3::difference_of_products(a, b, c, d);
    let expected = DVec3::from_vec3(a) * DVec3::from_vec3(b) - DVec3::from_vec3(c) * DVec3::from_vec3(d);
    assert_eq!(expected, DVec3::new(-75.165603637695312, 103.16904449462891, 36.836944580078125));
    // The products are in the order of 10^9, so the subtraction causes a large loss of precision and we get a very different result. This is expected when fused multiply add instructions are not available.
    assert_eq!(result, Vec3::new(-128.0, 64.0, 0.0));
}

#[test]
fn test_vec3_unit_sphere() {
    // Not in the C++ tests: the table is created like the C++ static initializer and holds only unit vectors
    let sphere = Vec3::unit_sphere();
    assert_eq!(sphere.len(), 1026);
    assert!(sphere.iter().all(|v| v.is_normalized_default()));
}
