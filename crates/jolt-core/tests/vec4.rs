//! Port of UnitTests/Math/Vec4Tests.cpp
//!
//! The `JPH_CROSS_PLATFORM_DETERMINISTIC` sections are always ported (the port behaves like that build), the
//! `JPH_USE_FMADD` branch is not (the port never uses FMA).

// The expected values are written like in the C++ test (e.g. `1 * 4 + 7`)
#![allow(clippy::identity_op)]

use jolt_core::math::*;
use jolt_test_utils::{UnitTestRandom, check_approx_equal};

#[test]
fn test_vec4_construct() {
    let v = Vec4::new(1.0, 2.0, 3.0, 4.0);

    // Test component access
    assert_eq!(v.x(), 1.0);
    assert_eq!(v.y(), 2.0);
    assert_eq!(v.z(), 3.0);
    assert_eq!(v.w(), 4.0);

    // Test component access by [] operators
    assert_eq!(v[0], 1.0);
    assert_eq!(v[1], 2.0);
    assert_eq!(v[2], 3.0);
    assert_eq!(v[3], 4.0);

    // Test == and != operators
    assert!(v == Vec4::new(1.0, 2.0, 3.0, 4.0));
    assert!(v != Vec4::new(1.0, 2.0, 4.0, 3.0));
}

#[test]
fn test_vec4_load_store_float4() {
    let f4 = Float4::new(1.0, 2.0, 3.0, 4.0);
    assert_eq!(Vec4::load_float4(&f4), Vec4::new(1.0, 2.0, 3.0, 4.0));
    assert_eq!(Vec4::load_float4_aligned(&f4), Vec4::new(1.0, 2.0, 3.0, 4.0));

    let f4_out = Vec4::new(1.0, 2.0, 3.0, 4.0).store_float4();
    assert_eq!(f4_out[0], 1.0);
    assert_eq!(f4_out[1], 2.0);
    assert_eq!(f4_out[2], 3.0);
    assert_eq!(f4_out[3], 4.0);

    let sf: [f32; 20] =
        [0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 2.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 3.0, 0.0, 4.0, 0.0];
    assert_eq!(
        Vec4::gather_float4::<{ 2 * size_of::<f32>() }>(&sf, UVec4::new(1, 3, 8, 9)),
        Vec4::new(1.0, 2.0, 3.0, 4.0)
    );
}

#[test]
fn test_vec4_construct_vec3() {
    let v3 = Vec3::new(1.0, 2.0, 3.0);
    assert_eq!(Vec4::from_vec3_w(v3, 4.0), Vec4::new(1.0, 2.0, 3.0, 4.0));
}

#[test]
fn test_vec4_zero() {
    let v = Vec4::zero();

    assert_eq!(v.x(), 0.0);
    assert_eq!(v.y(), 0.0);
    assert_eq!(v.z(), 0.0);
    assert_eq!(v.w(), 0.0);
}

#[test]
fn test_vec4_na_n() {
    let mut v = Vec4::nan();

    assert!(v.x().is_nan());
    assert!(v.y().is_nan());
    assert!(v.z().is_nan());
    assert!(v.w().is_nan());
    assert!(v.is_nan());

    v.set_x(0.0);
    assert!(v.is_nan());
    v.set_y(0.0);
    assert!(v.is_nan());
    v.set_z(0.0);
    assert!(v.is_nan());
    v.set_w(0.0);
    assert!(!v.is_nan());
}

#[test]
fn test_vec4_replicate() {
    assert_eq!(Vec4::replicate(2.0), Vec4::new(2.0, 2.0, 2.0, 2.0));
}

#[test]
fn test_vec4_min_max() {
    let v1 = Vec4::new(1.0, 6.0, 3.0, 8.0);
    let v2 = Vec4::new(5.0, 2.0, 7.0, 4.0);
    let v3 = Vec4::new(5.0, 7.0, 2.0, 4.0);
    let v4 = Vec4::new(7.0, 5.0, 4.0, 2.0);

    assert_eq!(Vec4::min(v1, v2), Vec4::new(1.0, 2.0, 3.0, 4.0));
    assert_eq!(Vec4::max(v1, v2), Vec4::new(5.0, 6.0, 7.0, 8.0));

    assert_eq!(v1.reduce_min(), 1.0);
    assert_eq!(v1.reduce_max(), 8.0);
    assert_eq!(v2.reduce_min(), 2.0);
    assert_eq!(v2.reduce_max(), 7.0);

    assert_eq!(v1.lowest_component_index(), 0);
    assert_eq!(v1.highest_component_index(), 3);
    assert_eq!(v2.lowest_component_index(), 1);
    assert_eq!(v2.highest_component_index(), 2);
    assert_eq!(v3.lowest_component_index(), 2);
    assert_eq!(v3.highest_component_index(), 1);
    assert_eq!(v4.lowest_component_index(), 3);
    assert_eq!(v4.highest_component_index(), 0);
}

/// All combinations of +0 and -0 for 4 components, in the order of the C++ test (first one is all -0)
fn signed_zeros4() -> impl Iterator<Item = Vec4> {
    (0..16).map(|i: u32| {
        let z = |bit: u32| if i & (8 >> bit) != 0 { 0.0 } else { -0.0 };
        Vec4::new(z(0), z(1), z(2), z(3))
    })
}

#[test]
fn test_vec4_reduce_sum() {
    let v1 = Vec4::new(1.0, 6.0, 3.0, 8.0);
    let v2 = Vec4::new(-5.0, 2.0, -7.0, 4.0);

    assert_eq!(v1.reduce_sum(), (1 + 6 + 3 + 8) as f32);
    assert_eq!(v2.reduce_sum(), (-5 + 2 - 7 + 4) as f32);

    // Test handling of -0.0f
    for (i, v) in signed_zeros4().enumerate() {
        assert_eq!(v.reduce_sum().to_bits(), if i == 0 { 0x80000000 } else { 0 }, "{v:?}");
    }
}

#[test]
fn test_vec4_clamp() {
    let v1 = Vec4::new(1.0, 2.0, 3.0, 4.0);
    let v2 = Vec4::new(5.0, 6.0, 7.0, 8.0);
    let v = Vec4::new(-1.0, 3.0, 9.0, -11.0);

    assert_eq!(Vec4::clamp(v, v1, v2), Vec4::new(1.0, 3.0, 7.0, 4.0));
}

#[test]
fn test_vec4_comparisons() {
    let a = Vec4::new(1.0, 2.0, 3.0, 4.0);
    let b = Vec4::new(2.0, 1.0, 3.0, 4.0);
    assert_eq!(Vec4::equals(a, b), UVec4::new(0, 0, 0xffffffff, 0xffffffff));
    assert_eq!(Vec4::less(a, b), UVec4::new(0xffffffff, 0, 0, 0));
    assert_eq!(Vec4::less_or_equal(a, b), UVec4::new(0xffffffff, 0, 0xffffffff, 0xffffffff));
    assert_eq!(Vec4::greater(a, b), UVec4::new(0, 0xffffffff, 0, 0));
    assert_eq!(Vec4::greater_or_equal(a, b), UVec4::new(0, 0xffffffff, 0xffffffff, 0xffffffff));
}

#[test]
fn test_vec4_fma() {
    assert_eq!(
        Vec4::fused_multiply_add(
            Vec4::new(1.0, 2.0, 3.0, 4.0),
            Vec4::new(5.0, 6.0, 7.0, 8.0),
            Vec4::new(9.0, 10.0, 11.0, 12.0)
        ),
        Vec4::new((1 * 5 + 9) as f32, (2 * 6 + 10) as f32, (3 * 7 + 11) as f32, (4 * 8 + 12) as f32)
    );
}

#[test]
fn test_vec4_select() {
    let a = Vec4::new(1.0, 2.0, 3.0, 4.0);
    let b = Vec4::new(5.0, 6.0, 7.0, 8.0);
    assert_eq!(Vec4::select(a, b, UVec4::new(0x80000000, 0, 0x80000000, 0)), Vec4::new(5.0, 2.0, 7.0, 4.0));
    assert_eq!(Vec4::select(a, b, UVec4::new(0, 0x80000000, 0, 0x80000000)), Vec4::new(1.0, 6.0, 3.0, 8.0));
    assert_eq!(
        Vec4::select(a, b, UVec4::new(0xffffffff, 0x7fffffff, 0xffffffff, 0x7fffffff)),
        Vec4::new(5.0, 2.0, 7.0, 4.0)
    );
    assert_eq!(
        Vec4::select(a, b, UVec4::new(0x7fffffff, 0xffffffff, 0x7fffffff, 0xffffffff)),
        Vec4::new(1.0, 6.0, 3.0, 8.0)
    );
}

#[test]
fn test_vec4_bit_ops() {
    // Test all bit permutations
    let v1 = UVec4::new(0b0011, 0b00110, 0b001100, 0b0011000).reinterpret_as_float();
    let v2 = UVec4::new(0b0101, 0b01010, 0b010100, 0b0101000).reinterpret_as_float();

    assert_eq!(Vec4::or(v1, v2), UVec4::new(0b0111, 0b01110, 0b011100, 0b0111000).reinterpret_as_float());
    assert_eq!(Vec4::xor(v1, v2), UVec4::new(0b0110, 0b01100, 0b011000, 0b0110000).reinterpret_as_float());
    assert_eq!(Vec4::and(v1, v2), UVec4::new(0b0001, 0b00010, 0b000100, 0b0001000).reinterpret_as_float());
}

#[test]
fn test_vec4_close() {
    assert!(Vec4::new(1.0, 2.0, 3.0, 4.0).is_close(Vec4::new(1.001, 2.001, 3.001, 4.001), 1.0e-4));
    assert!(!Vec4::new(1.0, 2.0, 3.0, 4.0).is_close(Vec4::new(1.001, 2.001, 3.001, 4.001), 1.0e-6));

    assert!(Vec4::new(1.001, 0.0, 0.0, 0.0).is_normalized(1.0e-2));
    assert!(!Vec4::new(0.0, 1.001, 0.0, 0.0).is_normalized(1.0e-4));

    assert!(Vec4::new(-1.0e-7, 1.0e-7, 1.0e-8, -1.0e-8).is_near_zero_default());
    assert!(!Vec4::new(-1.0e-7, 1.0e-7, -1.0e-5, 1.0e-5).is_near_zero_default());
}

#[test]
fn test_vec4_operators() {
    assert_eq!(-Vec4::new(1.0, 2.0, 3.0, 4.0), Vec4::new(-1.0, -2.0, -3.0, -4.0));

    let neg_zero = -Vec4::zero();
    assert_eq!(neg_zero, Vec4::zero());

    // When cross platform deterministic, we want to make sure that -0 is represented as 0
    let neg_zero_bin = neg_zero.reinterpret_as_int();
    assert_eq!(neg_zero_bin.x(), 0);
    assert_eq!(neg_zero_bin.y(), 0);
    assert_eq!(neg_zero_bin.z(), 0);
    assert_eq!(neg_zero_bin.w(), 0);

    assert_eq!(Vec4::new(1.0, 2.0, 3.0, 4.0) + Vec4::new(5.0, 6.0, 7.0, 8.0), Vec4::new(6.0, 8.0, 10.0, 12.0));
    assert_eq!(Vec4::new(1.0, 2.0, 3.0, 4.0) - Vec4::new(8.0, 7.0, 6.0, 5.0), Vec4::new(-7.0, -5.0, -3.0, -1.0));

    assert_eq!(Vec4::new(1.0, 2.0, 3.0, 4.0) * Vec4::new(5.0, 6.0, 7.0, 8.0), Vec4::new(5.0, 12.0, 21.0, 32.0));
    assert_eq!(Vec4::new(1.0, 2.0, 3.0, 4.0) * 2.0, Vec4::new(2.0, 4.0, 6.0, 8.0));
    assert_eq!(4.0 * Vec4::new(1.0, 2.0, 3.0, 4.0), Vec4::new(4.0, 8.0, 12.0, 16.0));

    assert_eq!(Vec4::new(1.0, 2.0, 3.0, 4.0) / 2.0, Vec4::new(0.5, 1.0, 1.5, 2.0));
    assert_eq!(Vec4::new(1.0, 2.0, 3.0, 4.0) / Vec4::new(2.0, 8.0, 24.0, 64.0), Vec4::new(0.5, 0.25, 0.125, 0.0625));

    let mut v = Vec4::new(1.0, 2.0, 3.0, 4.0);
    v *= Vec4::new(5.0, 6.0, 7.0, 8.0);
    assert_eq!(v, Vec4::new(5.0, 12.0, 21.0, 32.0));
    v *= 2.0;
    assert_eq!(v, Vec4::new(10.0, 24.0, 42.0, 64.0));
    v /= 2.0;
    assert_eq!(v, Vec4::new(5.0, 12.0, 21.0, 32.0));
    v += Vec4::new(1.0, 2.0, 3.0, 4.0);
    assert_eq!(v, Vec4::new(6.0, 14.0, 24.0, 36.0));

    assert_eq!(Vec4::new(2.0, 4.0, 8.0, 16.0).reciprocal(), Vec4::new(0.5, 0.25, 0.125, 0.0625));
}

#[test]
fn test_vec4_swizzle() {
    let v = Vec4::new(1.0, 2.0, 3.0, 4.0);

    assert_eq!(v.splat_x(), Vec4::replicate(1.0));
    assert_eq!(v.splat_y(), Vec4::replicate(2.0));
    assert_eq!(v.splat_z(), Vec4::replicate(3.0));
    assert_eq!(v.splat_w(), Vec4::replicate(4.0));

    assert_eq!(v.splat_x3(), Vec3::replicate(1.0));
    assert_eq!(v.splat_y3(), Vec3::replicate(2.0));
    assert_eq!(v.splat_z3(), Vec3::replicate(3.0));
    assert_eq!(v.splat_w3(), Vec3::replicate(4.0));

    assert_eq!(v.swizzle::<SWIZZLE_X, SWIZZLE_X, SWIZZLE_X, SWIZZLE_X>(), Vec4::new(1.0, 1.0, 1.0, 1.0));
    assert_eq!(v.swizzle::<SWIZZLE_X, SWIZZLE_X, SWIZZLE_X, SWIZZLE_Y>(), Vec4::new(1.0, 1.0, 1.0, 2.0));
    assert_eq!(v.swizzle::<SWIZZLE_X, SWIZZLE_X, SWIZZLE_X, SWIZZLE_Z>(), Vec4::new(1.0, 1.0, 1.0, 3.0));
    assert_eq!(v.swizzle::<SWIZZLE_X, SWIZZLE_X, SWIZZLE_X, SWIZZLE_W>(), Vec4::new(1.0, 1.0, 1.0, 4.0));
    assert_eq!(v.swizzle::<SWIZZLE_X, SWIZZLE_X, SWIZZLE_Y, SWIZZLE_X>(), Vec4::new(1.0, 1.0, 2.0, 1.0));
    assert_eq!(v.swizzle::<SWIZZLE_X, SWIZZLE_X, SWIZZLE_Y, SWIZZLE_Y>(), Vec4::new(1.0, 1.0, 2.0, 2.0));
    assert_eq!(v.swizzle::<SWIZZLE_X, SWIZZLE_X, SWIZZLE_Y, SWIZZLE_Z>(), Vec4::new(1.0, 1.0, 2.0, 3.0));
    assert_eq!(v.swizzle::<SWIZZLE_X, SWIZZLE_X, SWIZZLE_Y, SWIZZLE_W>(), Vec4::new(1.0, 1.0, 2.0, 4.0));
    assert_eq!(v.swizzle::<SWIZZLE_X, SWIZZLE_X, SWIZZLE_Z, SWIZZLE_X>(), Vec4::new(1.0, 1.0, 3.0, 1.0));
    assert_eq!(v.swizzle::<SWIZZLE_X, SWIZZLE_X, SWIZZLE_Z, SWIZZLE_Y>(), Vec4::new(1.0, 1.0, 3.0, 2.0));
    assert_eq!(v.swizzle::<SWIZZLE_X, SWIZZLE_X, SWIZZLE_Z, SWIZZLE_Z>(), Vec4::new(1.0, 1.0, 3.0, 3.0));
    assert_eq!(v.swizzle::<SWIZZLE_X, SWIZZLE_X, SWIZZLE_Z, SWIZZLE_W>(), Vec4::new(1.0, 1.0, 3.0, 4.0));
    assert_eq!(v.swizzle::<SWIZZLE_X, SWIZZLE_X, SWIZZLE_W, SWIZZLE_X>(), Vec4::new(1.0, 1.0, 4.0, 1.0));
    assert_eq!(v.swizzle::<SWIZZLE_X, SWIZZLE_X, SWIZZLE_W, SWIZZLE_Y>(), Vec4::new(1.0, 1.0, 4.0, 2.0));
    assert_eq!(v.swizzle::<SWIZZLE_X, SWIZZLE_X, SWIZZLE_W, SWIZZLE_Z>(), Vec4::new(1.0, 1.0, 4.0, 3.0));
    assert_eq!(v.swizzle::<SWIZZLE_X, SWIZZLE_X, SWIZZLE_W, SWIZZLE_W>(), Vec4::new(1.0, 1.0, 4.0, 4.0));
    assert_eq!(v.swizzle::<SWIZZLE_X, SWIZZLE_Y, SWIZZLE_X, SWIZZLE_X>(), Vec4::new(1.0, 2.0, 1.0, 1.0));
    assert_eq!(v.swizzle::<SWIZZLE_X, SWIZZLE_Y, SWIZZLE_X, SWIZZLE_Y>(), Vec4::new(1.0, 2.0, 1.0, 2.0));
    assert_eq!(v.swizzle::<SWIZZLE_X, SWIZZLE_Y, SWIZZLE_X, SWIZZLE_Z>(), Vec4::new(1.0, 2.0, 1.0, 3.0));
    assert_eq!(v.swizzle::<SWIZZLE_X, SWIZZLE_Y, SWIZZLE_X, SWIZZLE_W>(), Vec4::new(1.0, 2.0, 1.0, 4.0));
    assert_eq!(v.swizzle::<SWIZZLE_X, SWIZZLE_Y, SWIZZLE_Y, SWIZZLE_X>(), Vec4::new(1.0, 2.0, 2.0, 1.0));
    assert_eq!(v.swizzle::<SWIZZLE_X, SWIZZLE_Y, SWIZZLE_Y, SWIZZLE_Y>(), Vec4::new(1.0, 2.0, 2.0, 2.0));
    assert_eq!(v.swizzle::<SWIZZLE_X, SWIZZLE_Y, SWIZZLE_Y, SWIZZLE_Z>(), Vec4::new(1.0, 2.0, 2.0, 3.0));
    assert_eq!(v.swizzle::<SWIZZLE_X, SWIZZLE_Y, SWIZZLE_Y, SWIZZLE_W>(), Vec4::new(1.0, 2.0, 2.0, 4.0));
    assert_eq!(v.swizzle::<SWIZZLE_X, SWIZZLE_Y, SWIZZLE_Z, SWIZZLE_X>(), Vec4::new(1.0, 2.0, 3.0, 1.0));
    assert_eq!(v.swizzle::<SWIZZLE_X, SWIZZLE_Y, SWIZZLE_Z, SWIZZLE_Y>(), Vec4::new(1.0, 2.0, 3.0, 2.0));
    assert_eq!(v.swizzle::<SWIZZLE_X, SWIZZLE_Y, SWIZZLE_Z, SWIZZLE_Z>(), Vec4::new(1.0, 2.0, 3.0, 3.0));
    assert_eq!(v.swizzle::<SWIZZLE_X, SWIZZLE_Y, SWIZZLE_Z, SWIZZLE_W>(), Vec4::new(1.0, 2.0, 3.0, 4.0));
    assert_eq!(v.swizzle::<SWIZZLE_X, SWIZZLE_Y, SWIZZLE_W, SWIZZLE_X>(), Vec4::new(1.0, 2.0, 4.0, 1.0));
    assert_eq!(v.swizzle::<SWIZZLE_X, SWIZZLE_Y, SWIZZLE_W, SWIZZLE_Y>(), Vec4::new(1.0, 2.0, 4.0, 2.0));
    assert_eq!(v.swizzle::<SWIZZLE_X, SWIZZLE_Y, SWIZZLE_W, SWIZZLE_Z>(), Vec4::new(1.0, 2.0, 4.0, 3.0));
    assert_eq!(v.swizzle::<SWIZZLE_X, SWIZZLE_Y, SWIZZLE_W, SWIZZLE_W>(), Vec4::new(1.0, 2.0, 4.0, 4.0));
    assert_eq!(v.swizzle::<SWIZZLE_X, SWIZZLE_Z, SWIZZLE_X, SWIZZLE_X>(), Vec4::new(1.0, 3.0, 1.0, 1.0));
    assert_eq!(v.swizzle::<SWIZZLE_X, SWIZZLE_Z, SWIZZLE_X, SWIZZLE_Y>(), Vec4::new(1.0, 3.0, 1.0, 2.0));
    assert_eq!(v.swizzle::<SWIZZLE_X, SWIZZLE_Z, SWIZZLE_X, SWIZZLE_Z>(), Vec4::new(1.0, 3.0, 1.0, 3.0));
    assert_eq!(v.swizzle::<SWIZZLE_X, SWIZZLE_Z, SWIZZLE_X, SWIZZLE_W>(), Vec4::new(1.0, 3.0, 1.0, 4.0));
    assert_eq!(v.swizzle::<SWIZZLE_X, SWIZZLE_Z, SWIZZLE_Y, SWIZZLE_X>(), Vec4::new(1.0, 3.0, 2.0, 1.0));
    assert_eq!(v.swizzle::<SWIZZLE_X, SWIZZLE_Z, SWIZZLE_Y, SWIZZLE_Y>(), Vec4::new(1.0, 3.0, 2.0, 2.0));
    assert_eq!(v.swizzle::<SWIZZLE_X, SWIZZLE_Z, SWIZZLE_Y, SWIZZLE_Z>(), Vec4::new(1.0, 3.0, 2.0, 3.0));
    assert_eq!(v.swizzle::<SWIZZLE_X, SWIZZLE_Z, SWIZZLE_Y, SWIZZLE_W>(), Vec4::new(1.0, 3.0, 2.0, 4.0));
    assert_eq!(v.swizzle::<SWIZZLE_X, SWIZZLE_Z, SWIZZLE_Z, SWIZZLE_X>(), Vec4::new(1.0, 3.0, 3.0, 1.0));
    assert_eq!(v.swizzle::<SWIZZLE_X, SWIZZLE_Z, SWIZZLE_Z, SWIZZLE_Y>(), Vec4::new(1.0, 3.0, 3.0, 2.0));
    assert_eq!(v.swizzle::<SWIZZLE_X, SWIZZLE_Z, SWIZZLE_Z, SWIZZLE_Z>(), Vec4::new(1.0, 3.0, 3.0, 3.0));
    assert_eq!(v.swizzle::<SWIZZLE_X, SWIZZLE_Z, SWIZZLE_Z, SWIZZLE_W>(), Vec4::new(1.0, 3.0, 3.0, 4.0));
    assert_eq!(v.swizzle::<SWIZZLE_X, SWIZZLE_Z, SWIZZLE_W, SWIZZLE_X>(), Vec4::new(1.0, 3.0, 4.0, 1.0));
    assert_eq!(v.swizzle::<SWIZZLE_X, SWIZZLE_Z, SWIZZLE_W, SWIZZLE_Y>(), Vec4::new(1.0, 3.0, 4.0, 2.0));
    assert_eq!(v.swizzle::<SWIZZLE_X, SWIZZLE_Z, SWIZZLE_W, SWIZZLE_Z>(), Vec4::new(1.0, 3.0, 4.0, 3.0));
    assert_eq!(v.swizzle::<SWIZZLE_X, SWIZZLE_Z, SWIZZLE_W, SWIZZLE_W>(), Vec4::new(1.0, 3.0, 4.0, 4.0));
    assert_eq!(v.swizzle::<SWIZZLE_X, SWIZZLE_W, SWIZZLE_X, SWIZZLE_X>(), Vec4::new(1.0, 4.0, 1.0, 1.0));
    assert_eq!(v.swizzle::<SWIZZLE_X, SWIZZLE_W, SWIZZLE_X, SWIZZLE_Y>(), Vec4::new(1.0, 4.0, 1.0, 2.0));
    assert_eq!(v.swizzle::<SWIZZLE_X, SWIZZLE_W, SWIZZLE_X, SWIZZLE_Z>(), Vec4::new(1.0, 4.0, 1.0, 3.0));
    assert_eq!(v.swizzle::<SWIZZLE_X, SWIZZLE_W, SWIZZLE_X, SWIZZLE_W>(), Vec4::new(1.0, 4.0, 1.0, 4.0));
    assert_eq!(v.swizzle::<SWIZZLE_X, SWIZZLE_W, SWIZZLE_Y, SWIZZLE_X>(), Vec4::new(1.0, 4.0, 2.0, 1.0));
    assert_eq!(v.swizzle::<SWIZZLE_X, SWIZZLE_W, SWIZZLE_Y, SWIZZLE_Y>(), Vec4::new(1.0, 4.0, 2.0, 2.0));
    assert_eq!(v.swizzle::<SWIZZLE_X, SWIZZLE_W, SWIZZLE_Y, SWIZZLE_Z>(), Vec4::new(1.0, 4.0, 2.0, 3.0));
    assert_eq!(v.swizzle::<SWIZZLE_X, SWIZZLE_W, SWIZZLE_Y, SWIZZLE_W>(), Vec4::new(1.0, 4.0, 2.0, 4.0));
    assert_eq!(v.swizzle::<SWIZZLE_X, SWIZZLE_W, SWIZZLE_Z, SWIZZLE_X>(), Vec4::new(1.0, 4.0, 3.0, 1.0));
    assert_eq!(v.swizzle::<SWIZZLE_X, SWIZZLE_W, SWIZZLE_Z, SWIZZLE_Y>(), Vec4::new(1.0, 4.0, 3.0, 2.0));
    assert_eq!(v.swizzle::<SWIZZLE_X, SWIZZLE_W, SWIZZLE_Z, SWIZZLE_Z>(), Vec4::new(1.0, 4.0, 3.0, 3.0));
    assert_eq!(v.swizzle::<SWIZZLE_X, SWIZZLE_W, SWIZZLE_Z, SWIZZLE_W>(), Vec4::new(1.0, 4.0, 3.0, 4.0));
    assert_eq!(v.swizzle::<SWIZZLE_X, SWIZZLE_W, SWIZZLE_W, SWIZZLE_X>(), Vec4::new(1.0, 4.0, 4.0, 1.0));
    assert_eq!(v.swizzle::<SWIZZLE_X, SWIZZLE_W, SWIZZLE_W, SWIZZLE_Y>(), Vec4::new(1.0, 4.0, 4.0, 2.0));
    assert_eq!(v.swizzle::<SWIZZLE_X, SWIZZLE_W, SWIZZLE_W, SWIZZLE_Z>(), Vec4::new(1.0, 4.0, 4.0, 3.0));
    assert_eq!(v.swizzle::<SWIZZLE_X, SWIZZLE_W, SWIZZLE_W, SWIZZLE_W>(), Vec4::new(1.0, 4.0, 4.0, 4.0));

    assert_eq!(v.swizzle::<SWIZZLE_Y, SWIZZLE_X, SWIZZLE_X, SWIZZLE_X>(), Vec4::new(2.0, 1.0, 1.0, 1.0));
    assert_eq!(v.swizzle::<SWIZZLE_Y, SWIZZLE_X, SWIZZLE_X, SWIZZLE_Y>(), Vec4::new(2.0, 1.0, 1.0, 2.0));
    assert_eq!(v.swizzle::<SWIZZLE_Y, SWIZZLE_X, SWIZZLE_X, SWIZZLE_Z>(), Vec4::new(2.0, 1.0, 1.0, 3.0));
    assert_eq!(v.swizzle::<SWIZZLE_Y, SWIZZLE_X, SWIZZLE_X, SWIZZLE_W>(), Vec4::new(2.0, 1.0, 1.0, 4.0));
    assert_eq!(v.swizzle::<SWIZZLE_Y, SWIZZLE_X, SWIZZLE_Y, SWIZZLE_X>(), Vec4::new(2.0, 1.0, 2.0, 1.0));
    assert_eq!(v.swizzle::<SWIZZLE_Y, SWIZZLE_X, SWIZZLE_Y, SWIZZLE_Y>(), Vec4::new(2.0, 1.0, 2.0, 2.0));
    assert_eq!(v.swizzle::<SWIZZLE_Y, SWIZZLE_X, SWIZZLE_Y, SWIZZLE_Z>(), Vec4::new(2.0, 1.0, 2.0, 3.0));
    assert_eq!(v.swizzle::<SWIZZLE_Y, SWIZZLE_X, SWIZZLE_Y, SWIZZLE_W>(), Vec4::new(2.0, 1.0, 2.0, 4.0));
    assert_eq!(v.swizzle::<SWIZZLE_Y, SWIZZLE_X, SWIZZLE_Z, SWIZZLE_X>(), Vec4::new(2.0, 1.0, 3.0, 1.0));
    assert_eq!(v.swizzle::<SWIZZLE_Y, SWIZZLE_X, SWIZZLE_Z, SWIZZLE_Y>(), Vec4::new(2.0, 1.0, 3.0, 2.0));
    assert_eq!(v.swizzle::<SWIZZLE_Y, SWIZZLE_X, SWIZZLE_Z, SWIZZLE_Z>(), Vec4::new(2.0, 1.0, 3.0, 3.0));
    assert_eq!(v.swizzle::<SWIZZLE_Y, SWIZZLE_X, SWIZZLE_Z, SWIZZLE_W>(), Vec4::new(2.0, 1.0, 3.0, 4.0));
    assert_eq!(v.swizzle::<SWIZZLE_Y, SWIZZLE_X, SWIZZLE_W, SWIZZLE_X>(), Vec4::new(2.0, 1.0, 4.0, 1.0));
    assert_eq!(v.swizzle::<SWIZZLE_Y, SWIZZLE_X, SWIZZLE_W, SWIZZLE_Y>(), Vec4::new(2.0, 1.0, 4.0, 2.0));
    assert_eq!(v.swizzle::<SWIZZLE_Y, SWIZZLE_X, SWIZZLE_W, SWIZZLE_Z>(), Vec4::new(2.0, 1.0, 4.0, 3.0));
    assert_eq!(v.swizzle::<SWIZZLE_Y, SWIZZLE_X, SWIZZLE_W, SWIZZLE_W>(), Vec4::new(2.0, 1.0, 4.0, 4.0));
    assert_eq!(v.swizzle::<SWIZZLE_Y, SWIZZLE_Y, SWIZZLE_X, SWIZZLE_X>(), Vec4::new(2.0, 2.0, 1.0, 1.0));
    assert_eq!(v.swizzle::<SWIZZLE_Y, SWIZZLE_Y, SWIZZLE_X, SWIZZLE_Y>(), Vec4::new(2.0, 2.0, 1.0, 2.0));
    assert_eq!(v.swizzle::<SWIZZLE_Y, SWIZZLE_Y, SWIZZLE_X, SWIZZLE_Z>(), Vec4::new(2.0, 2.0, 1.0, 3.0));
    assert_eq!(v.swizzle::<SWIZZLE_Y, SWIZZLE_Y, SWIZZLE_X, SWIZZLE_W>(), Vec4::new(2.0, 2.0, 1.0, 4.0));
    assert_eq!(v.swizzle::<SWIZZLE_Y, SWIZZLE_Y, SWIZZLE_Y, SWIZZLE_X>(), Vec4::new(2.0, 2.0, 2.0, 1.0));
    assert_eq!(v.swizzle::<SWIZZLE_Y, SWIZZLE_Y, SWIZZLE_Y, SWIZZLE_Y>(), Vec4::new(2.0, 2.0, 2.0, 2.0));
    assert_eq!(v.swizzle::<SWIZZLE_Y, SWIZZLE_Y, SWIZZLE_Y, SWIZZLE_Z>(), Vec4::new(2.0, 2.0, 2.0, 3.0));
    assert_eq!(v.swizzle::<SWIZZLE_Y, SWIZZLE_Y, SWIZZLE_Y, SWIZZLE_W>(), Vec4::new(2.0, 2.0, 2.0, 4.0));
    assert_eq!(v.swizzle::<SWIZZLE_Y, SWIZZLE_Y, SWIZZLE_Z, SWIZZLE_X>(), Vec4::new(2.0, 2.0, 3.0, 1.0));
    assert_eq!(v.swizzle::<SWIZZLE_Y, SWIZZLE_Y, SWIZZLE_Z, SWIZZLE_Y>(), Vec4::new(2.0, 2.0, 3.0, 2.0));
    assert_eq!(v.swizzle::<SWIZZLE_Y, SWIZZLE_Y, SWIZZLE_Z, SWIZZLE_Z>(), Vec4::new(2.0, 2.0, 3.0, 3.0));
    assert_eq!(v.swizzle::<SWIZZLE_Y, SWIZZLE_Y, SWIZZLE_Z, SWIZZLE_W>(), Vec4::new(2.0, 2.0, 3.0, 4.0));
    assert_eq!(v.swizzle::<SWIZZLE_Y, SWIZZLE_Y, SWIZZLE_W, SWIZZLE_X>(), Vec4::new(2.0, 2.0, 4.0, 1.0));
    assert_eq!(v.swizzle::<SWIZZLE_Y, SWIZZLE_Y, SWIZZLE_W, SWIZZLE_Y>(), Vec4::new(2.0, 2.0, 4.0, 2.0));
    assert_eq!(v.swizzle::<SWIZZLE_Y, SWIZZLE_Y, SWIZZLE_W, SWIZZLE_Z>(), Vec4::new(2.0, 2.0, 4.0, 3.0));
    assert_eq!(v.swizzle::<SWIZZLE_Y, SWIZZLE_Y, SWIZZLE_W, SWIZZLE_W>(), Vec4::new(2.0, 2.0, 4.0, 4.0));
    assert_eq!(v.swizzle::<SWIZZLE_Y, SWIZZLE_Z, SWIZZLE_X, SWIZZLE_X>(), Vec4::new(2.0, 3.0, 1.0, 1.0));
    assert_eq!(v.swizzle::<SWIZZLE_Y, SWIZZLE_Z, SWIZZLE_X, SWIZZLE_Y>(), Vec4::new(2.0, 3.0, 1.0, 2.0));
    assert_eq!(v.swizzle::<SWIZZLE_Y, SWIZZLE_Z, SWIZZLE_X, SWIZZLE_Z>(), Vec4::new(2.0, 3.0, 1.0, 3.0));
    assert_eq!(v.swizzle::<SWIZZLE_Y, SWIZZLE_Z, SWIZZLE_X, SWIZZLE_W>(), Vec4::new(2.0, 3.0, 1.0, 4.0));
    assert_eq!(v.swizzle::<SWIZZLE_Y, SWIZZLE_Z, SWIZZLE_Y, SWIZZLE_X>(), Vec4::new(2.0, 3.0, 2.0, 1.0));
    assert_eq!(v.swizzle::<SWIZZLE_Y, SWIZZLE_Z, SWIZZLE_Y, SWIZZLE_Y>(), Vec4::new(2.0, 3.0, 2.0, 2.0));
    assert_eq!(v.swizzle::<SWIZZLE_Y, SWIZZLE_Z, SWIZZLE_Y, SWIZZLE_Z>(), Vec4::new(2.0, 3.0, 2.0, 3.0));
    assert_eq!(v.swizzle::<SWIZZLE_Y, SWIZZLE_Z, SWIZZLE_Y, SWIZZLE_W>(), Vec4::new(2.0, 3.0, 2.0, 4.0));
    assert_eq!(v.swizzle::<SWIZZLE_Y, SWIZZLE_Z, SWIZZLE_Z, SWIZZLE_X>(), Vec4::new(2.0, 3.0, 3.0, 1.0));
    assert_eq!(v.swizzle::<SWIZZLE_Y, SWIZZLE_Z, SWIZZLE_Z, SWIZZLE_Y>(), Vec4::new(2.0, 3.0, 3.0, 2.0));
    assert_eq!(v.swizzle::<SWIZZLE_Y, SWIZZLE_Z, SWIZZLE_Z, SWIZZLE_Z>(), Vec4::new(2.0, 3.0, 3.0, 3.0));
    assert_eq!(v.swizzle::<SWIZZLE_Y, SWIZZLE_Z, SWIZZLE_Z, SWIZZLE_W>(), Vec4::new(2.0, 3.0, 3.0, 4.0));
    assert_eq!(v.swizzle::<SWIZZLE_Y, SWIZZLE_Z, SWIZZLE_W, SWIZZLE_X>(), Vec4::new(2.0, 3.0, 4.0, 1.0));
    assert_eq!(v.swizzle::<SWIZZLE_Y, SWIZZLE_Z, SWIZZLE_W, SWIZZLE_Y>(), Vec4::new(2.0, 3.0, 4.0, 2.0));
    assert_eq!(v.swizzle::<SWIZZLE_Y, SWIZZLE_Z, SWIZZLE_W, SWIZZLE_Z>(), Vec4::new(2.0, 3.0, 4.0, 3.0));
    assert_eq!(v.swizzle::<SWIZZLE_Y, SWIZZLE_Z, SWIZZLE_W, SWIZZLE_W>(), Vec4::new(2.0, 3.0, 4.0, 4.0));
    assert_eq!(v.swizzle::<SWIZZLE_Y, SWIZZLE_W, SWIZZLE_X, SWIZZLE_X>(), Vec4::new(2.0, 4.0, 1.0, 1.0));
    assert_eq!(v.swizzle::<SWIZZLE_Y, SWIZZLE_W, SWIZZLE_X, SWIZZLE_Y>(), Vec4::new(2.0, 4.0, 1.0, 2.0));
    assert_eq!(v.swizzle::<SWIZZLE_Y, SWIZZLE_W, SWIZZLE_X, SWIZZLE_Z>(), Vec4::new(2.0, 4.0, 1.0, 3.0));
    assert_eq!(v.swizzle::<SWIZZLE_Y, SWIZZLE_W, SWIZZLE_X, SWIZZLE_W>(), Vec4::new(2.0, 4.0, 1.0, 4.0));
    assert_eq!(v.swizzle::<SWIZZLE_Y, SWIZZLE_W, SWIZZLE_Y, SWIZZLE_X>(), Vec4::new(2.0, 4.0, 2.0, 1.0));
    assert_eq!(v.swizzle::<SWIZZLE_Y, SWIZZLE_W, SWIZZLE_Y, SWIZZLE_Y>(), Vec4::new(2.0, 4.0, 2.0, 2.0));
    assert_eq!(v.swizzle::<SWIZZLE_Y, SWIZZLE_W, SWIZZLE_Y, SWIZZLE_Z>(), Vec4::new(2.0, 4.0, 2.0, 3.0));
    assert_eq!(v.swizzle::<SWIZZLE_Y, SWIZZLE_W, SWIZZLE_Y, SWIZZLE_W>(), Vec4::new(2.0, 4.0, 2.0, 4.0));
    assert_eq!(v.swizzle::<SWIZZLE_Y, SWIZZLE_W, SWIZZLE_Z, SWIZZLE_X>(), Vec4::new(2.0, 4.0, 3.0, 1.0));
    assert_eq!(v.swizzle::<SWIZZLE_Y, SWIZZLE_W, SWIZZLE_Z, SWIZZLE_Y>(), Vec4::new(2.0, 4.0, 3.0, 2.0));
    assert_eq!(v.swizzle::<SWIZZLE_Y, SWIZZLE_W, SWIZZLE_Z, SWIZZLE_Z>(), Vec4::new(2.0, 4.0, 3.0, 3.0));
    assert_eq!(v.swizzle::<SWIZZLE_Y, SWIZZLE_W, SWIZZLE_Z, SWIZZLE_W>(), Vec4::new(2.0, 4.0, 3.0, 4.0));
    assert_eq!(v.swizzle::<SWIZZLE_Y, SWIZZLE_W, SWIZZLE_W, SWIZZLE_X>(), Vec4::new(2.0, 4.0, 4.0, 1.0));
    assert_eq!(v.swizzle::<SWIZZLE_Y, SWIZZLE_W, SWIZZLE_W, SWIZZLE_Y>(), Vec4::new(2.0, 4.0, 4.0, 2.0));
    assert_eq!(v.swizzle::<SWIZZLE_Y, SWIZZLE_W, SWIZZLE_W, SWIZZLE_Z>(), Vec4::new(2.0, 4.0, 4.0, 3.0));
    assert_eq!(v.swizzle::<SWIZZLE_Y, SWIZZLE_W, SWIZZLE_W, SWIZZLE_W>(), Vec4::new(2.0, 4.0, 4.0, 4.0));

    assert_eq!(v.swizzle::<SWIZZLE_Z, SWIZZLE_X, SWIZZLE_X, SWIZZLE_X>(), Vec4::new(3.0, 1.0, 1.0, 1.0));
    assert_eq!(v.swizzle::<SWIZZLE_Z, SWIZZLE_X, SWIZZLE_X, SWIZZLE_Y>(), Vec4::new(3.0, 1.0, 1.0, 2.0));
    assert_eq!(v.swizzle::<SWIZZLE_Z, SWIZZLE_X, SWIZZLE_X, SWIZZLE_Z>(), Vec4::new(3.0, 1.0, 1.0, 3.0));
    assert_eq!(v.swizzle::<SWIZZLE_Z, SWIZZLE_X, SWIZZLE_X, SWIZZLE_W>(), Vec4::new(3.0, 1.0, 1.0, 4.0));
    assert_eq!(v.swizzle::<SWIZZLE_Z, SWIZZLE_X, SWIZZLE_Y, SWIZZLE_X>(), Vec4::new(3.0, 1.0, 2.0, 1.0));
    assert_eq!(v.swizzle::<SWIZZLE_Z, SWIZZLE_X, SWIZZLE_Y, SWIZZLE_Y>(), Vec4::new(3.0, 1.0, 2.0, 2.0));
    assert_eq!(v.swizzle::<SWIZZLE_Z, SWIZZLE_X, SWIZZLE_Y, SWIZZLE_Z>(), Vec4::new(3.0, 1.0, 2.0, 3.0));
    assert_eq!(v.swizzle::<SWIZZLE_Z, SWIZZLE_X, SWIZZLE_Y, SWIZZLE_W>(), Vec4::new(3.0, 1.0, 2.0, 4.0));
    assert_eq!(v.swizzle::<SWIZZLE_Z, SWIZZLE_X, SWIZZLE_Z, SWIZZLE_X>(), Vec4::new(3.0, 1.0, 3.0, 1.0));
    assert_eq!(v.swizzle::<SWIZZLE_Z, SWIZZLE_X, SWIZZLE_Z, SWIZZLE_Y>(), Vec4::new(3.0, 1.0, 3.0, 2.0));
    assert_eq!(v.swizzle::<SWIZZLE_Z, SWIZZLE_X, SWIZZLE_Z, SWIZZLE_Z>(), Vec4::new(3.0, 1.0, 3.0, 3.0));
    assert_eq!(v.swizzle::<SWIZZLE_Z, SWIZZLE_X, SWIZZLE_Z, SWIZZLE_W>(), Vec4::new(3.0, 1.0, 3.0, 4.0));
    assert_eq!(v.swizzle::<SWIZZLE_Z, SWIZZLE_X, SWIZZLE_W, SWIZZLE_X>(), Vec4::new(3.0, 1.0, 4.0, 1.0));
    assert_eq!(v.swizzle::<SWIZZLE_Z, SWIZZLE_X, SWIZZLE_W, SWIZZLE_Y>(), Vec4::new(3.0, 1.0, 4.0, 2.0));
    assert_eq!(v.swizzle::<SWIZZLE_Z, SWIZZLE_X, SWIZZLE_W, SWIZZLE_Z>(), Vec4::new(3.0, 1.0, 4.0, 3.0));
    assert_eq!(v.swizzle::<SWIZZLE_Z, SWIZZLE_X, SWIZZLE_W, SWIZZLE_W>(), Vec4::new(3.0, 1.0, 4.0, 4.0));
    assert_eq!(v.swizzle::<SWIZZLE_Z, SWIZZLE_Y, SWIZZLE_X, SWIZZLE_X>(), Vec4::new(3.0, 2.0, 1.0, 1.0));
    assert_eq!(v.swizzle::<SWIZZLE_Z, SWIZZLE_Y, SWIZZLE_X, SWIZZLE_Y>(), Vec4::new(3.0, 2.0, 1.0, 2.0));
    assert_eq!(v.swizzle::<SWIZZLE_Z, SWIZZLE_Y, SWIZZLE_X, SWIZZLE_Z>(), Vec4::new(3.0, 2.0, 1.0, 3.0));
    assert_eq!(v.swizzle::<SWIZZLE_Z, SWIZZLE_Y, SWIZZLE_X, SWIZZLE_W>(), Vec4::new(3.0, 2.0, 1.0, 4.0));
    assert_eq!(v.swizzle::<SWIZZLE_Z, SWIZZLE_Y, SWIZZLE_Y, SWIZZLE_X>(), Vec4::new(3.0, 2.0, 2.0, 1.0));
    assert_eq!(v.swizzle::<SWIZZLE_Z, SWIZZLE_Y, SWIZZLE_Y, SWIZZLE_Y>(), Vec4::new(3.0, 2.0, 2.0, 2.0));
    assert_eq!(v.swizzle::<SWIZZLE_Z, SWIZZLE_Y, SWIZZLE_Y, SWIZZLE_Z>(), Vec4::new(3.0, 2.0, 2.0, 3.0));
    assert_eq!(v.swizzle::<SWIZZLE_Z, SWIZZLE_Y, SWIZZLE_Y, SWIZZLE_W>(), Vec4::new(3.0, 2.0, 2.0, 4.0));
    assert_eq!(v.swizzle::<SWIZZLE_Z, SWIZZLE_Y, SWIZZLE_Z, SWIZZLE_X>(), Vec4::new(3.0, 2.0, 3.0, 1.0));
    assert_eq!(v.swizzle::<SWIZZLE_Z, SWIZZLE_Y, SWIZZLE_Z, SWIZZLE_Y>(), Vec4::new(3.0, 2.0, 3.0, 2.0));
    assert_eq!(v.swizzle::<SWIZZLE_Z, SWIZZLE_Y, SWIZZLE_Z, SWIZZLE_Z>(), Vec4::new(3.0, 2.0, 3.0, 3.0));
    assert_eq!(v.swizzle::<SWIZZLE_Z, SWIZZLE_Y, SWIZZLE_Z, SWIZZLE_W>(), Vec4::new(3.0, 2.0, 3.0, 4.0));
    assert_eq!(v.swizzle::<SWIZZLE_Z, SWIZZLE_Y, SWIZZLE_W, SWIZZLE_X>(), Vec4::new(3.0, 2.0, 4.0, 1.0));
    assert_eq!(v.swizzle::<SWIZZLE_Z, SWIZZLE_Y, SWIZZLE_W, SWIZZLE_Y>(), Vec4::new(3.0, 2.0, 4.0, 2.0));
    assert_eq!(v.swizzle::<SWIZZLE_Z, SWIZZLE_Y, SWIZZLE_W, SWIZZLE_Z>(), Vec4::new(3.0, 2.0, 4.0, 3.0));
    assert_eq!(v.swizzle::<SWIZZLE_Z, SWIZZLE_Y, SWIZZLE_W, SWIZZLE_W>(), Vec4::new(3.0, 2.0, 4.0, 4.0));
    assert_eq!(v.swizzle::<SWIZZLE_Z, SWIZZLE_Z, SWIZZLE_X, SWIZZLE_X>(), Vec4::new(3.0, 3.0, 1.0, 1.0));
    assert_eq!(v.swizzle::<SWIZZLE_Z, SWIZZLE_Z, SWIZZLE_X, SWIZZLE_Y>(), Vec4::new(3.0, 3.0, 1.0, 2.0));
    assert_eq!(v.swizzle::<SWIZZLE_Z, SWIZZLE_Z, SWIZZLE_X, SWIZZLE_Z>(), Vec4::new(3.0, 3.0, 1.0, 3.0));
    assert_eq!(v.swizzle::<SWIZZLE_Z, SWIZZLE_Z, SWIZZLE_X, SWIZZLE_W>(), Vec4::new(3.0, 3.0, 1.0, 4.0));
    assert_eq!(v.swizzle::<SWIZZLE_Z, SWIZZLE_Z, SWIZZLE_Y, SWIZZLE_X>(), Vec4::new(3.0, 3.0, 2.0, 1.0));
    assert_eq!(v.swizzle::<SWIZZLE_Z, SWIZZLE_Z, SWIZZLE_Y, SWIZZLE_Y>(), Vec4::new(3.0, 3.0, 2.0, 2.0));
    assert_eq!(v.swizzle::<SWIZZLE_Z, SWIZZLE_Z, SWIZZLE_Y, SWIZZLE_Z>(), Vec4::new(3.0, 3.0, 2.0, 3.0));
    assert_eq!(v.swizzle::<SWIZZLE_Z, SWIZZLE_Z, SWIZZLE_Y, SWIZZLE_W>(), Vec4::new(3.0, 3.0, 2.0, 4.0));
    assert_eq!(v.swizzle::<SWIZZLE_Z, SWIZZLE_Z, SWIZZLE_Z, SWIZZLE_X>(), Vec4::new(3.0, 3.0, 3.0, 1.0));
    assert_eq!(v.swizzle::<SWIZZLE_Z, SWIZZLE_Z, SWIZZLE_Z, SWIZZLE_Y>(), Vec4::new(3.0, 3.0, 3.0, 2.0));
    assert_eq!(v.swizzle::<SWIZZLE_Z, SWIZZLE_Z, SWIZZLE_Z, SWIZZLE_Z>(), Vec4::new(3.0, 3.0, 3.0, 3.0));
    assert_eq!(v.swizzle::<SWIZZLE_Z, SWIZZLE_Z, SWIZZLE_Z, SWIZZLE_W>(), Vec4::new(3.0, 3.0, 3.0, 4.0));
    assert_eq!(v.swizzle::<SWIZZLE_Z, SWIZZLE_Z, SWIZZLE_W, SWIZZLE_X>(), Vec4::new(3.0, 3.0, 4.0, 1.0));
    assert_eq!(v.swizzle::<SWIZZLE_Z, SWIZZLE_Z, SWIZZLE_W, SWIZZLE_Y>(), Vec4::new(3.0, 3.0, 4.0, 2.0));
    assert_eq!(v.swizzle::<SWIZZLE_Z, SWIZZLE_Z, SWIZZLE_W, SWIZZLE_Z>(), Vec4::new(3.0, 3.0, 4.0, 3.0));
    assert_eq!(v.swizzle::<SWIZZLE_Z, SWIZZLE_Z, SWIZZLE_W, SWIZZLE_W>(), Vec4::new(3.0, 3.0, 4.0, 4.0));
    assert_eq!(v.swizzle::<SWIZZLE_Z, SWIZZLE_W, SWIZZLE_X, SWIZZLE_X>(), Vec4::new(3.0, 4.0, 1.0, 1.0));
    assert_eq!(v.swizzle::<SWIZZLE_Z, SWIZZLE_W, SWIZZLE_X, SWIZZLE_Y>(), Vec4::new(3.0, 4.0, 1.0, 2.0));
    assert_eq!(v.swizzle::<SWIZZLE_Z, SWIZZLE_W, SWIZZLE_X, SWIZZLE_Z>(), Vec4::new(3.0, 4.0, 1.0, 3.0));
    assert_eq!(v.swizzle::<SWIZZLE_Z, SWIZZLE_W, SWIZZLE_X, SWIZZLE_W>(), Vec4::new(3.0, 4.0, 1.0, 4.0));
    assert_eq!(v.swizzle::<SWIZZLE_Z, SWIZZLE_W, SWIZZLE_Y, SWIZZLE_X>(), Vec4::new(3.0, 4.0, 2.0, 1.0));
    assert_eq!(v.swizzle::<SWIZZLE_Z, SWIZZLE_W, SWIZZLE_Y, SWIZZLE_Y>(), Vec4::new(3.0, 4.0, 2.0, 2.0));
    assert_eq!(v.swizzle::<SWIZZLE_Z, SWIZZLE_W, SWIZZLE_Y, SWIZZLE_Z>(), Vec4::new(3.0, 4.0, 2.0, 3.0));
    assert_eq!(v.swizzle::<SWIZZLE_Z, SWIZZLE_W, SWIZZLE_Y, SWIZZLE_W>(), Vec4::new(3.0, 4.0, 2.0, 4.0));
    assert_eq!(v.swizzle::<SWIZZLE_Z, SWIZZLE_W, SWIZZLE_Z, SWIZZLE_X>(), Vec4::new(3.0, 4.0, 3.0, 1.0));
    assert_eq!(v.swizzle::<SWIZZLE_Z, SWIZZLE_W, SWIZZLE_Z, SWIZZLE_Y>(), Vec4::new(3.0, 4.0, 3.0, 2.0));
    assert_eq!(v.swizzle::<SWIZZLE_Z, SWIZZLE_W, SWIZZLE_Z, SWIZZLE_Z>(), Vec4::new(3.0, 4.0, 3.0, 3.0));
    assert_eq!(v.swizzle::<SWIZZLE_Z, SWIZZLE_W, SWIZZLE_Z, SWIZZLE_W>(), Vec4::new(3.0, 4.0, 3.0, 4.0));
    assert_eq!(v.swizzle::<SWIZZLE_Z, SWIZZLE_W, SWIZZLE_W, SWIZZLE_X>(), Vec4::new(3.0, 4.0, 4.0, 1.0));
    assert_eq!(v.swizzle::<SWIZZLE_Z, SWIZZLE_W, SWIZZLE_W, SWIZZLE_Y>(), Vec4::new(3.0, 4.0, 4.0, 2.0));
    assert_eq!(v.swizzle::<SWIZZLE_Z, SWIZZLE_W, SWIZZLE_W, SWIZZLE_Z>(), Vec4::new(3.0, 4.0, 4.0, 3.0));
    assert_eq!(v.swizzle::<SWIZZLE_Z, SWIZZLE_W, SWIZZLE_W, SWIZZLE_W>(), Vec4::new(3.0, 4.0, 4.0, 4.0));

    assert_eq!(v.swizzle::<SWIZZLE_W, SWIZZLE_X, SWIZZLE_X, SWIZZLE_X>(), Vec4::new(4.0, 1.0, 1.0, 1.0));
    assert_eq!(v.swizzle::<SWIZZLE_W, SWIZZLE_X, SWIZZLE_X, SWIZZLE_Y>(), Vec4::new(4.0, 1.0, 1.0, 2.0));
    assert_eq!(v.swizzle::<SWIZZLE_W, SWIZZLE_X, SWIZZLE_X, SWIZZLE_Z>(), Vec4::new(4.0, 1.0, 1.0, 3.0));
    assert_eq!(v.swizzle::<SWIZZLE_W, SWIZZLE_X, SWIZZLE_X, SWIZZLE_W>(), Vec4::new(4.0, 1.0, 1.0, 4.0));
    assert_eq!(v.swizzle::<SWIZZLE_W, SWIZZLE_X, SWIZZLE_Y, SWIZZLE_X>(), Vec4::new(4.0, 1.0, 2.0, 1.0));
    assert_eq!(v.swizzle::<SWIZZLE_W, SWIZZLE_X, SWIZZLE_Y, SWIZZLE_Y>(), Vec4::new(4.0, 1.0, 2.0, 2.0));
    assert_eq!(v.swizzle::<SWIZZLE_W, SWIZZLE_X, SWIZZLE_Y, SWIZZLE_Z>(), Vec4::new(4.0, 1.0, 2.0, 3.0));
    assert_eq!(v.swizzle::<SWIZZLE_W, SWIZZLE_X, SWIZZLE_Y, SWIZZLE_W>(), Vec4::new(4.0, 1.0, 2.0, 4.0));
    assert_eq!(v.swizzle::<SWIZZLE_W, SWIZZLE_X, SWIZZLE_Z, SWIZZLE_X>(), Vec4::new(4.0, 1.0, 3.0, 1.0));
    assert_eq!(v.swizzle::<SWIZZLE_W, SWIZZLE_X, SWIZZLE_Z, SWIZZLE_Y>(), Vec4::new(4.0, 1.0, 3.0, 2.0));
    assert_eq!(v.swizzle::<SWIZZLE_W, SWIZZLE_X, SWIZZLE_Z, SWIZZLE_Z>(), Vec4::new(4.0, 1.0, 3.0, 3.0));
    assert_eq!(v.swizzle::<SWIZZLE_W, SWIZZLE_X, SWIZZLE_Z, SWIZZLE_W>(), Vec4::new(4.0, 1.0, 3.0, 4.0));
    assert_eq!(v.swizzle::<SWIZZLE_W, SWIZZLE_X, SWIZZLE_W, SWIZZLE_X>(), Vec4::new(4.0, 1.0, 4.0, 1.0));
    assert_eq!(v.swizzle::<SWIZZLE_W, SWIZZLE_X, SWIZZLE_W, SWIZZLE_Y>(), Vec4::new(4.0, 1.0, 4.0, 2.0));
    assert_eq!(v.swizzle::<SWIZZLE_W, SWIZZLE_X, SWIZZLE_W, SWIZZLE_Z>(), Vec4::new(4.0, 1.0, 4.0, 3.0));
    assert_eq!(v.swizzle::<SWIZZLE_W, SWIZZLE_X, SWIZZLE_W, SWIZZLE_W>(), Vec4::new(4.0, 1.0, 4.0, 4.0));
    assert_eq!(v.swizzle::<SWIZZLE_W, SWIZZLE_Y, SWIZZLE_X, SWIZZLE_X>(), Vec4::new(4.0, 2.0, 1.0, 1.0));
    assert_eq!(v.swizzle::<SWIZZLE_W, SWIZZLE_Y, SWIZZLE_X, SWIZZLE_Y>(), Vec4::new(4.0, 2.0, 1.0, 2.0));
    assert_eq!(v.swizzle::<SWIZZLE_W, SWIZZLE_Y, SWIZZLE_X, SWIZZLE_Z>(), Vec4::new(4.0, 2.0, 1.0, 3.0));
    assert_eq!(v.swizzle::<SWIZZLE_W, SWIZZLE_Y, SWIZZLE_X, SWIZZLE_W>(), Vec4::new(4.0, 2.0, 1.0, 4.0));
    assert_eq!(v.swizzle::<SWIZZLE_W, SWIZZLE_Y, SWIZZLE_Y, SWIZZLE_X>(), Vec4::new(4.0, 2.0, 2.0, 1.0));
    assert_eq!(v.swizzle::<SWIZZLE_W, SWIZZLE_Y, SWIZZLE_Y, SWIZZLE_Y>(), Vec4::new(4.0, 2.0, 2.0, 2.0));
    assert_eq!(v.swizzle::<SWIZZLE_W, SWIZZLE_Y, SWIZZLE_Y, SWIZZLE_Z>(), Vec4::new(4.0, 2.0, 2.0, 3.0));
    assert_eq!(v.swizzle::<SWIZZLE_W, SWIZZLE_Y, SWIZZLE_Y, SWIZZLE_W>(), Vec4::new(4.0, 2.0, 2.0, 4.0));
    assert_eq!(v.swizzle::<SWIZZLE_W, SWIZZLE_Y, SWIZZLE_Z, SWIZZLE_X>(), Vec4::new(4.0, 2.0, 3.0, 1.0));
    assert_eq!(v.swizzle::<SWIZZLE_W, SWIZZLE_Y, SWIZZLE_Z, SWIZZLE_Y>(), Vec4::new(4.0, 2.0, 3.0, 2.0));
    assert_eq!(v.swizzle::<SWIZZLE_W, SWIZZLE_Y, SWIZZLE_Z, SWIZZLE_Z>(), Vec4::new(4.0, 2.0, 3.0, 3.0));
    assert_eq!(v.swizzle::<SWIZZLE_W, SWIZZLE_Y, SWIZZLE_Z, SWIZZLE_W>(), Vec4::new(4.0, 2.0, 3.0, 4.0));
    assert_eq!(v.swizzle::<SWIZZLE_W, SWIZZLE_Y, SWIZZLE_W, SWIZZLE_X>(), Vec4::new(4.0, 2.0, 4.0, 1.0));
    assert_eq!(v.swizzle::<SWIZZLE_W, SWIZZLE_Y, SWIZZLE_W, SWIZZLE_Y>(), Vec4::new(4.0, 2.0, 4.0, 2.0));
    assert_eq!(v.swizzle::<SWIZZLE_W, SWIZZLE_Y, SWIZZLE_W, SWIZZLE_Z>(), Vec4::new(4.0, 2.0, 4.0, 3.0));
    assert_eq!(v.swizzle::<SWIZZLE_W, SWIZZLE_Y, SWIZZLE_W, SWIZZLE_W>(), Vec4::new(4.0, 2.0, 4.0, 4.0));
    assert_eq!(v.swizzle::<SWIZZLE_W, SWIZZLE_Z, SWIZZLE_X, SWIZZLE_X>(), Vec4::new(4.0, 3.0, 1.0, 1.0));
    assert_eq!(v.swizzle::<SWIZZLE_W, SWIZZLE_Z, SWIZZLE_X, SWIZZLE_Y>(), Vec4::new(4.0, 3.0, 1.0, 2.0));
    assert_eq!(v.swizzle::<SWIZZLE_W, SWIZZLE_Z, SWIZZLE_X, SWIZZLE_Z>(), Vec4::new(4.0, 3.0, 1.0, 3.0));
    assert_eq!(v.swizzle::<SWIZZLE_W, SWIZZLE_Z, SWIZZLE_X, SWIZZLE_W>(), Vec4::new(4.0, 3.0, 1.0, 4.0));
    assert_eq!(v.swizzle::<SWIZZLE_W, SWIZZLE_Z, SWIZZLE_Y, SWIZZLE_X>(), Vec4::new(4.0, 3.0, 2.0, 1.0));
    assert_eq!(v.swizzle::<SWIZZLE_W, SWIZZLE_Z, SWIZZLE_Y, SWIZZLE_Y>(), Vec4::new(4.0, 3.0, 2.0, 2.0));
    assert_eq!(v.swizzle::<SWIZZLE_W, SWIZZLE_Z, SWIZZLE_Y, SWIZZLE_Z>(), Vec4::new(4.0, 3.0, 2.0, 3.0));
    assert_eq!(v.swizzle::<SWIZZLE_W, SWIZZLE_Z, SWIZZLE_Y, SWIZZLE_W>(), Vec4::new(4.0, 3.0, 2.0, 4.0));
    assert_eq!(v.swizzle::<SWIZZLE_W, SWIZZLE_Z, SWIZZLE_Z, SWIZZLE_X>(), Vec4::new(4.0, 3.0, 3.0, 1.0));
    assert_eq!(v.swizzle::<SWIZZLE_W, SWIZZLE_Z, SWIZZLE_Z, SWIZZLE_Y>(), Vec4::new(4.0, 3.0, 3.0, 2.0));
    assert_eq!(v.swizzle::<SWIZZLE_W, SWIZZLE_Z, SWIZZLE_Z, SWIZZLE_Z>(), Vec4::new(4.0, 3.0, 3.0, 3.0));
    assert_eq!(v.swizzle::<SWIZZLE_W, SWIZZLE_Z, SWIZZLE_Z, SWIZZLE_W>(), Vec4::new(4.0, 3.0, 3.0, 4.0));
    assert_eq!(v.swizzle::<SWIZZLE_W, SWIZZLE_Z, SWIZZLE_W, SWIZZLE_X>(), Vec4::new(4.0, 3.0, 4.0, 1.0));
    assert_eq!(v.swizzle::<SWIZZLE_W, SWIZZLE_Z, SWIZZLE_W, SWIZZLE_Y>(), Vec4::new(4.0, 3.0, 4.0, 2.0));
    assert_eq!(v.swizzle::<SWIZZLE_W, SWIZZLE_Z, SWIZZLE_W, SWIZZLE_Z>(), Vec4::new(4.0, 3.0, 4.0, 3.0));
    assert_eq!(v.swizzle::<SWIZZLE_W, SWIZZLE_Z, SWIZZLE_W, SWIZZLE_W>(), Vec4::new(4.0, 3.0, 4.0, 4.0));
    assert_eq!(v.swizzle::<SWIZZLE_W, SWIZZLE_W, SWIZZLE_X, SWIZZLE_X>(), Vec4::new(4.0, 4.0, 1.0, 1.0));
    assert_eq!(v.swizzle::<SWIZZLE_W, SWIZZLE_W, SWIZZLE_X, SWIZZLE_Y>(), Vec4::new(4.0, 4.0, 1.0, 2.0));
    assert_eq!(v.swizzle::<SWIZZLE_W, SWIZZLE_W, SWIZZLE_X, SWIZZLE_Z>(), Vec4::new(4.0, 4.0, 1.0, 3.0));
    assert_eq!(v.swizzle::<SWIZZLE_W, SWIZZLE_W, SWIZZLE_X, SWIZZLE_W>(), Vec4::new(4.0, 4.0, 1.0, 4.0));
    assert_eq!(v.swizzle::<SWIZZLE_W, SWIZZLE_W, SWIZZLE_Y, SWIZZLE_X>(), Vec4::new(4.0, 4.0, 2.0, 1.0));
    assert_eq!(v.swizzle::<SWIZZLE_W, SWIZZLE_W, SWIZZLE_Y, SWIZZLE_Y>(), Vec4::new(4.0, 4.0, 2.0, 2.0));
    assert_eq!(v.swizzle::<SWIZZLE_W, SWIZZLE_W, SWIZZLE_Y, SWIZZLE_Z>(), Vec4::new(4.0, 4.0, 2.0, 3.0));
    assert_eq!(v.swizzle::<SWIZZLE_W, SWIZZLE_W, SWIZZLE_Y, SWIZZLE_W>(), Vec4::new(4.0, 4.0, 2.0, 4.0));
    assert_eq!(v.swizzle::<SWIZZLE_W, SWIZZLE_W, SWIZZLE_Z, SWIZZLE_X>(), Vec4::new(4.0, 4.0, 3.0, 1.0));
    assert_eq!(v.swizzle::<SWIZZLE_W, SWIZZLE_W, SWIZZLE_Z, SWIZZLE_Y>(), Vec4::new(4.0, 4.0, 3.0, 2.0));
    assert_eq!(v.swizzle::<SWIZZLE_W, SWIZZLE_W, SWIZZLE_Z, SWIZZLE_Z>(), Vec4::new(4.0, 4.0, 3.0, 3.0));
    assert_eq!(v.swizzle::<SWIZZLE_W, SWIZZLE_W, SWIZZLE_Z, SWIZZLE_W>(), Vec4::new(4.0, 4.0, 3.0, 4.0));
    assert_eq!(v.swizzle::<SWIZZLE_W, SWIZZLE_W, SWIZZLE_W, SWIZZLE_X>(), Vec4::new(4.0, 4.0, 4.0, 1.0));
    assert_eq!(v.swizzle::<SWIZZLE_W, SWIZZLE_W, SWIZZLE_W, SWIZZLE_Y>(), Vec4::new(4.0, 4.0, 4.0, 2.0));
    assert_eq!(v.swizzle::<SWIZZLE_W, SWIZZLE_W, SWIZZLE_W, SWIZZLE_Z>(), Vec4::new(4.0, 4.0, 4.0, 3.0));
    assert_eq!(v.swizzle::<SWIZZLE_W, SWIZZLE_W, SWIZZLE_W, SWIZZLE_W>(), Vec4::new(4.0, 4.0, 4.0, 4.0));
}

#[test]
fn test_vec4_abs() {
    assert_eq!(Vec4::new(1.0, -2.0, 3.0, -4.0).abs(), Vec4::new(1.0, 2.0, 3.0, 4.0));
    assert_eq!(Vec4::new(-1.0, 2.0, -3.0, 4.0).abs(), Vec4::new(1.0, 2.0, 3.0, 4.0));
}

#[test]
fn test_vec4_dot() {
    assert_eq!(
        Vec4::new(1.0, 2.0, 3.0, 4.0).dot(Vec4::new(5.0, 6.0, 7.0, 8.0)),
        (1 * 5 + 2 * 6 + 3 * 7 + 4 * 8) as f32
    );
    assert_eq!(
        Vec4::new(1.0, 2.0, 3.0, 4.0).dot_v(Vec4::new(5.0, 6.0, 7.0, 8.0)),
        Vec4::replicate((1 * 5 + 2 * 6 + 3 * 7 + 4 * 8) as f32)
    );

    // Test handling of -0.0f
    for (i, v) in signed_zeros4().enumerate() {
        assert_eq!(v.dot(Vec4::replicate(1.0)).to_bits(), if i == 0 { 0x80000000 } else { 0 }, "{v:?}");
    }
}

#[test]
fn test_vec4_length() {
    assert_eq!(Vec4::new(1.0, 2.0, 3.0, 4.0).length_sq(), (1 + 4 + 9 + 16) as f32);
    assert_eq!(Vec4::new(1.0, 2.0, 3.0, 4.0).length(), sqrt((1 + 4 + 9 + 16) as f32));
}

#[test]
fn test_vec4_sqrt() {
    check_approx_equal(
        Vec4::new(13.0, 15.0, 17.0, 19.0).sqrt(),
        Vec4::new(sqrt(13.0), sqrt(15.0), sqrt(17.0), sqrt(19.0)),
    );
}

#[test]
fn test_vec4_normalize() {
    assert_eq!(Vec4::new(1.0, 2.0, 3.0, 4.0).normalized(), Vec4::new(1.0, 2.0, 3.0, 4.0) / sqrt(30.0f32));
}

#[test]
fn test_vec4_cast() {
    assert_eq!(Vec4::new(1.0, 2.0, 3.0, 4.0).to_int(), UVec4::new(1, 2, 3, 4));
    assert_eq!(
        Vec4::new(1.0, 2.0, 3.0, 4.0).reinterpret_as_int(),
        UVec4::new(0x3f800000, 0x40000000, 0x40400000, 0x40800000)
    );
}

#[test]
fn test_vec4_sign() {
    assert_eq!(Vec4::new(1.2345, -6.7891, 0.0, 1.0).sign(), Vec4::new(1.0, -1.0, 1.0, 1.0));
    assert_eq!(Vec4::new(0.0, 2.3456, -7.8912, -1.0).sign(), Vec4::new(1.0, 1.0, -1.0, -1.0));
    assert_eq!(Vec4::new(-0.0, 0.0, -0.0, 0.0).sign(), Vec4::new(-1.0, 1.0, -1.0, 1.0));
    assert_eq!(Vec4::new(1.0, -1.0, 1.0, -1.0).sign(), Vec4::new(1.0, -1.0, 1.0, -1.0));
    let true_min = f32::from_bits(1); // FLT_TRUE_MIN
    assert_eq!(Vec4::new(true_min, -true_min, true_min, -true_min).sign(), Vec4::new(1.0, -1.0, 1.0, -1.0)); // Denormal number
    assert_eq!(
        Vec4::new(f32::INFINITY, -f32::INFINITY, f32::INFINITY, -f32::INFINITY).sign(),
        Vec4::new(1.0, -1.0, 1.0, -1.0)
    );
}

#[test]
fn test_vec4_flip_sign() {
    let v = Vec4::new(1.0, 2.0, 3.0, 4.0);
    assert_eq!(v.flip_sign::<-1, 1, 1, 1>(), Vec4::new(-1.0, 2.0, 3.0, 4.0));
    assert_eq!(v.flip_sign::<1, -1, 1, 1>(), Vec4::new(1.0, -2.0, 3.0, 4.0));
    assert_eq!(v.flip_sign::<1, 1, -1, 1>(), Vec4::new(1.0, 2.0, -3.0, 4.0));
    assert_eq!(v.flip_sign::<1, 1, 1, -1>(), Vec4::new(1.0, 2.0, 3.0, -4.0));
}

#[test]
fn test_vec4_sign_bit() {
    assert_eq!(Vec4::new(2.0, -3.0, 4.0, -5.0).sign_bits(), 0b1010);
    assert_eq!(Vec4::new(-2.0, 3.0, -4.0, 5.0).sign_bits(), 0b0101);
}

#[test]
fn test_vec4_sort() {
    for i in 0..4u32 {
        for j in (0..4u32).filter(|&j| j != i) {
            for k in (0..4u32).filter(|&k| k != i && k != j) {
                for l in (0..4u32).filter(|&l| l != i && l != j && l != k) {
                    let mut v1 = Vec4::new(i as f32, j as f32, k as f32, l as f32);
                    let mut v2 = v1;
                    let mut idx1 = UVec4::new(i + 4, j + 4, k + 4, l + 4);
                    let mut idx2 = idx1;
                    Vec4::sort4(&mut v1, &mut idx1);
                    Vec4::sort4_reverse(&mut v2, &mut idx2);
                    for m in 0..4 {
                        assert_eq!(v1[m], m as f32);
                        assert_eq!(v2[m], (3 - m) as f32);
                        assert_eq!(idx1[m], (m + 4) as u32);
                        assert_eq!(idx2[m], (3 - m + 4) as u32);
                    }
                }
            }
        }
    }
}

/// `for (float x = start; x < end; x += step)` (accumulating in float like the C++ loop)
fn float_range(start: f32, end: f32, step: f32, inclusive: bool) -> impl Iterator<Item = f32> {
    std::iter::successors(Some(start), move |x| Some(x + step))
        .take_while(move |&x| if inclusive { x <= end } else { x < end })
}

#[test]
fn test_vec4_sin_cos() {
    // Check edge cases
    let (vs, vc) = Vec4::new(0.0, 0.5 * JPH_PI, JPH_PI, -0.5 * JPH_PI).sin_cos();
    assert!(vs.is_close(Vec4::new(0.0, 1.0, 0.0, -1.0), 1.0e-7));
    assert!(vc.is_close(Vec4::new(1.0, 0.0, -1.0, 0.0), 1.0e-7));

    let mut ms = 0.0f64;
    let mut mc = 0.0f64;

    for x in float_range(-100.0 * JPH_PI, 100.0 * JPH_PI, 1.0e-3, false) {
        // Create a vector with intermediate values
        let xv = Vec4::replicate(x) + Vec4::new(0.0e-4, 2.5e-4, 5.0e-4, 7.5e-4);

        // Calculate sin and cos
        let (vs, vc) = xv.sin_cos();

        for i in 0..4 {
            // Check accuracy of sin
            let s1 = (xv[i] as f64).sin();
            let s2 = vs[i] as f64;
            let ds = (s2 - s1).abs();
            ms = ms.max(ds);

            // Check accuracy of cos
            let c1 = (xv[i] as f64).cos();
            let c2 = vc[i] as f64;
            let dc = (c2 - c1).abs();
            mc = mc.max(dc);
        }
    }

    assert!(ms < 1.0e-7, "{ms}");
    assert!(mc < 1.0e-7, "{mc}");
}

#[test]
fn test_vec4_tan() {
    // Check edge cases
    assert_eq!(Vec4::replicate(0.0).tan(), Vec4::zero());
    assert!(Vec4::replicate(0.5 * JPH_PI - 1.0e-6).tan().x() > 1.0e6);
    assert!(Vec4::replicate(0.5 * JPH_PI + 1.0e-6).tan().x() < -1.0e6);

    let mut mt = 0.0f64;

    for x in float_range(-100.0 * JPH_PI, 100.0 * JPH_PI, 1.0e-3, false) {
        // Create a vector with intermediate values
        let xv = Vec4::replicate(x) + Vec4::new(0.0e-4, 2.5e-4, 5.0e-4, 7.5e-4);

        // Calculate tan
        let vt = xv.tan();

        for i in 0..4 {
            // Check accuracy of tan
            let t1 = (xv[i] as f64).tan();
            let t2 = vt[i] as f64;
            let dt = (t2 - t1).abs();
            mt = mt.max(dt) / 1.0f64.max(t1.abs()); // Take relative error
        }
    }

    assert!(mt < 1.5e-7, "{mt}");
}

#[test]
fn test_vec4_a_sin() {
    // Check edge cases
    assert_eq!(Vec4::replicate(0.0).asin(), Vec4::zero());
    assert_eq!(Vec4::one().asin(), Vec4::replicate(0.5 * JPH_PI));
    assert_eq!(Vec4::replicate(-1.0).asin(), Vec4::replicate(-0.5 * JPH_PI));

    let mut ma = 0.0f64;

    for x in float_range(-1.0, 1.0, 1.0e-3, true) {
        // Create a vector with intermediate values
        let xv = Vec4::min(Vec4::replicate(x) + Vec4::new(0.0e-4, 2.5e-4, 5.0e-4, 7.5e-4), Vec4::one());

        // Calculate asin
        let va = xv.asin();

        for i in 0..4 {
            // Check accuracy of asin
            let a1 = (xv[i] as f64).asin();
            let a2 = va[i] as f64;
            let da = (a2 - a1).abs();
            ma = ma.max(da);
        }
    }

    assert!(ma < 2.0e-7, "{ma}");

    // Check that inputs are clamped as promised
    assert_eq!(Vec4::replicate(-1.1).asin(), Vec4::replicate(-0.5 * JPH_PI));
    assert_eq!(Vec4::replicate(1.1).asin(), Vec4::replicate(0.5 * JPH_PI));
}

#[test]
fn test_vec4_a_cos() {
    // Check edge cases
    assert_eq!(Vec4::replicate(0.0).acos(), Vec4::replicate(0.5 * JPH_PI));
    assert_eq!(Vec4::one().acos(), Vec4::zero());
    assert_eq!(Vec4::replicate(-1.0).acos(), Vec4::replicate(JPH_PI));

    let mut ma = 0.0f64;

    for x in float_range(-1.0, 1.0, 1.0e-3, true) {
        // Create a vector with intermediate values
        let xv = Vec4::min(Vec4::replicate(x) + Vec4::new(0.0e-4, 2.5e-4, 5.0e-4, 7.5e-4), Vec4::one());

        // Calculate acos
        let va = xv.acos();

        for i in 0..4 {
            // Check accuracy of acos
            let a1 = (xv[i] as f64).acos();
            let a2 = va[i] as f64;
            let da = (a2 - a1).abs();
            ma = ma.max(da);
        }
    }

    assert!(ma < 3.5e-7, "{ma}");

    // Check that inputs are clamped as promised
    assert_eq!(Vec4::replicate(-1.1).acos(), Vec4::replicate(JPH_PI));
    assert_eq!(Vec4::replicate(1.1).acos(), Vec4::zero());
}

#[test]
fn test_vec4_a_tan() {
    // Check edge cases
    assert_eq!(Vec4::replicate(0.0).atan(), Vec4::zero());
    assert_eq!(Vec4::replicate(f32::MAX).atan(), Vec4::replicate(0.5 * JPH_PI));
    assert_eq!(Vec4::replicate(-f32::MAX).atan(), Vec4::replicate(-0.5 * JPH_PI));

    let mut ma = 0.0f64;

    for x in float_range(-100.0, 100.0, 1.0e-3, false) {
        // Create a vector with intermediate values
        let xv = Vec4::replicate(x) + Vec4::new(0.0e-4, 2.5e-4, 5.0e-4, 7.5e-4);

        // Calculate atan
        let va = xv.atan();

        for i in 0..4 {
            // Check accuracy of atan
            let a1 = (xv[i] as f64).atan();
            let a2 = va[i] as f64;
            let da = (a2 - a1).abs();
            ma = ma.max(da);
        }
    }

    assert!(ma < 1.5e-7, "{ma}");
}

#[test]
fn test_vec4_a_tan2() {
    let mut ma = 0.0f64;

    // Test the axis
    assert_eq!(Vec4::atan2(Vec4::zero(), Vec4::replicate(10.0)), Vec4::zero());
    assert_eq!(Vec4::atan2(Vec4::zero(), Vec4::replicate(-10.0)), Vec4::replicate(JPH_PI));
    assert_eq!(Vec4::atan2(Vec4::replicate(10.0), Vec4::zero()), Vec4::replicate(0.5 * JPH_PI));
    assert_eq!(Vec4::atan2(Vec4::replicate(-10.0), Vec4::zero()), Vec4::replicate(-0.5 * JPH_PI));

    // Test the 4 quadrants
    assert_eq!(Vec4::atan2(Vec4::replicate(10.0), Vec4::replicate(10.0)), Vec4::replicate(0.25 * JPH_PI));
    assert_eq!(Vec4::atan2(Vec4::replicate(10.0), Vec4::replicate(-10.0)), Vec4::replicate(0.75 * JPH_PI));
    assert_eq!(Vec4::atan2(Vec4::replicate(-10.0), Vec4::replicate(-10.0)), Vec4::replicate(-0.75 * JPH_PI));
    assert_eq!(Vec4::atan2(Vec4::replicate(-10.0), Vec4::replicate(10.0)), Vec4::replicate(-0.25 * JPH_PI));

    for y in float_range(-5.0, 5.0, 1.0e-2, false) {
        // Create a vector with intermediate values
        let yv = Vec4::replicate(y) + Vec4::new(0.0e-3, 2.5e-3, 5.0e-3, 7.5e-3);

        for x in float_range(-5.0, 5.0, 1.0e-2, false) {
            // Create a vector with intermediate values
            let xv = Vec4::replicate(x) + Vec4::new(0.0e-3, 2.5e-3, 5.0e-3, 7.5e-3);

            // Calculate atan
            let va = Vec4::atan2(yv, xv);

            for i in 0..4 {
                // Check accuracy of atan
                let a1 = (yv[i] as f64).atan2(xv[i] as f64);
                let a2 = va[i] as f64;
                let da = (a2 - a1).abs();
                ma = ma.max(da);
            }
        }
    }

    assert!(ma < 3.0e-7, "{ma}");
}

#[test]
fn test_vec4_convert_to_string() {
    let v = Vec4::new(1.0, 2.0, 3.0, 4.0);
    assert_eq!(v.to_string(), "1, 2, 3, 4");
}

#[test]
fn test_vec4_compress_unit_vector() {
    // We want these to be preserved exactly
    for v in [
        Vec4::new(1.0, 0.0, 0.0, 0.0),
        Vec4::new(0.0, 1.0, 0.0, 0.0),
        Vec4::new(0.0, 0.0, 1.0, 0.0),
        Vec4::new(0.0, 0.0, 0.0, 1.0),
        Vec4::new(-1.0, 0.0, 0.0, 0.0),
        Vec4::new(0.0, -1.0, 0.0, 0.0),
        Vec4::new(0.0, 0.0, -1.0, 0.0),
        Vec4::new(0.0, 0.0, 0.0, -1.0),
    ] {
        assert_eq!(Vec4::decompress_unit_vector(v.compress_unit_vector()), v);
    }

    let mut random = UnitTestRandom::default();
    for _ in 0..1000 {
        // C++ evaluates the four scale(random) arguments in an unspecified order, here they are drawn x, y, z, w
        let x = random.uniform_f32(-1.0, 1.0);
        let y = random.uniform_f32(-1.0, 1.0);
        let z = random.uniform_f32(-1.0, 1.0);
        let w = random.uniform_f32(-1.0, 1.0);
        let v = Vec4::new(x, y, z, w).normalized();
        let compressed = v.compress_unit_vector();
        let decompressed = Vec4::decompress_unit_vector(compressed);
        let diff = (decompressed - v).length();
        assert!(diff < 5.0e-3, "{v:?} {decompressed:?} {diff}");
    }
}

#[test]
fn test_difference_of_products() {
    let a = Vec4::new(33962.035, 33962.0351, 33962.0352, 33962.0353);
    let b = Vec4::new(-30438.8, -30438.801, -30438.802, -30438.803);
    let c = Vec4::new(41563.4, 41563.401, 41563.402, 41563.403);
    let d = Vec4::new(-24871.969, -24871.970, -24871.971, -24871.972);
    let result = Vec4::difference_of_products(a, b, c, d);
    let expected: [f64; 4] = std::array::from_fn(|i| a[i] as f64 * b[i] as f64 - c[i] as f64 * d[i] as f64);
    assert_eq!(expected[0], -75.165603637695312);
    assert_eq!(expected[1], 103.16904449462891);
    assert_eq!(expected[2], 36.836944580078125);
    assert_eq!(expected[3], 118.01546478271484);
    // The products are in the order of 10^9, so the subtraction causes a large loss of precision and we get a very different result. This is expected when fused multiply add instructions are not available.
    assert_eq!(result, Vec4::new(-128.0, 64.0, 0.0, 64.0));
}
