//! Port of UnitTests/Math/UVec4Tests.cpp
//!
//! The 16 explicit `GetTrues` / `CountTrues` / `Test*True` checks per function of `TestUVec4Comparisons` are written
//! as a loop over the same 16 bit patterns.

// The expected values are written like in the C++ test (e.g. `1 * 4 + 7`)
#![allow(clippy::identity_op)]

use jolt_core::math::*;

#[test]
fn test_u_vec4_construct() {
    let v = UVec4::new(1, 2, 3, 4);

    assert_eq!(v.x(), 1);
    assert_eq!(v.y(), 2);
    assert_eq!(v.z(), 3);
    assert_eq!(v.w(), 4);

    assert_eq!(v[0], 1);
    assert_eq!(v[1], 2);
    assert_eq!(v[2], 3);
    assert_eq!(v[3], 4);

    // Test == and != operators
    assert!(v == UVec4::new(1, 2, 3, 4));
    assert!(v != UVec4::new(1, 2, 4, 3));
}

#[test]
fn test_u_vec4_components() {
    let mut v = UVec4::new(1, 2, 3, 4);
    v.set_x(5);
    v.set_y(6);
    v.set_z(7);
    v.set_w(8);
    assert_eq!(v, UVec4::new(5, 6, 7, 8));
}

#[test]
fn test_u_vec4_load_store_int4() {
    let i4: [u32; 4] = [1, 2, 3, 4];
    assert_eq!(UVec4::load_int(&i4), UVec4::new(1, 0, 0, 0));
    assert_eq!(UVec4::load_int4(&i4), UVec4::new(1, 2, 3, 4));
    assert_eq!(UVec4::load_int4_aligned(&i4), UVec4::new(1, 2, 3, 4));

    let i4_out1 = UVec4::new(1, 2, 3, 4).store_int4();
    assert_eq!(i4_out1[0], 1);
    assert_eq!(i4_out1[1], 2);
    assert_eq!(i4_out1[2], 3);
    assert_eq!(i4_out1[3], 4);

    let i4_out2 = UVec4::new(1, 2, 3, 4).store_int4_aligned();
    assert_eq!(i4_out2[0], 1);
    assert_eq!(i4_out2[1], 2);
    assert_eq!(i4_out2[2], 3);
    assert_eq!(i4_out2[3], 4);

    let si: [u32; 20] = [0, 0, 1, 0, 0, 0, 2, 0, 0, 0, 0, 0, 0, 0, 0, 0, 3, 0, 4, 0];
    assert_eq!(UVec4::gather_int4::<{ 2 * size_of::<u32>() }>(&si, UVec4::new(1, 3, 8, 9)), UVec4::new(1, 2, 3, 4));
}

#[test]
fn test_u_vec4_zero() {
    let v = UVec4::zero();

    assert_eq!(v.x(), 0);
    assert_eq!(v.y(), 0);
    assert_eq!(v.z(), 0);
    assert_eq!(v.w(), 0);
}

#[test]
fn test_u_vec4_replicate() {
    assert_eq!(UVec4::replicate(2), UVec4::new(2, 2, 2, 2));
}

#[test]
fn test_u_vec4_min_max() {
    let v1 = UVec4::new(1, 6, 3, 8);
    let v2 = UVec4::new(5, 2, 7, 4);

    assert_eq!(UVec4::min(v1, v2), UVec4::new(1, 2, 3, 4));
    assert_eq!(UVec4::max(v1, v2), UVec4::new(5, 6, 7, 8));
}

#[test]
fn test_u_vec4_comparisons() {
    assert_eq!(UVec4::equals(UVec4::new(1, 2, 3, 4), UVec4::new(2, 1, 3, 4)), UVec4::new(0, 0, 0xffffffff, 0xffffffff));

    for bits in 0..16i32 {
        let c = |i: i32| if bits & (1 << i) != 0 { 0xffffffff } else { 0 };
        let v = UVec4::new(c(0), c(1), c(2), c(3));
        assert_eq!(v.trues(), bits, "{v}");
        assert_eq!(v.count_trues(), bits.count_ones() as i32, "{v}");
        assert_eq!(v.test_all_true(), bits == 0b1111, "{v}");
        assert_eq!(v.test_all_xyz_true(), (bits & 0b111) == 0b111, "{v}");
        assert_eq!(v.test_any_true(), bits != 0, "{v}");
        assert_eq!(v.test_any_xyz_true(), (bits & 0b111) != 0, "{v}");
    }
}

#[test]
fn test_u_vec4_select() {
    let a = UVec4::new(1, 2, 3, 4);
    let b = UVec4::new(5, 6, 7, 8);
    assert_eq!(UVec4::select(a, b, UVec4::new(0x80000000, 0, 0x80000000, 0)), UVec4::new(5, 2, 7, 4));
    assert_eq!(UVec4::select(a, b, UVec4::new(0, 0x80000000, 0, 0x80000000)), UVec4::new(1, 6, 3, 8));
    assert_eq!(UVec4::select(a, b, UVec4::new(0xffffffff, 0x7fffffff, 0xffffffff, 0x7fffffff)), UVec4::new(5, 2, 7, 4));
    assert_eq!(UVec4::select(a, b, UVec4::new(0x7fffffff, 0xffffffff, 0x7fffffff, 0xffffffff)), UVec4::new(1, 6, 3, 8));
}

#[test]
fn test_u_vec4_bit_ops() {
    // Test all bit permutations
    let v1 = UVec4::new(0b0011, 0b00110, 0b001100, 0b0011000);
    let v2 = UVec4::new(0b0101, 0b01010, 0b010100, 0b0101000);

    assert_eq!(UVec4::or(v1, v2), UVec4::new(0b0111, 0b01110, 0b011100, 0b0111000));
    assert_eq!(UVec4::xor(v1, v2), UVec4::new(0b0110, 0b01100, 0b011000, 0b0110000));
    assert_eq!(UVec4::and(v1, v2), UVec4::new(0b0001, 0b00010, 0b000100, 0b0001000));

    assert_eq!(UVec4::not(v1), UVec4::new(0xfffffffc, 0xfffffff9, 0xfffffff3, 0xffffffe7));
    assert_eq!(UVec4::not(v2), UVec4::new(0xfffffffa, 0xfffffff5, 0xffffffeb, 0xffffffd7));

    assert_eq!(
        UVec4::new(0x80000000, 0x40000000, 0x20000000, 0x10000000).logical_shift_right::<1>(),
        UVec4::new(0x40000000, 0x20000000, 0x10000000, 0x08000000)
    );
    assert_eq!(
        UVec4::new(0x80000000, 0x40000000, 0x20000000, 0x10000000).arithmetic_shift_right::<1>(),
        UVec4::new(0xC0000000, 0x20000000, 0x10000000, 0x08000000)
    );
    assert_eq!(
        UVec4::new(0x40000000, 0x20000000, 0x10000000, 0x08000001).logical_shift_left::<1>(),
        UVec4::new(0x80000000, 0x40000000, 0x20000000, 0x10000002)
    );
}

#[test]
fn test_u_vec4_operators() {
    assert_eq!(UVec4::new(1, 2, 3, 4) + UVec4::new(5, 6, 7, 8), UVec4::new(6, 8, 10, 12));
    assert_eq!(UVec4::new(5, 6, 7, 8) - UVec4::new(4, 3, 2, 1), UVec4::new(1, 3, 5, 7));
    assert_eq!(UVec4::new(1, 2, 3, 4) * UVec4::new(5, 6, 7, 8), UVec4::new(1 * 5, 2 * 6, 3 * 7, 4 * 8));

    let mut v = UVec4::new(1, 2, 3, 4);
    v += UVec4::new(5, 6, 7, 8);
    assert_eq!(v, UVec4::new(6, 8, 10, 12));
    v -= UVec4::new(4, 3, 2, 1);
    assert_eq!(v, UVec4::new(2, 5, 8, 11));
}

#[test]
fn test_u_vec4_swizzle() {
    let v = UVec4::new(1, 2, 3, 4);

    assert_eq!(v.splat_x(), UVec4::replicate(1));
    assert_eq!(v.splat_y(), UVec4::replicate(2));
    assert_eq!(v.splat_z(), UVec4::replicate(3));
    assert_eq!(v.splat_w(), UVec4::replicate(4));

    assert_eq!(v.swizzle::<SWIZZLE_X, SWIZZLE_X, SWIZZLE_X, SWIZZLE_X>(), UVec4::new(1, 1, 1, 1));
    assert_eq!(v.swizzle::<SWIZZLE_X, SWIZZLE_X, SWIZZLE_X, SWIZZLE_Y>(), UVec4::new(1, 1, 1, 2));
    assert_eq!(v.swizzle::<SWIZZLE_X, SWIZZLE_X, SWIZZLE_X, SWIZZLE_Z>(), UVec4::new(1, 1, 1, 3));
    assert_eq!(v.swizzle::<SWIZZLE_X, SWIZZLE_X, SWIZZLE_X, SWIZZLE_W>(), UVec4::new(1, 1, 1, 4));
    assert_eq!(v.swizzle::<SWIZZLE_X, SWIZZLE_X, SWIZZLE_Y, SWIZZLE_X>(), UVec4::new(1, 1, 2, 1));
    assert_eq!(v.swizzle::<SWIZZLE_X, SWIZZLE_X, SWIZZLE_Y, SWIZZLE_Y>(), UVec4::new(1, 1, 2, 2));
    assert_eq!(v.swizzle::<SWIZZLE_X, SWIZZLE_X, SWIZZLE_Y, SWIZZLE_Z>(), UVec4::new(1, 1, 2, 3));
    assert_eq!(v.swizzle::<SWIZZLE_X, SWIZZLE_X, SWIZZLE_Y, SWIZZLE_W>(), UVec4::new(1, 1, 2, 4));
    assert_eq!(v.swizzle::<SWIZZLE_X, SWIZZLE_X, SWIZZLE_Z, SWIZZLE_X>(), UVec4::new(1, 1, 3, 1));
    assert_eq!(v.swizzle::<SWIZZLE_X, SWIZZLE_X, SWIZZLE_Z, SWIZZLE_Y>(), UVec4::new(1, 1, 3, 2));
    assert_eq!(v.swizzle::<SWIZZLE_X, SWIZZLE_X, SWIZZLE_Z, SWIZZLE_Z>(), UVec4::new(1, 1, 3, 3));
    assert_eq!(v.swizzle::<SWIZZLE_X, SWIZZLE_X, SWIZZLE_Z, SWIZZLE_W>(), UVec4::new(1, 1, 3, 4));
    assert_eq!(v.swizzle::<SWIZZLE_X, SWIZZLE_X, SWIZZLE_W, SWIZZLE_X>(), UVec4::new(1, 1, 4, 1));
    assert_eq!(v.swizzle::<SWIZZLE_X, SWIZZLE_X, SWIZZLE_W, SWIZZLE_Y>(), UVec4::new(1, 1, 4, 2));
    assert_eq!(v.swizzle::<SWIZZLE_X, SWIZZLE_X, SWIZZLE_W, SWIZZLE_Z>(), UVec4::new(1, 1, 4, 3));
    assert_eq!(v.swizzle::<SWIZZLE_X, SWIZZLE_X, SWIZZLE_W, SWIZZLE_W>(), UVec4::new(1, 1, 4, 4));
    assert_eq!(v.swizzle::<SWIZZLE_X, SWIZZLE_Y, SWIZZLE_X, SWIZZLE_X>(), UVec4::new(1, 2, 1, 1));
    assert_eq!(v.swizzle::<SWIZZLE_X, SWIZZLE_Y, SWIZZLE_X, SWIZZLE_Y>(), UVec4::new(1, 2, 1, 2));
    assert_eq!(v.swizzle::<SWIZZLE_X, SWIZZLE_Y, SWIZZLE_X, SWIZZLE_Z>(), UVec4::new(1, 2, 1, 3));
    assert_eq!(v.swizzle::<SWIZZLE_X, SWIZZLE_Y, SWIZZLE_X, SWIZZLE_W>(), UVec4::new(1, 2, 1, 4));
    assert_eq!(v.swizzle::<SWIZZLE_X, SWIZZLE_Y, SWIZZLE_Y, SWIZZLE_X>(), UVec4::new(1, 2, 2, 1));
    assert_eq!(v.swizzle::<SWIZZLE_X, SWIZZLE_Y, SWIZZLE_Y, SWIZZLE_Y>(), UVec4::new(1, 2, 2, 2));
    assert_eq!(v.swizzle::<SWIZZLE_X, SWIZZLE_Y, SWIZZLE_Y, SWIZZLE_Z>(), UVec4::new(1, 2, 2, 3));
    assert_eq!(v.swizzle::<SWIZZLE_X, SWIZZLE_Y, SWIZZLE_Y, SWIZZLE_W>(), UVec4::new(1, 2, 2, 4));
    assert_eq!(v.swizzle::<SWIZZLE_X, SWIZZLE_Y, SWIZZLE_Z, SWIZZLE_X>(), UVec4::new(1, 2, 3, 1));
    assert_eq!(v.swizzle::<SWIZZLE_X, SWIZZLE_Y, SWIZZLE_Z, SWIZZLE_Y>(), UVec4::new(1, 2, 3, 2));
    assert_eq!(v.swizzle::<SWIZZLE_X, SWIZZLE_Y, SWIZZLE_Z, SWIZZLE_Z>(), UVec4::new(1, 2, 3, 3));
    assert_eq!(v.swizzle::<SWIZZLE_X, SWIZZLE_Y, SWIZZLE_Z, SWIZZLE_W>(), UVec4::new(1, 2, 3, 4));
    assert_eq!(v.swizzle::<SWIZZLE_X, SWIZZLE_Y, SWIZZLE_W, SWIZZLE_X>(), UVec4::new(1, 2, 4, 1));
    assert_eq!(v.swizzle::<SWIZZLE_X, SWIZZLE_Y, SWIZZLE_W, SWIZZLE_Y>(), UVec4::new(1, 2, 4, 2));
    assert_eq!(v.swizzle::<SWIZZLE_X, SWIZZLE_Y, SWIZZLE_W, SWIZZLE_Z>(), UVec4::new(1, 2, 4, 3));
    assert_eq!(v.swizzle::<SWIZZLE_X, SWIZZLE_Y, SWIZZLE_W, SWIZZLE_W>(), UVec4::new(1, 2, 4, 4));
    assert_eq!(v.swizzle::<SWIZZLE_X, SWIZZLE_Z, SWIZZLE_X, SWIZZLE_X>(), UVec4::new(1, 3, 1, 1));
    assert_eq!(v.swizzle::<SWIZZLE_X, SWIZZLE_Z, SWIZZLE_X, SWIZZLE_Y>(), UVec4::new(1, 3, 1, 2));
    assert_eq!(v.swizzle::<SWIZZLE_X, SWIZZLE_Z, SWIZZLE_X, SWIZZLE_Z>(), UVec4::new(1, 3, 1, 3));
    assert_eq!(v.swizzle::<SWIZZLE_X, SWIZZLE_Z, SWIZZLE_X, SWIZZLE_W>(), UVec4::new(1, 3, 1, 4));
    assert_eq!(v.swizzle::<SWIZZLE_X, SWIZZLE_Z, SWIZZLE_Y, SWIZZLE_X>(), UVec4::new(1, 3, 2, 1));
    assert_eq!(v.swizzle::<SWIZZLE_X, SWIZZLE_Z, SWIZZLE_Y, SWIZZLE_Y>(), UVec4::new(1, 3, 2, 2));
    assert_eq!(v.swizzle::<SWIZZLE_X, SWIZZLE_Z, SWIZZLE_Y, SWIZZLE_Z>(), UVec4::new(1, 3, 2, 3));
    assert_eq!(v.swizzle::<SWIZZLE_X, SWIZZLE_Z, SWIZZLE_Y, SWIZZLE_W>(), UVec4::new(1, 3, 2, 4));
    assert_eq!(v.swizzle::<SWIZZLE_X, SWIZZLE_Z, SWIZZLE_Z, SWIZZLE_X>(), UVec4::new(1, 3, 3, 1));
    assert_eq!(v.swizzle::<SWIZZLE_X, SWIZZLE_Z, SWIZZLE_Z, SWIZZLE_Y>(), UVec4::new(1, 3, 3, 2));
    assert_eq!(v.swizzle::<SWIZZLE_X, SWIZZLE_Z, SWIZZLE_Z, SWIZZLE_Z>(), UVec4::new(1, 3, 3, 3));
    assert_eq!(v.swizzle::<SWIZZLE_X, SWIZZLE_Z, SWIZZLE_Z, SWIZZLE_W>(), UVec4::new(1, 3, 3, 4));
    assert_eq!(v.swizzle::<SWIZZLE_X, SWIZZLE_Z, SWIZZLE_W, SWIZZLE_X>(), UVec4::new(1, 3, 4, 1));
    assert_eq!(v.swizzle::<SWIZZLE_X, SWIZZLE_Z, SWIZZLE_W, SWIZZLE_Y>(), UVec4::new(1, 3, 4, 2));
    assert_eq!(v.swizzle::<SWIZZLE_X, SWIZZLE_Z, SWIZZLE_W, SWIZZLE_Z>(), UVec4::new(1, 3, 4, 3));
    assert_eq!(v.swizzle::<SWIZZLE_X, SWIZZLE_Z, SWIZZLE_W, SWIZZLE_W>(), UVec4::new(1, 3, 4, 4));
    assert_eq!(v.swizzle::<SWIZZLE_X, SWIZZLE_W, SWIZZLE_X, SWIZZLE_X>(), UVec4::new(1, 4, 1, 1));
    assert_eq!(v.swizzle::<SWIZZLE_X, SWIZZLE_W, SWIZZLE_X, SWIZZLE_Y>(), UVec4::new(1, 4, 1, 2));
    assert_eq!(v.swizzle::<SWIZZLE_X, SWIZZLE_W, SWIZZLE_X, SWIZZLE_Z>(), UVec4::new(1, 4, 1, 3));
    assert_eq!(v.swizzle::<SWIZZLE_X, SWIZZLE_W, SWIZZLE_X, SWIZZLE_W>(), UVec4::new(1, 4, 1, 4));
    assert_eq!(v.swizzle::<SWIZZLE_X, SWIZZLE_W, SWIZZLE_Y, SWIZZLE_X>(), UVec4::new(1, 4, 2, 1));
    assert_eq!(v.swizzle::<SWIZZLE_X, SWIZZLE_W, SWIZZLE_Y, SWIZZLE_Y>(), UVec4::new(1, 4, 2, 2));
    assert_eq!(v.swizzle::<SWIZZLE_X, SWIZZLE_W, SWIZZLE_Y, SWIZZLE_Z>(), UVec4::new(1, 4, 2, 3));
    assert_eq!(v.swizzle::<SWIZZLE_X, SWIZZLE_W, SWIZZLE_Y, SWIZZLE_W>(), UVec4::new(1, 4, 2, 4));
    assert_eq!(v.swizzle::<SWIZZLE_X, SWIZZLE_W, SWIZZLE_Z, SWIZZLE_X>(), UVec4::new(1, 4, 3, 1));
    assert_eq!(v.swizzle::<SWIZZLE_X, SWIZZLE_W, SWIZZLE_Z, SWIZZLE_Y>(), UVec4::new(1, 4, 3, 2));
    assert_eq!(v.swizzle::<SWIZZLE_X, SWIZZLE_W, SWIZZLE_Z, SWIZZLE_Z>(), UVec4::new(1, 4, 3, 3));
    assert_eq!(v.swizzle::<SWIZZLE_X, SWIZZLE_W, SWIZZLE_Z, SWIZZLE_W>(), UVec4::new(1, 4, 3, 4));
    assert_eq!(v.swizzle::<SWIZZLE_X, SWIZZLE_W, SWIZZLE_W, SWIZZLE_X>(), UVec4::new(1, 4, 4, 1));
    assert_eq!(v.swizzle::<SWIZZLE_X, SWIZZLE_W, SWIZZLE_W, SWIZZLE_Y>(), UVec4::new(1, 4, 4, 2));
    assert_eq!(v.swizzle::<SWIZZLE_X, SWIZZLE_W, SWIZZLE_W, SWIZZLE_Z>(), UVec4::new(1, 4, 4, 3));
    assert_eq!(v.swizzle::<SWIZZLE_X, SWIZZLE_W, SWIZZLE_W, SWIZZLE_W>(), UVec4::new(1, 4, 4, 4));

    assert_eq!(v.swizzle::<SWIZZLE_Y, SWIZZLE_X, SWIZZLE_X, SWIZZLE_X>(), UVec4::new(2, 1, 1, 1));
    assert_eq!(v.swizzle::<SWIZZLE_Y, SWIZZLE_X, SWIZZLE_X, SWIZZLE_Y>(), UVec4::new(2, 1, 1, 2));
    assert_eq!(v.swizzle::<SWIZZLE_Y, SWIZZLE_X, SWIZZLE_X, SWIZZLE_Z>(), UVec4::new(2, 1, 1, 3));
    assert_eq!(v.swizzle::<SWIZZLE_Y, SWIZZLE_X, SWIZZLE_X, SWIZZLE_W>(), UVec4::new(2, 1, 1, 4));
    assert_eq!(v.swizzle::<SWIZZLE_Y, SWIZZLE_X, SWIZZLE_Y, SWIZZLE_X>(), UVec4::new(2, 1, 2, 1));
    assert_eq!(v.swizzle::<SWIZZLE_Y, SWIZZLE_X, SWIZZLE_Y, SWIZZLE_Y>(), UVec4::new(2, 1, 2, 2));
    assert_eq!(v.swizzle::<SWIZZLE_Y, SWIZZLE_X, SWIZZLE_Y, SWIZZLE_Z>(), UVec4::new(2, 1, 2, 3));
    assert_eq!(v.swizzle::<SWIZZLE_Y, SWIZZLE_X, SWIZZLE_Y, SWIZZLE_W>(), UVec4::new(2, 1, 2, 4));
    assert_eq!(v.swizzle::<SWIZZLE_Y, SWIZZLE_X, SWIZZLE_Z, SWIZZLE_X>(), UVec4::new(2, 1, 3, 1));
    assert_eq!(v.swizzle::<SWIZZLE_Y, SWIZZLE_X, SWIZZLE_Z, SWIZZLE_Y>(), UVec4::new(2, 1, 3, 2));
    assert_eq!(v.swizzle::<SWIZZLE_Y, SWIZZLE_X, SWIZZLE_Z, SWIZZLE_Z>(), UVec4::new(2, 1, 3, 3));
    assert_eq!(v.swizzle::<SWIZZLE_Y, SWIZZLE_X, SWIZZLE_Z, SWIZZLE_W>(), UVec4::new(2, 1, 3, 4));
    assert_eq!(v.swizzle::<SWIZZLE_Y, SWIZZLE_X, SWIZZLE_W, SWIZZLE_X>(), UVec4::new(2, 1, 4, 1));
    assert_eq!(v.swizzle::<SWIZZLE_Y, SWIZZLE_X, SWIZZLE_W, SWIZZLE_Y>(), UVec4::new(2, 1, 4, 2));
    assert_eq!(v.swizzle::<SWIZZLE_Y, SWIZZLE_X, SWIZZLE_W, SWIZZLE_Z>(), UVec4::new(2, 1, 4, 3));
    assert_eq!(v.swizzle::<SWIZZLE_Y, SWIZZLE_X, SWIZZLE_W, SWIZZLE_W>(), UVec4::new(2, 1, 4, 4));
    assert_eq!(v.swizzle::<SWIZZLE_Y, SWIZZLE_Y, SWIZZLE_X, SWIZZLE_X>(), UVec4::new(2, 2, 1, 1));
    assert_eq!(v.swizzle::<SWIZZLE_Y, SWIZZLE_Y, SWIZZLE_X, SWIZZLE_Y>(), UVec4::new(2, 2, 1, 2));
    assert_eq!(v.swizzle::<SWIZZLE_Y, SWIZZLE_Y, SWIZZLE_X, SWIZZLE_Z>(), UVec4::new(2, 2, 1, 3));
    assert_eq!(v.swizzle::<SWIZZLE_Y, SWIZZLE_Y, SWIZZLE_X, SWIZZLE_W>(), UVec4::new(2, 2, 1, 4));
    assert_eq!(v.swizzle::<SWIZZLE_Y, SWIZZLE_Y, SWIZZLE_Y, SWIZZLE_X>(), UVec4::new(2, 2, 2, 1));
    assert_eq!(v.swizzle::<SWIZZLE_Y, SWIZZLE_Y, SWIZZLE_Y, SWIZZLE_Y>(), UVec4::new(2, 2, 2, 2));
    assert_eq!(v.swizzle::<SWIZZLE_Y, SWIZZLE_Y, SWIZZLE_Y, SWIZZLE_Z>(), UVec4::new(2, 2, 2, 3));
    assert_eq!(v.swizzle::<SWIZZLE_Y, SWIZZLE_Y, SWIZZLE_Y, SWIZZLE_W>(), UVec4::new(2, 2, 2, 4));
    assert_eq!(v.swizzle::<SWIZZLE_Y, SWIZZLE_Y, SWIZZLE_Z, SWIZZLE_X>(), UVec4::new(2, 2, 3, 1));
    assert_eq!(v.swizzle::<SWIZZLE_Y, SWIZZLE_Y, SWIZZLE_Z, SWIZZLE_Y>(), UVec4::new(2, 2, 3, 2));
    assert_eq!(v.swizzle::<SWIZZLE_Y, SWIZZLE_Y, SWIZZLE_Z, SWIZZLE_Z>(), UVec4::new(2, 2, 3, 3));
    assert_eq!(v.swizzle::<SWIZZLE_Y, SWIZZLE_Y, SWIZZLE_Z, SWIZZLE_W>(), UVec4::new(2, 2, 3, 4));
    assert_eq!(v.swizzle::<SWIZZLE_Y, SWIZZLE_Y, SWIZZLE_W, SWIZZLE_X>(), UVec4::new(2, 2, 4, 1));
    assert_eq!(v.swizzle::<SWIZZLE_Y, SWIZZLE_Y, SWIZZLE_W, SWIZZLE_Y>(), UVec4::new(2, 2, 4, 2));
    assert_eq!(v.swizzle::<SWIZZLE_Y, SWIZZLE_Y, SWIZZLE_W, SWIZZLE_Z>(), UVec4::new(2, 2, 4, 3));
    assert_eq!(v.swizzle::<SWIZZLE_Y, SWIZZLE_Y, SWIZZLE_W, SWIZZLE_W>(), UVec4::new(2, 2, 4, 4));
    assert_eq!(v.swizzle::<SWIZZLE_Y, SWIZZLE_Z, SWIZZLE_X, SWIZZLE_X>(), UVec4::new(2, 3, 1, 1));
    assert_eq!(v.swizzle::<SWIZZLE_Y, SWIZZLE_Z, SWIZZLE_X, SWIZZLE_Y>(), UVec4::new(2, 3, 1, 2));
    assert_eq!(v.swizzle::<SWIZZLE_Y, SWIZZLE_Z, SWIZZLE_X, SWIZZLE_Z>(), UVec4::new(2, 3, 1, 3));
    assert_eq!(v.swizzle::<SWIZZLE_Y, SWIZZLE_Z, SWIZZLE_X, SWIZZLE_W>(), UVec4::new(2, 3, 1, 4));
    assert_eq!(v.swizzle::<SWIZZLE_Y, SWIZZLE_Z, SWIZZLE_Y, SWIZZLE_X>(), UVec4::new(2, 3, 2, 1));
    assert_eq!(v.swizzle::<SWIZZLE_Y, SWIZZLE_Z, SWIZZLE_Y, SWIZZLE_Y>(), UVec4::new(2, 3, 2, 2));
    assert_eq!(v.swizzle::<SWIZZLE_Y, SWIZZLE_Z, SWIZZLE_Y, SWIZZLE_Z>(), UVec4::new(2, 3, 2, 3));
    assert_eq!(v.swizzle::<SWIZZLE_Y, SWIZZLE_Z, SWIZZLE_Y, SWIZZLE_W>(), UVec4::new(2, 3, 2, 4));
    assert_eq!(v.swizzle::<SWIZZLE_Y, SWIZZLE_Z, SWIZZLE_Z, SWIZZLE_X>(), UVec4::new(2, 3, 3, 1));
    assert_eq!(v.swizzle::<SWIZZLE_Y, SWIZZLE_Z, SWIZZLE_Z, SWIZZLE_Y>(), UVec4::new(2, 3, 3, 2));
    assert_eq!(v.swizzle::<SWIZZLE_Y, SWIZZLE_Z, SWIZZLE_Z, SWIZZLE_Z>(), UVec4::new(2, 3, 3, 3));
    assert_eq!(v.swizzle::<SWIZZLE_Y, SWIZZLE_Z, SWIZZLE_Z, SWIZZLE_W>(), UVec4::new(2, 3, 3, 4));
    assert_eq!(v.swizzle::<SWIZZLE_Y, SWIZZLE_Z, SWIZZLE_W, SWIZZLE_X>(), UVec4::new(2, 3, 4, 1));
    assert_eq!(v.swizzle::<SWIZZLE_Y, SWIZZLE_Z, SWIZZLE_W, SWIZZLE_Y>(), UVec4::new(2, 3, 4, 2));
    assert_eq!(v.swizzle::<SWIZZLE_Y, SWIZZLE_Z, SWIZZLE_W, SWIZZLE_Z>(), UVec4::new(2, 3, 4, 3));
    assert_eq!(v.swizzle::<SWIZZLE_Y, SWIZZLE_Z, SWIZZLE_W, SWIZZLE_W>(), UVec4::new(2, 3, 4, 4));
    assert_eq!(v.swizzle::<SWIZZLE_Y, SWIZZLE_W, SWIZZLE_X, SWIZZLE_X>(), UVec4::new(2, 4, 1, 1));
    assert_eq!(v.swizzle::<SWIZZLE_Y, SWIZZLE_W, SWIZZLE_X, SWIZZLE_Y>(), UVec4::new(2, 4, 1, 2));
    assert_eq!(v.swizzle::<SWIZZLE_Y, SWIZZLE_W, SWIZZLE_X, SWIZZLE_Z>(), UVec4::new(2, 4, 1, 3));
    assert_eq!(v.swizzle::<SWIZZLE_Y, SWIZZLE_W, SWIZZLE_X, SWIZZLE_W>(), UVec4::new(2, 4, 1, 4));
    assert_eq!(v.swizzle::<SWIZZLE_Y, SWIZZLE_W, SWIZZLE_Y, SWIZZLE_X>(), UVec4::new(2, 4, 2, 1));
    assert_eq!(v.swizzle::<SWIZZLE_Y, SWIZZLE_W, SWIZZLE_Y, SWIZZLE_Y>(), UVec4::new(2, 4, 2, 2));
    assert_eq!(v.swizzle::<SWIZZLE_Y, SWIZZLE_W, SWIZZLE_Y, SWIZZLE_Z>(), UVec4::new(2, 4, 2, 3));
    assert_eq!(v.swizzle::<SWIZZLE_Y, SWIZZLE_W, SWIZZLE_Y, SWIZZLE_W>(), UVec4::new(2, 4, 2, 4));
    assert_eq!(v.swizzle::<SWIZZLE_Y, SWIZZLE_W, SWIZZLE_Z, SWIZZLE_X>(), UVec4::new(2, 4, 3, 1));
    assert_eq!(v.swizzle::<SWIZZLE_Y, SWIZZLE_W, SWIZZLE_Z, SWIZZLE_Y>(), UVec4::new(2, 4, 3, 2));
    assert_eq!(v.swizzle::<SWIZZLE_Y, SWIZZLE_W, SWIZZLE_Z, SWIZZLE_Z>(), UVec4::new(2, 4, 3, 3));
    assert_eq!(v.swizzle::<SWIZZLE_Y, SWIZZLE_W, SWIZZLE_Z, SWIZZLE_W>(), UVec4::new(2, 4, 3, 4));
    assert_eq!(v.swizzle::<SWIZZLE_Y, SWIZZLE_W, SWIZZLE_W, SWIZZLE_X>(), UVec4::new(2, 4, 4, 1));
    assert_eq!(v.swizzle::<SWIZZLE_Y, SWIZZLE_W, SWIZZLE_W, SWIZZLE_Y>(), UVec4::new(2, 4, 4, 2));
    assert_eq!(v.swizzle::<SWIZZLE_Y, SWIZZLE_W, SWIZZLE_W, SWIZZLE_Z>(), UVec4::new(2, 4, 4, 3));
    assert_eq!(v.swizzle::<SWIZZLE_Y, SWIZZLE_W, SWIZZLE_W, SWIZZLE_W>(), UVec4::new(2, 4, 4, 4));

    assert_eq!(v.swizzle::<SWIZZLE_Z, SWIZZLE_X, SWIZZLE_X, SWIZZLE_X>(), UVec4::new(3, 1, 1, 1));
    assert_eq!(v.swizzle::<SWIZZLE_Z, SWIZZLE_X, SWIZZLE_X, SWIZZLE_Y>(), UVec4::new(3, 1, 1, 2));
    assert_eq!(v.swizzle::<SWIZZLE_Z, SWIZZLE_X, SWIZZLE_X, SWIZZLE_Z>(), UVec4::new(3, 1, 1, 3));
    assert_eq!(v.swizzle::<SWIZZLE_Z, SWIZZLE_X, SWIZZLE_X, SWIZZLE_W>(), UVec4::new(3, 1, 1, 4));
    assert_eq!(v.swizzle::<SWIZZLE_Z, SWIZZLE_X, SWIZZLE_Y, SWIZZLE_X>(), UVec4::new(3, 1, 2, 1));
    assert_eq!(v.swizzle::<SWIZZLE_Z, SWIZZLE_X, SWIZZLE_Y, SWIZZLE_Y>(), UVec4::new(3, 1, 2, 2));
    assert_eq!(v.swizzle::<SWIZZLE_Z, SWIZZLE_X, SWIZZLE_Y, SWIZZLE_Z>(), UVec4::new(3, 1, 2, 3));
    assert_eq!(v.swizzle::<SWIZZLE_Z, SWIZZLE_X, SWIZZLE_Y, SWIZZLE_W>(), UVec4::new(3, 1, 2, 4));
    assert_eq!(v.swizzle::<SWIZZLE_Z, SWIZZLE_X, SWIZZLE_Z, SWIZZLE_X>(), UVec4::new(3, 1, 3, 1));
    assert_eq!(v.swizzle::<SWIZZLE_Z, SWIZZLE_X, SWIZZLE_Z, SWIZZLE_Y>(), UVec4::new(3, 1, 3, 2));
    assert_eq!(v.swizzle::<SWIZZLE_Z, SWIZZLE_X, SWIZZLE_Z, SWIZZLE_Z>(), UVec4::new(3, 1, 3, 3));
    assert_eq!(v.swizzle::<SWIZZLE_Z, SWIZZLE_X, SWIZZLE_Z, SWIZZLE_W>(), UVec4::new(3, 1, 3, 4));
    assert_eq!(v.swizzle::<SWIZZLE_Z, SWIZZLE_X, SWIZZLE_W, SWIZZLE_X>(), UVec4::new(3, 1, 4, 1));
    assert_eq!(v.swizzle::<SWIZZLE_Z, SWIZZLE_X, SWIZZLE_W, SWIZZLE_Y>(), UVec4::new(3, 1, 4, 2));
    assert_eq!(v.swizzle::<SWIZZLE_Z, SWIZZLE_X, SWIZZLE_W, SWIZZLE_Z>(), UVec4::new(3, 1, 4, 3));
    assert_eq!(v.swizzle::<SWIZZLE_Z, SWIZZLE_X, SWIZZLE_W, SWIZZLE_W>(), UVec4::new(3, 1, 4, 4));
    assert_eq!(v.swizzle::<SWIZZLE_Z, SWIZZLE_Y, SWIZZLE_X, SWIZZLE_X>(), UVec4::new(3, 2, 1, 1));
    assert_eq!(v.swizzle::<SWIZZLE_Z, SWIZZLE_Y, SWIZZLE_X, SWIZZLE_Y>(), UVec4::new(3, 2, 1, 2));
    assert_eq!(v.swizzle::<SWIZZLE_Z, SWIZZLE_Y, SWIZZLE_X, SWIZZLE_Z>(), UVec4::new(3, 2, 1, 3));
    assert_eq!(v.swizzle::<SWIZZLE_Z, SWIZZLE_Y, SWIZZLE_X, SWIZZLE_W>(), UVec4::new(3, 2, 1, 4));
    assert_eq!(v.swizzle::<SWIZZLE_Z, SWIZZLE_Y, SWIZZLE_Y, SWIZZLE_X>(), UVec4::new(3, 2, 2, 1));
    assert_eq!(v.swizzle::<SWIZZLE_Z, SWIZZLE_Y, SWIZZLE_Y, SWIZZLE_Y>(), UVec4::new(3, 2, 2, 2));
    assert_eq!(v.swizzle::<SWIZZLE_Z, SWIZZLE_Y, SWIZZLE_Y, SWIZZLE_Z>(), UVec4::new(3, 2, 2, 3));
    assert_eq!(v.swizzle::<SWIZZLE_Z, SWIZZLE_Y, SWIZZLE_Y, SWIZZLE_W>(), UVec4::new(3, 2, 2, 4));
    assert_eq!(v.swizzle::<SWIZZLE_Z, SWIZZLE_Y, SWIZZLE_Z, SWIZZLE_X>(), UVec4::new(3, 2, 3, 1));
    assert_eq!(v.swizzle::<SWIZZLE_Z, SWIZZLE_Y, SWIZZLE_Z, SWIZZLE_Y>(), UVec4::new(3, 2, 3, 2));
    assert_eq!(v.swizzle::<SWIZZLE_Z, SWIZZLE_Y, SWIZZLE_Z, SWIZZLE_Z>(), UVec4::new(3, 2, 3, 3));
    assert_eq!(v.swizzle::<SWIZZLE_Z, SWIZZLE_Y, SWIZZLE_Z, SWIZZLE_W>(), UVec4::new(3, 2, 3, 4));
    assert_eq!(v.swizzle::<SWIZZLE_Z, SWIZZLE_Y, SWIZZLE_W, SWIZZLE_X>(), UVec4::new(3, 2, 4, 1));
    assert_eq!(v.swizzle::<SWIZZLE_Z, SWIZZLE_Y, SWIZZLE_W, SWIZZLE_Y>(), UVec4::new(3, 2, 4, 2));
    assert_eq!(v.swizzle::<SWIZZLE_Z, SWIZZLE_Y, SWIZZLE_W, SWIZZLE_Z>(), UVec4::new(3, 2, 4, 3));
    assert_eq!(v.swizzle::<SWIZZLE_Z, SWIZZLE_Y, SWIZZLE_W, SWIZZLE_W>(), UVec4::new(3, 2, 4, 4));
    assert_eq!(v.swizzle::<SWIZZLE_Z, SWIZZLE_Z, SWIZZLE_X, SWIZZLE_X>(), UVec4::new(3, 3, 1, 1));
    assert_eq!(v.swizzle::<SWIZZLE_Z, SWIZZLE_Z, SWIZZLE_X, SWIZZLE_Y>(), UVec4::new(3, 3, 1, 2));
    assert_eq!(v.swizzle::<SWIZZLE_Z, SWIZZLE_Z, SWIZZLE_X, SWIZZLE_Z>(), UVec4::new(3, 3, 1, 3));
    assert_eq!(v.swizzle::<SWIZZLE_Z, SWIZZLE_Z, SWIZZLE_X, SWIZZLE_W>(), UVec4::new(3, 3, 1, 4));
    assert_eq!(v.swizzle::<SWIZZLE_Z, SWIZZLE_Z, SWIZZLE_Y, SWIZZLE_X>(), UVec4::new(3, 3, 2, 1));
    assert_eq!(v.swizzle::<SWIZZLE_Z, SWIZZLE_Z, SWIZZLE_Y, SWIZZLE_Y>(), UVec4::new(3, 3, 2, 2));
    assert_eq!(v.swizzle::<SWIZZLE_Z, SWIZZLE_Z, SWIZZLE_Y, SWIZZLE_Z>(), UVec4::new(3, 3, 2, 3));
    assert_eq!(v.swizzle::<SWIZZLE_Z, SWIZZLE_Z, SWIZZLE_Y, SWIZZLE_W>(), UVec4::new(3, 3, 2, 4));
    assert_eq!(v.swizzle::<SWIZZLE_Z, SWIZZLE_Z, SWIZZLE_Z, SWIZZLE_X>(), UVec4::new(3, 3, 3, 1));
    assert_eq!(v.swizzle::<SWIZZLE_Z, SWIZZLE_Z, SWIZZLE_Z, SWIZZLE_Y>(), UVec4::new(3, 3, 3, 2));
    assert_eq!(v.swizzle::<SWIZZLE_Z, SWIZZLE_Z, SWIZZLE_Z, SWIZZLE_Z>(), UVec4::new(3, 3, 3, 3));
    assert_eq!(v.swizzle::<SWIZZLE_Z, SWIZZLE_Z, SWIZZLE_Z, SWIZZLE_W>(), UVec4::new(3, 3, 3, 4));
    assert_eq!(v.swizzle::<SWIZZLE_Z, SWIZZLE_Z, SWIZZLE_W, SWIZZLE_X>(), UVec4::new(3, 3, 4, 1));
    assert_eq!(v.swizzle::<SWIZZLE_Z, SWIZZLE_Z, SWIZZLE_W, SWIZZLE_Y>(), UVec4::new(3, 3, 4, 2));
    assert_eq!(v.swizzle::<SWIZZLE_Z, SWIZZLE_Z, SWIZZLE_W, SWIZZLE_Z>(), UVec4::new(3, 3, 4, 3));
    assert_eq!(v.swizzle::<SWIZZLE_Z, SWIZZLE_Z, SWIZZLE_W, SWIZZLE_W>(), UVec4::new(3, 3, 4, 4));
    assert_eq!(v.swizzle::<SWIZZLE_Z, SWIZZLE_W, SWIZZLE_X, SWIZZLE_X>(), UVec4::new(3, 4, 1, 1));
    assert_eq!(v.swizzle::<SWIZZLE_Z, SWIZZLE_W, SWIZZLE_X, SWIZZLE_Y>(), UVec4::new(3, 4, 1, 2));
    assert_eq!(v.swizzle::<SWIZZLE_Z, SWIZZLE_W, SWIZZLE_X, SWIZZLE_Z>(), UVec4::new(3, 4, 1, 3));
    assert_eq!(v.swizzle::<SWIZZLE_Z, SWIZZLE_W, SWIZZLE_X, SWIZZLE_W>(), UVec4::new(3, 4, 1, 4));
    assert_eq!(v.swizzle::<SWIZZLE_Z, SWIZZLE_W, SWIZZLE_Y, SWIZZLE_X>(), UVec4::new(3, 4, 2, 1));
    assert_eq!(v.swizzle::<SWIZZLE_Z, SWIZZLE_W, SWIZZLE_Y, SWIZZLE_Y>(), UVec4::new(3, 4, 2, 2));
    assert_eq!(v.swizzle::<SWIZZLE_Z, SWIZZLE_W, SWIZZLE_Y, SWIZZLE_Z>(), UVec4::new(3, 4, 2, 3));
    assert_eq!(v.swizzle::<SWIZZLE_Z, SWIZZLE_W, SWIZZLE_Y, SWIZZLE_W>(), UVec4::new(3, 4, 2, 4));
    assert_eq!(v.swizzle::<SWIZZLE_Z, SWIZZLE_W, SWIZZLE_Z, SWIZZLE_X>(), UVec4::new(3, 4, 3, 1));
    assert_eq!(v.swizzle::<SWIZZLE_Z, SWIZZLE_W, SWIZZLE_Z, SWIZZLE_Y>(), UVec4::new(3, 4, 3, 2));
    assert_eq!(v.swizzle::<SWIZZLE_Z, SWIZZLE_W, SWIZZLE_Z, SWIZZLE_Z>(), UVec4::new(3, 4, 3, 3));
    assert_eq!(v.swizzle::<SWIZZLE_Z, SWIZZLE_W, SWIZZLE_Z, SWIZZLE_W>(), UVec4::new(3, 4, 3, 4));
    assert_eq!(v.swizzle::<SWIZZLE_Z, SWIZZLE_W, SWIZZLE_W, SWIZZLE_X>(), UVec4::new(3, 4, 4, 1));
    assert_eq!(v.swizzle::<SWIZZLE_Z, SWIZZLE_W, SWIZZLE_W, SWIZZLE_Y>(), UVec4::new(3, 4, 4, 2));
    assert_eq!(v.swizzle::<SWIZZLE_Z, SWIZZLE_W, SWIZZLE_W, SWIZZLE_Z>(), UVec4::new(3, 4, 4, 3));
    assert_eq!(v.swizzle::<SWIZZLE_Z, SWIZZLE_W, SWIZZLE_W, SWIZZLE_W>(), UVec4::new(3, 4, 4, 4));

    assert_eq!(v.swizzle::<SWIZZLE_W, SWIZZLE_X, SWIZZLE_X, SWIZZLE_X>(), UVec4::new(4, 1, 1, 1));
    assert_eq!(v.swizzle::<SWIZZLE_W, SWIZZLE_X, SWIZZLE_X, SWIZZLE_Y>(), UVec4::new(4, 1, 1, 2));
    assert_eq!(v.swizzle::<SWIZZLE_W, SWIZZLE_X, SWIZZLE_X, SWIZZLE_Z>(), UVec4::new(4, 1, 1, 3));
    assert_eq!(v.swizzle::<SWIZZLE_W, SWIZZLE_X, SWIZZLE_X, SWIZZLE_W>(), UVec4::new(4, 1, 1, 4));
    assert_eq!(v.swizzle::<SWIZZLE_W, SWIZZLE_X, SWIZZLE_Y, SWIZZLE_X>(), UVec4::new(4, 1, 2, 1));
    assert_eq!(v.swizzle::<SWIZZLE_W, SWIZZLE_X, SWIZZLE_Y, SWIZZLE_Y>(), UVec4::new(4, 1, 2, 2));
    assert_eq!(v.swizzle::<SWIZZLE_W, SWIZZLE_X, SWIZZLE_Y, SWIZZLE_Z>(), UVec4::new(4, 1, 2, 3));
    assert_eq!(v.swizzle::<SWIZZLE_W, SWIZZLE_X, SWIZZLE_Y, SWIZZLE_W>(), UVec4::new(4, 1, 2, 4));
    assert_eq!(v.swizzle::<SWIZZLE_W, SWIZZLE_X, SWIZZLE_Z, SWIZZLE_X>(), UVec4::new(4, 1, 3, 1));
    assert_eq!(v.swizzle::<SWIZZLE_W, SWIZZLE_X, SWIZZLE_Z, SWIZZLE_Y>(), UVec4::new(4, 1, 3, 2));
    assert_eq!(v.swizzle::<SWIZZLE_W, SWIZZLE_X, SWIZZLE_Z, SWIZZLE_Z>(), UVec4::new(4, 1, 3, 3));
    assert_eq!(v.swizzle::<SWIZZLE_W, SWIZZLE_X, SWIZZLE_Z, SWIZZLE_W>(), UVec4::new(4, 1, 3, 4));
    assert_eq!(v.swizzle::<SWIZZLE_W, SWIZZLE_X, SWIZZLE_W, SWIZZLE_X>(), UVec4::new(4, 1, 4, 1));
    assert_eq!(v.swizzle::<SWIZZLE_W, SWIZZLE_X, SWIZZLE_W, SWIZZLE_Y>(), UVec4::new(4, 1, 4, 2));
    assert_eq!(v.swizzle::<SWIZZLE_W, SWIZZLE_X, SWIZZLE_W, SWIZZLE_Z>(), UVec4::new(4, 1, 4, 3));
    assert_eq!(v.swizzle::<SWIZZLE_W, SWIZZLE_X, SWIZZLE_W, SWIZZLE_W>(), UVec4::new(4, 1, 4, 4));
    assert_eq!(v.swizzle::<SWIZZLE_W, SWIZZLE_Y, SWIZZLE_X, SWIZZLE_X>(), UVec4::new(4, 2, 1, 1));
    assert_eq!(v.swizzle::<SWIZZLE_W, SWIZZLE_Y, SWIZZLE_X, SWIZZLE_Y>(), UVec4::new(4, 2, 1, 2));
    assert_eq!(v.swizzle::<SWIZZLE_W, SWIZZLE_Y, SWIZZLE_X, SWIZZLE_Z>(), UVec4::new(4, 2, 1, 3));
    assert_eq!(v.swizzle::<SWIZZLE_W, SWIZZLE_Y, SWIZZLE_X, SWIZZLE_W>(), UVec4::new(4, 2, 1, 4));
    assert_eq!(v.swizzle::<SWIZZLE_W, SWIZZLE_Y, SWIZZLE_Y, SWIZZLE_X>(), UVec4::new(4, 2, 2, 1));
    assert_eq!(v.swizzle::<SWIZZLE_W, SWIZZLE_Y, SWIZZLE_Y, SWIZZLE_Y>(), UVec4::new(4, 2, 2, 2));
    assert_eq!(v.swizzle::<SWIZZLE_W, SWIZZLE_Y, SWIZZLE_Y, SWIZZLE_Z>(), UVec4::new(4, 2, 2, 3));
    assert_eq!(v.swizzle::<SWIZZLE_W, SWIZZLE_Y, SWIZZLE_Y, SWIZZLE_W>(), UVec4::new(4, 2, 2, 4));
    assert_eq!(v.swizzle::<SWIZZLE_W, SWIZZLE_Y, SWIZZLE_Z, SWIZZLE_X>(), UVec4::new(4, 2, 3, 1));
    assert_eq!(v.swizzle::<SWIZZLE_W, SWIZZLE_Y, SWIZZLE_Z, SWIZZLE_Y>(), UVec4::new(4, 2, 3, 2));
    assert_eq!(v.swizzle::<SWIZZLE_W, SWIZZLE_Y, SWIZZLE_Z, SWIZZLE_Z>(), UVec4::new(4, 2, 3, 3));
    assert_eq!(v.swizzle::<SWIZZLE_W, SWIZZLE_Y, SWIZZLE_Z, SWIZZLE_W>(), UVec4::new(4, 2, 3, 4));
    assert_eq!(v.swizzle::<SWIZZLE_W, SWIZZLE_Y, SWIZZLE_W, SWIZZLE_X>(), UVec4::new(4, 2, 4, 1));
    assert_eq!(v.swizzle::<SWIZZLE_W, SWIZZLE_Y, SWIZZLE_W, SWIZZLE_Y>(), UVec4::new(4, 2, 4, 2));
    assert_eq!(v.swizzle::<SWIZZLE_W, SWIZZLE_Y, SWIZZLE_W, SWIZZLE_Z>(), UVec4::new(4, 2, 4, 3));
    assert_eq!(v.swizzle::<SWIZZLE_W, SWIZZLE_Y, SWIZZLE_W, SWIZZLE_W>(), UVec4::new(4, 2, 4, 4));
    assert_eq!(v.swizzle::<SWIZZLE_W, SWIZZLE_Z, SWIZZLE_X, SWIZZLE_X>(), UVec4::new(4, 3, 1, 1));
    assert_eq!(v.swizzle::<SWIZZLE_W, SWIZZLE_Z, SWIZZLE_X, SWIZZLE_Y>(), UVec4::new(4, 3, 1, 2));
    assert_eq!(v.swizzle::<SWIZZLE_W, SWIZZLE_Z, SWIZZLE_X, SWIZZLE_Z>(), UVec4::new(4, 3, 1, 3));
    assert_eq!(v.swizzle::<SWIZZLE_W, SWIZZLE_Z, SWIZZLE_X, SWIZZLE_W>(), UVec4::new(4, 3, 1, 4));
    assert_eq!(v.swizzle::<SWIZZLE_W, SWIZZLE_Z, SWIZZLE_Y, SWIZZLE_X>(), UVec4::new(4, 3, 2, 1));
    assert_eq!(v.swizzle::<SWIZZLE_W, SWIZZLE_Z, SWIZZLE_Y, SWIZZLE_Y>(), UVec4::new(4, 3, 2, 2));
    assert_eq!(v.swizzle::<SWIZZLE_W, SWIZZLE_Z, SWIZZLE_Y, SWIZZLE_Z>(), UVec4::new(4, 3, 2, 3));
    assert_eq!(v.swizzle::<SWIZZLE_W, SWIZZLE_Z, SWIZZLE_Y, SWIZZLE_W>(), UVec4::new(4, 3, 2, 4));
    assert_eq!(v.swizzle::<SWIZZLE_W, SWIZZLE_Z, SWIZZLE_Z, SWIZZLE_X>(), UVec4::new(4, 3, 3, 1));
    assert_eq!(v.swizzle::<SWIZZLE_W, SWIZZLE_Z, SWIZZLE_Z, SWIZZLE_Y>(), UVec4::new(4, 3, 3, 2));
    assert_eq!(v.swizzle::<SWIZZLE_W, SWIZZLE_Z, SWIZZLE_Z, SWIZZLE_Z>(), UVec4::new(4, 3, 3, 3));
    assert_eq!(v.swizzle::<SWIZZLE_W, SWIZZLE_Z, SWIZZLE_Z, SWIZZLE_W>(), UVec4::new(4, 3, 3, 4));
    assert_eq!(v.swizzle::<SWIZZLE_W, SWIZZLE_Z, SWIZZLE_W, SWIZZLE_X>(), UVec4::new(4, 3, 4, 1));
    assert_eq!(v.swizzle::<SWIZZLE_W, SWIZZLE_Z, SWIZZLE_W, SWIZZLE_Y>(), UVec4::new(4, 3, 4, 2));
    assert_eq!(v.swizzle::<SWIZZLE_W, SWIZZLE_Z, SWIZZLE_W, SWIZZLE_Z>(), UVec4::new(4, 3, 4, 3));
    assert_eq!(v.swizzle::<SWIZZLE_W, SWIZZLE_Z, SWIZZLE_W, SWIZZLE_W>(), UVec4::new(4, 3, 4, 4));
    assert_eq!(v.swizzle::<SWIZZLE_W, SWIZZLE_W, SWIZZLE_X, SWIZZLE_X>(), UVec4::new(4, 4, 1, 1));
    assert_eq!(v.swizzle::<SWIZZLE_W, SWIZZLE_W, SWIZZLE_X, SWIZZLE_Y>(), UVec4::new(4, 4, 1, 2));
    assert_eq!(v.swizzle::<SWIZZLE_W, SWIZZLE_W, SWIZZLE_X, SWIZZLE_Z>(), UVec4::new(4, 4, 1, 3));
    assert_eq!(v.swizzle::<SWIZZLE_W, SWIZZLE_W, SWIZZLE_X, SWIZZLE_W>(), UVec4::new(4, 4, 1, 4));
    assert_eq!(v.swizzle::<SWIZZLE_W, SWIZZLE_W, SWIZZLE_Y, SWIZZLE_X>(), UVec4::new(4, 4, 2, 1));
    assert_eq!(v.swizzle::<SWIZZLE_W, SWIZZLE_W, SWIZZLE_Y, SWIZZLE_Y>(), UVec4::new(4, 4, 2, 2));
    assert_eq!(v.swizzle::<SWIZZLE_W, SWIZZLE_W, SWIZZLE_Y, SWIZZLE_Z>(), UVec4::new(4, 4, 2, 3));
    assert_eq!(v.swizzle::<SWIZZLE_W, SWIZZLE_W, SWIZZLE_Y, SWIZZLE_W>(), UVec4::new(4, 4, 2, 4));
    assert_eq!(v.swizzle::<SWIZZLE_W, SWIZZLE_W, SWIZZLE_Z, SWIZZLE_X>(), UVec4::new(4, 4, 3, 1));
    assert_eq!(v.swizzle::<SWIZZLE_W, SWIZZLE_W, SWIZZLE_Z, SWIZZLE_Y>(), UVec4::new(4, 4, 3, 2));
    assert_eq!(v.swizzle::<SWIZZLE_W, SWIZZLE_W, SWIZZLE_Z, SWIZZLE_Z>(), UVec4::new(4, 4, 3, 3));
    assert_eq!(v.swizzle::<SWIZZLE_W, SWIZZLE_W, SWIZZLE_Z, SWIZZLE_W>(), UVec4::new(4, 4, 3, 4));
    assert_eq!(v.swizzle::<SWIZZLE_W, SWIZZLE_W, SWIZZLE_W, SWIZZLE_X>(), UVec4::new(4, 4, 4, 1));
    assert_eq!(v.swizzle::<SWIZZLE_W, SWIZZLE_W, SWIZZLE_W, SWIZZLE_Y>(), UVec4::new(4, 4, 4, 2));
    assert_eq!(v.swizzle::<SWIZZLE_W, SWIZZLE_W, SWIZZLE_W, SWIZZLE_Z>(), UVec4::new(4, 4, 4, 3));
    assert_eq!(v.swizzle::<SWIZZLE_W, SWIZZLE_W, SWIZZLE_W, SWIZZLE_W>(), UVec4::new(4, 4, 4, 4));
}

#[test]
fn test_u_vec4_dot() {
    assert_eq!(UVec4::new(1, 2, 3, 4).dot(UVec4::new(5, 6, 7, 8)), 1 * 5 + 2 * 6 + 3 * 7 + 4 * 8);
    assert_eq!(UVec4::new(1, 2, 3, 4).dot_v(UVec4::new(5, 6, 7, 8)), UVec4::replicate(1 * 5 + 2 * 6 + 3 * 7 + 4 * 8));
}

#[test]
fn test_u_vec4_cast() {
    assert_eq!(UVec4::new(1, 2, 3, 4).to_float(), Vec4::new(1.0, 2.0, 3.0, 4.0));
    assert_eq!(
        UVec4::new(0x3f800000, 0x40000000, 0x40400000, 0x40800000).reinterpret_as_float(),
        Vec4::new(1.0, 2.0, 3.0, 4.0)
    );
}

#[test]
fn test_u_vec4_extract_u_int16() {
    let data: [u32; 4] = [0x0b020a01, 0x0d040c03, 0x0b060a05, 0x0d080c07];
    let vector = UVec4::load_int4(&data);
    assert_eq!(vector.expand4_uint16_lo(), UVec4::new(0x0a01, 0x0b02, 0x0c03, 0x0d04));
    assert_eq!(vector.expand4_uint16_hi(), UVec4::new(0x0a05, 0x0b06, 0x0c07, 0x0d08));
}

#[test]
fn test_u_vec4_extract_bytes() {
    let data: [u32; 4] = [0x14131211, 0x24232221, 0x34333231, 0x44434241];
    let vector = UVec4::load_int4(&data);
    assert_eq!(vector.expand4_byte0(), UVec4::new(0x11, 0x12, 0x13, 0x14));
    assert_eq!(vector.expand4_byte4(), UVec4::new(0x21, 0x22, 0x23, 0x24));
    assert_eq!(vector.expand4_byte8(), UVec4::new(0x31, 0x32, 0x33, 0x34));
    assert_eq!(vector.expand4_byte12(), UVec4::new(0x41, 0x42, 0x43, 0x44));
}

#[test]
fn test_u_vec4_shift_components() {
    let v = UVec4::new(1, 2, 3, 4);
    assert_eq!(v.shift_components4_minus(4), UVec4::new(1, 2, 3, 4));
    assert_eq!(v.shift_components4_minus(3), UVec4::new(2, 3, 4, 0));
    assert_eq!(v.shift_components4_minus(2), UVec4::new(3, 4, 0, 0));
    assert_eq!(v.shift_components4_minus(1), UVec4::new(4, 0, 0, 0));
    assert_eq!(v.shift_components4_minus(0), UVec4::new(0, 0, 0, 0));
}

#[test]
fn test_u_vec4_sort4_true() {
    assert_eq!(
        UVec4::sort4_true(UVec4::new(0x00000000, 0x00000000, 0x00000000, 0x00000000), UVec4::new(1, 2, 3, 4)),
        UVec4::new(4, 4, 4, 4)
    );
    assert_eq!(
        UVec4::sort4_true(UVec4::new(0xffffffff, 0x00000000, 0x00000000, 0x00000000), UVec4::new(1, 2, 3, 4)),
        UVec4::new(1, 4, 4, 4)
    );
    assert_eq!(
        UVec4::sort4_true(UVec4::new(0x00000000, 0xffffffff, 0x00000000, 0x00000000), UVec4::new(1, 2, 3, 4)),
        UVec4::new(2, 4, 4, 4)
    );
    assert_eq!(
        UVec4::sort4_true(UVec4::new(0xffffffff, 0xffffffff, 0x00000000, 0x00000000), UVec4::new(1, 2, 3, 4)),
        UVec4::new(1, 2, 4, 4)
    );
    assert_eq!(
        UVec4::sort4_true(UVec4::new(0x00000000, 0x00000000, 0xffffffff, 0x00000000), UVec4::new(1, 2, 3, 4)),
        UVec4::new(3, 4, 4, 4)
    );
    assert_eq!(
        UVec4::sort4_true(UVec4::new(0xffffffff, 0x00000000, 0xffffffff, 0x00000000), UVec4::new(1, 2, 3, 4)),
        UVec4::new(1, 3, 4, 4)
    );
    assert_eq!(
        UVec4::sort4_true(UVec4::new(0x00000000, 0xffffffff, 0xffffffff, 0x00000000), UVec4::new(1, 2, 3, 4)),
        UVec4::new(2, 3, 4, 4)
    );
    assert_eq!(
        UVec4::sort4_true(UVec4::new(0xffffffff, 0xffffffff, 0xffffffff, 0x00000000), UVec4::new(1, 2, 3, 4)),
        UVec4::new(1, 2, 3, 4)
    );
    assert_eq!(
        UVec4::sort4_true(UVec4::new(0x00000000, 0x00000000, 0x00000000, 0xffffffff), UVec4::new(1, 2, 3, 4)),
        UVec4::new(4, 4, 4, 4)
    );
    assert_eq!(
        UVec4::sort4_true(UVec4::new(0xffffffff, 0x00000000, 0x00000000, 0xffffffff), UVec4::new(1, 2, 3, 4)),
        UVec4::new(1, 4, 4, 4)
    );
    assert_eq!(
        UVec4::sort4_true(UVec4::new(0x00000000, 0xffffffff, 0x00000000, 0xffffffff), UVec4::new(1, 2, 3, 4)),
        UVec4::new(2, 4, 4, 4)
    );
    assert_eq!(
        UVec4::sort4_true(UVec4::new(0xffffffff, 0xffffffff, 0x00000000, 0xffffffff), UVec4::new(1, 2, 3, 4)),
        UVec4::new(1, 2, 4, 4)
    );
    assert_eq!(
        UVec4::sort4_true(UVec4::new(0x00000000, 0x00000000, 0xffffffff, 0xffffffff), UVec4::new(1, 2, 3, 4)),
        UVec4::new(3, 4, 4, 4)
    );
    assert_eq!(
        UVec4::sort4_true(UVec4::new(0xffffffff, 0x00000000, 0xffffffff, 0xffffffff), UVec4::new(1, 2, 3, 4)),
        UVec4::new(1, 3, 4, 4)
    );
    assert_eq!(
        UVec4::sort4_true(UVec4::new(0x00000000, 0xffffffff, 0xffffffff, 0xffffffff), UVec4::new(1, 2, 3, 4)),
        UVec4::new(2, 3, 4, 4)
    );
    assert_eq!(
        UVec4::sort4_true(UVec4::new(0xffffffff, 0xffffffff, 0xffffffff, 0xffffffff), UVec4::new(1, 2, 3, 4)),
        UVec4::new(1, 2, 3, 4)
    );
}

#[test]
fn test_u_vec4_convert_to_string() {
    let v = UVec4::new(1, 2, 3, 4);
    assert_eq!(v.to_string(), "1, 2, 3, 4");
}

#[test]
fn test_u_vec4_to_float_is_signed() {
    // Not in the C++ tests: like _mm_cvtepi32_ps (SSE) the components are converted as signed integers
    assert_eq!(UVec4::new(0xffffffff, 0x80000000, 0, 1).to_float(), Vec4::new(-1.0, -2147483648.0, 0.0, 1.0));
}
