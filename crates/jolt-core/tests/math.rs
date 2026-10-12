//! Port of UnitTests/Math/MathTests.cpp

use jolt_core::math::{
    JPH_PI, center_angle_around_zero, count_bits, count_leading_zeros, count_trailing_zeros, difference_of_products,
    find_root, get_next_power_of_2, is_power_of_2, mul_rsqrt_approximate, sqrt,
};
use jolt_test_utils::check_approx_equal_f32;
use std::hint::black_box;

#[test]
fn test_count_trailing_zeros() {
    assert_eq!(count_trailing_zeros(0), 32);
    for i in 0..32 {
        assert_eq!(count_trailing_zeros(1u32 << i), i);
    }
}

#[test]
fn test_count_leading_zeros() {
    assert_eq!(count_leading_zeros(0), 32);
    for i in 0..32 {
        assert_eq!(count_leading_zeros(1u32 << i), 31 - i);
    }
}

#[test]
fn test_count_bits() {
    assert_eq!(count_bits(0), 0);
    assert_eq!(count_bits(0b10000000000000000000000000000000), 1);
    assert_eq!(count_bits(0b00000000000000000000000000000001), 1);
    assert_eq!(count_bits(0b10000000000000001000000000000000), 2);
    assert_eq!(count_bits(0b00000000000000010000000000000001), 2);
    assert_eq!(count_bits(0b10000000100000001000000010000000), 4);
    assert_eq!(count_bits(0b00000001000000010000000100000001), 4);
    assert_eq!(count_bits(0b10001000100010001000100010001000), 8);
    assert_eq!(count_bits(0b00010001000100010001000100010001), 8);
    assert_eq!(count_bits(0b10101010101010101010101010101010), 16);
    assert_eq!(count_bits(0b01010101010101010101010101010101), 16);
    assert_eq!(count_bits(0b11111111111111111111111111111111), 32);
}

#[test]
fn test_next_power_of2() {
    assert_eq!(get_next_power_of_2(0), 1);

    for i in 0..31 {
        let pow = 1u32 << i;
        if pow > 2 {
            assert_eq!(get_next_power_of_2(pow - 1), pow);
        }
        assert_eq!(get_next_power_of_2(pow), pow);
        assert_eq!(get_next_power_of_2(pow + 1), pow << 1);
    }

    assert_eq!(get_next_power_of_2(0x8000000 - 1), 0x8000000);
    assert_eq!(get_next_power_of_2(0x8000000), 0x8000000);
}

#[test]
fn test_center_angle_around_zero() {
    for i in (0..10).step_by(2) {
        let i = i as f32;
        check_approx_equal_f32(center_angle_around_zero(i * JPH_PI), 0.0, 1.0e-5);
        check_approx_equal_f32(center_angle_around_zero((0.5 + i) * JPH_PI), 0.5 * JPH_PI, 1.0e-5);
        check_approx_equal_f32(center_angle_around_zero((1.5 + i) * JPH_PI), -0.5 * JPH_PI, 1.0e-5);
        check_approx_equal_f32(center_angle_around_zero(-(0.5 + i) * JPH_PI), -0.5 * JPH_PI, 1.0e-5);
        check_approx_equal_f32(center_angle_around_zero(-(1.5 + i) * JPH_PI), 0.5 * JPH_PI, 1.0e-5);
        check_approx_equal_f32(center_angle_around_zero(-(0.99 + i) * JPH_PI), -0.99 * JPH_PI, 1.0e-5);
        check_approx_equal_f32(center_angle_around_zero((0.99 + i) * JPH_PI), 0.99 * JPH_PI, 1.0e-5);
    }
}

#[test]
fn test_is_power_of2() {
    // Note: the C++ test checks `uint64(1) << 1` (instead of `<< i`) 63 times, ported as-is (see port/DEVIATIONS.md)
    for _ in 0..63 {
        assert!(is_power_of_2(1u64 << 1));
    }
    assert!(!is_power_of_2(-2));
    assert!(!is_power_of_2(0));
    assert!(!is_power_of_2(3));
    assert!(!is_power_of_2(5));
    assert!(!is_power_of_2(15));
    assert!(!is_power_of_2(17));
    assert!(!is_power_of_2(65535));
    assert!(!is_power_of_2(65537));
}

#[test]
fn test_find_root() {
    {
        // Quadratic, 2 solutions
        let (n, x1, x2) = find_root(2.0f32, 8.0, 6.0);
        assert_eq!(n, 2);
        assert_eq!(x1, -3.0);
        assert_eq!(x2, -1.0);
    }

    {
        // Quadratic without b term, 2 solutions
        let (n, x1, x2) = find_root(2.0f32, 0.0, -8.0);
        assert_eq!(n, 2);
        assert_eq!(x1, -2.0);
        assert_eq!(x2, 2.0);
    }

    {
        // Quadratic without solution
        let (n, _, _) = find_root(2.0f32, 1.0, 8.0);
        assert_eq!(n, 0);
    }

    {
        // Linear
        let (n, x1, x2) = find_root(0.0f32, 8.0, 4.0);
        assert_eq!(n, 1);
        assert_eq!(x1, -0.5);
        assert_eq!(x2, -0.5);
    }

    {
        // Constant
        let (n, x1, x2) = find_root(0.0f32, 0.0, 0.0);
        assert_eq!(n, 1);
        assert_eq!(x1, 0.0);
        assert_eq!(x2, 0.0);
    }

    {
        // Constant, no solution
        let (n, _, _) = find_root(0.0f32, 0.0, 1.0);
        assert_eq!(n, 0);
    }
}

#[test]
fn test_difference_of_products() {
    // Use black_box to try to prevent the compiler from doing the calculation at compile time (C++: volatile)
    let (a, b, c, d) = black_box((33962.035f32, -30438.8f32, 41563.4f32, -24871.969f32));
    let result = difference_of_products(a, b, c, d);
    let expected = a as f64 * b as f64 - c as f64 * d as f64;
    assert_eq!(expected, -75.165603637695312);
    // The port never uses fused multiply add (C++ without JPH_USE_FMADD): the products are in the order of 10^9, so
    // the subtraction causes a large loss of precision and we get a very different result.
    assert_eq!(result, -128.0);
}

#[test]
fn test_mul_r_sqrt_approximate() {
    let mut y = f32::MIN_POSITIVE;
    while y < 1.0e10 {
        let result = mul_rsqrt_approximate(5.0, y);
        let expected = 5.0 / sqrt(y);
        check_approx_equal_f32(expected, result, expected * 3.0e-7); // about 22 bits of precision
        y *= 1.5;
    }
}
