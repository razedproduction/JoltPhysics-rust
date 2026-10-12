//! Port of UnitTests/Math/BVec16Tests.cpp

use jolt_core::math::BVec16;

#[test]
fn test_b_vec16_construct() {
    let mut v = BVec16::new(1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16);

    for i in 0..16 {
        assert_eq!(v[i], i as u8 + 1);
    }

    // Test == and != operators
    assert!(v == BVec16::new(1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16));
    assert!(v != BVec16::new(1, 2, 3, 4, 5, 6, 7, 8, 10, 9, 11, 12, 13, 14, 15, 16));

    // Check element modification
    assert_eq!((&v)[15], 16); // Check const operator
    v[15] = 17;
    assert_eq!(v[15], 17);
    assert_eq!(v, BVec16::new(1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 17));
}

#[test]
fn test_b_vec16_load_byte16() {
    let u16: [u8; 16] = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16];
    assert_eq!(BVec16::load_byte16(&u16), BVec16::new(1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16));
}

#[test]
fn test_b_vec16_zero() {
    let v = BVec16::zero();

    for i in 0..16 {
        assert_eq!(v[i], 0);
    }
}

#[test]
fn test_b_vec16_replicate() {
    assert_eq!(BVec16::replicate(2), BVec16::new(2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2));
}

#[test]
fn test_b_vec16_comparisons() {
    let eq = BVec16::equals(
        BVec16::new(1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16),
        BVec16::new(6, 7, 3, 4, 5, 6, 7, 5, 9, 10, 11, 12, 13, 14, 15, 13),
    );
    assert_eq!(eq.trues(), 0b0111111101111100);
    assert!(eq.test_any_true());
    assert!(!eq.test_all_true());
}

#[test]
fn test_b_vec16_bit_ops() {
    // Test all bit permutations
    let v1 = BVec16::new(
        0b011, 0b0110, 0b01100, 0b011000, 0b0110000, 0b01100000, 0b011, 0b0110, 0b01100, 0b011000, 0b0110000,
        0b01100000, 0b011, 0b0110, 0b01100, 0b011000,
    );
    let v2 = BVec16::new(
        0b101, 0b1010, 0b10100, 0b101000, 0b1010000, 0b10100000, 0b101, 0b1010, 0b10100, 0b101000, 0b1010000,
        0b10100000, 0b101, 0b1010, 0b10100, 0b101000,
    );

    assert_eq!(
        BVec16::or(v1, v2),
        BVec16::new(
            0b111, 0b1110, 0b11100, 0b111000, 0b1110000, 0b11100000, 0b111, 0b1110, 0b11100, 0b111000, 0b1110000,
            0b11100000, 0b111, 0b1110, 0b11100, 0b111000
        )
    );
    assert_eq!(
        BVec16::xor(v1, v2),
        BVec16::new(
            0b110, 0b1100, 0b11000, 0b110000, 0b1100000, 0b11000000, 0b110, 0b1100, 0b11000, 0b110000, 0b1100000,
            0b11000000, 0b110, 0b1100, 0b11000, 0b110000
        )
    );
    assert_eq!(
        BVec16::and(v1, v2),
        BVec16::new(
            0b001, 0b0010, 0b00100, 0b001000, 0b0010000, 0b00100000, 0b001, 0b0010, 0b00100, 0b001000, 0b0010000,
            0b00100000, 0b001, 0b0010, 0b00100, 0b001000
        )
    );

    assert_eq!(
        BVec16::not(v1),
        BVec16::new(
            0b11111100, 0b11111001, 0b11110011, 0b11100111, 0b11001111, 0b10011111, 0b11111100, 0b11111001, 0b11110011,
            0b11100111, 0b11001111, 0b10011111, 0b11111100, 0b11111001, 0b11110011, 0b11100111
        )
    );
    assert_eq!(
        BVec16::not(v2),
        BVec16::new(
            0b11111010, 0b11110101, 0b11101011, 0b11010111, 0b10101111, 0b01011111, 0b11111010, 0b11110101, 0b11101011,
            0b11010111, 0b10101111, 0b01011111, 0b11111010, 0b11110101, 0b11101011, 0b11010111
        )
    );
}

#[test]
fn test_b_vec16_to_string() {
    let v = BVec16::new(1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16);

    assert_eq!(v.to_string(), "1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16");
}
