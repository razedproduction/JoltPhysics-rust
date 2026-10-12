//! Port of `Jolt/Math/HalfFloat.h` (C++ namespace `HalfFloatConversion` is this module).
//!
//! The F16C intrinsic paths of `FromFloat` / `ToFloat` are not ported: the fallbacks produce the same bits
//! (`UnitTests/Math/HalfFloatTests.cpp` checks this, see `tests/half_float.rs`).

use super::{UVec4, Vec4};

pub type HalfFloat = u16;

// Define half float constant values
pub const HALF_FLT_MAX: HalfFloat = 0x7bff;
pub const HALF_FLT_MAX_NEGATIVE: HalfFloat = 0xfbff;
pub const HALF_FLT_INF: HalfFloat = 0x7c00;
pub const HALF_FLT_INF_NEGATIVE: HalfFloat = 0xfc00;
pub const HALF_FLT_NANQ: HalfFloat = 0x7e00;
pub const HALF_FLT_NANQ_NEGATIVE: HalfFloat = 0xfe00;

// Layout of a float
pub const FLOAT_SIGN_POS: i32 = 31;
pub const FLOAT_EXPONENT_POS: i32 = 23;
pub const FLOAT_EXPONENT_BITS: i32 = 8;
pub const FLOAT_EXPONENT_MASK: i32 = (1 << FLOAT_EXPONENT_BITS) - 1;
pub const FLOAT_EXPONENT_BIAS: i32 = 127;
pub const FLOAT_MANTISSA_BITS: i32 = 23;
pub const FLOAT_MANTISSA_MASK: i32 = (1 << FLOAT_MANTISSA_BITS) - 1;
pub const FLOAT_EXPONENT_AND_MANTISSA_MASK: i32 = FLOAT_MANTISSA_MASK + (FLOAT_EXPONENT_MASK << FLOAT_EXPONENT_POS);

// Layout of half float
pub const HALF_FLT_SIGN_POS: i32 = 15;
pub const HALF_FLT_EXPONENT_POS: i32 = 10;
pub const HALF_FLT_EXPONENT_BITS: i32 = 5;
pub const HALF_FLT_EXPONENT_MASK: i32 = (1 << HALF_FLT_EXPONENT_BITS) - 1;
pub const HALF_FLT_EXPONENT_BIAS: i32 = 15;
pub const HALF_FLT_MANTISSA_BITS: i32 = 10;
pub const HALF_FLT_MANTISSA_MASK: i32 = (1 << HALF_FLT_MANTISSA_BITS) - 1;
pub const HALF_FLT_EXPONENT_AND_MANTISSA_MASK: i32 =
    HALF_FLT_MANTISSA_MASK + (HALF_FLT_EXPONENT_MASK << HALF_FLT_EXPONENT_POS);

/// Define half-float rounding modes
///
/// C++ passes the rounding mode as a template argument; here it is a regular argument (constant folded after inlining).
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum RoundingMode {
    /// Round to negative infinity
    RoundToNegInf,
    /// Round to positive infinity
    RoundToPosInf,
    /// Round to nearest value
    RoundToNearest,
}

/// Convert a float (32-bits) to a half float (16-bits), fallback version when no intrinsics available
#[inline]
pub fn from_float_fallback(v: f32, rounding_mode: RoundingMode) -> HalfFloat {
    // Reinterpret the float as an uint32
    let value = v.to_bits();

    // Extract exponent
    let exponent = (value >> FLOAT_EXPONENT_POS) & FLOAT_EXPONENT_MASK as u32;

    // Extract mantissa
    let mut mantissa = value & FLOAT_MANTISSA_MASK as u32;

    // Extract the sign and move it into the right spot for the half float (so we can just or it in at the end)
    let hf_sign = ((value >> (FLOAT_SIGN_POS - HALF_FLT_SIGN_POS)) as HalfFloat) & (1 << HALF_FLT_SIGN_POS);

    // Check NaN or INF
    if exponent == FLOAT_EXPONENT_MASK as u32 {
        // NaN or INF
        return hf_sign | if mantissa == 0 { HALF_FLT_INF } else { HALF_FLT_NANQ };
    }

    // Rebias the exponent for half floats
    let rebiased_exponent = exponent as i32 - FLOAT_EXPONENT_BIAS + HALF_FLT_EXPONENT_BIAS;

    // Check overflow to infinity
    if rebiased_exponent >= HALF_FLT_EXPONENT_MASK {
        let round_up = rounding_mode == RoundingMode::RoundToNearest
            || (hf_sign == 0) == (rounding_mode == RoundingMode::RoundToPosInf);
        return hf_sign | if round_up { HALF_FLT_INF } else { HALF_FLT_MAX };
    }

    // Check underflow to zero
    if rebiased_exponent < -HALF_FLT_MANTISSA_BITS {
        let round_up = rounding_mode != RoundingMode::RoundToNearest
            && (hf_sign == 0) == (rounding_mode == RoundingMode::RoundToPosInf)
            && (value & FLOAT_EXPONENT_AND_MANTISSA_MASK as u32) != 0;
        return hf_sign | if round_up { 1 } else { 0 };
    }

    let hf_exponent: HalfFloat;
    let shift: i32;
    if rebiased_exponent <= 0 {
        // Underflow to denormalized number
        hf_exponent = 0;
        mantissa |= 1 << FLOAT_MANTISSA_BITS; // Add the implicit 1 bit to the mantissa
        shift = FLOAT_MANTISSA_BITS - HALF_FLT_MANTISSA_BITS + 1 - rebiased_exponent;
    } else {
        // Normal half float
        hf_exponent = (rebiased_exponent << HALF_FLT_EXPONENT_POS) as HalfFloat;
        shift = FLOAT_MANTISSA_BITS - HALF_FLT_MANTISSA_BITS;
    }

    // Compose the half float
    let hf_mantissa = (mantissa >> shift) as HalfFloat;
    let mut hf = hf_sign | hf_exponent | hf_mantissa;

    // Calculate the remaining bits that we're discarding
    let remainder = mantissa & ((1u32 << shift) - 1);

    if rounding_mode == RoundingMode::RoundToNearest {
        // Round to nearest
        let round_threshold = 1u32 << (shift - 1);
        if remainder > round_threshold // Above threshold, we must always round
            || (remainder == round_threshold && (hf_mantissa & 1) != 0)
        // When equal, round to nearest even
        {
            hf += 1; // May overflow to infinity
        }
    } else {
        // Round up or down (truncate) depending on the rounding mode
        let round_up = (hf_sign == 0) == (rounding_mode == RoundingMode::RoundToPosInf) && remainder != 0;
        if round_up {
            hf += 1; // May overflow to infinity
        }
    }

    hf
}

/// Convert a float (32-bits) to a half float (16-bits)
#[inline]
pub fn from_float(v: f32, rounding_mode: RoundingMode) -> HalfFloat {
    from_float_fallback(v, rounding_mode)
}

/// Convert 4 half floats (lower 64 bits) to floats, fallback version when no intrinsics available
#[inline]
pub fn to_float_fallback(value: UVec4) -> Vec4 {
    // Unpack half floats to 4 uint32's
    let value = value.expand4_uint16_lo();

    // Normal half float path, extract the exponent and mantissa, shift them into place and update the exponent bias
    let exponent_mantissa = UVec4::and(value, UVec4::replicate(HALF_FLT_EXPONENT_AND_MANTISSA_MASK as u32))
        .logical_shift_left::<{ (FLOAT_EXPONENT_POS - HALF_FLT_EXPONENT_POS) as u32 }>()
        + UVec4::replicate(((FLOAT_EXPONENT_BIAS - HALF_FLT_EXPONENT_BIAS) << FLOAT_EXPONENT_POS) as u32);

    // Denormalized half float path, renormalize the float
    let exponent_mantissa_denormalized = ((exponent_mantissa + UVec4::replicate((1 << FLOAT_EXPONENT_POS) as u32))
        .reinterpret_as_float()
        - UVec4::replicate(((FLOAT_EXPONENT_BIAS - HALF_FLT_EXPONENT_BIAS + 1) << FLOAT_EXPONENT_POS) as u32)
            .reinterpret_as_float())
    .reinterpret_as_int();

    // NaN / INF path, set all exponent bits
    let exponent_mantissa_nan_inf =
        UVec4::or(exponent_mantissa, UVec4::replicate((FLOAT_EXPONENT_MASK << FLOAT_EXPONENT_POS) as u32));

    // Get the exponent to determine which of the paths we should take
    let exponent_mask = UVec4::replicate((HALF_FLT_EXPONENT_MASK << HALF_FLT_EXPONENT_POS) as u32);
    let exponent = UVec4::and(value, exponent_mask);
    let is_denormalized = UVec4::equals(exponent, UVec4::zero());
    let is_nan_inf = UVec4::equals(exponent, exponent_mask);

    // Select the correct result
    let result_exponent_mantissa = UVec4::select(
        UVec4::select(exponent_mantissa, exponent_mantissa_nan_inf, is_nan_inf),
        exponent_mantissa_denormalized,
        is_denormalized,
    );

    // Extract the sign bit and shift it to the left
    let sign = UVec4::and(value, UVec4::replicate((1 << HALF_FLT_SIGN_POS) as u32))
        .logical_shift_left::<{ (FLOAT_SIGN_POS - HALF_FLT_SIGN_POS) as u32 }>();

    // Construct the float
    UVec4::or(sign, result_exponent_mantissa).reinterpret_as_float()
}

/// Convert 4 half floats (lower 64 bits) to floats
#[inline]
pub fn to_float(value: UVec4) -> Vec4 {
    to_float_fallback(value)
}
