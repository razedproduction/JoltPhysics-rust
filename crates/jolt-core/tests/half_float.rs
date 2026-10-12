//! Port of UnitTests/Math/HalfFloatTests.cpp
//!
//! `TestHalfFloatToFloat` and the half float -> float halves of `TestHalfFloatINF` / `TestHalfFloatNaN` need
//! `UVec4` / `Vec4` and follow in slice 4. `TestFloatToHalfFloat` compares the fallback with the F16C instruction
//! (`_mm_cvtps_ph`), so it only checks something on x86 CPUs that have F16C.

use jolt_core::math::half_float::{
    FLOAT_EXPONENT_BIAS, FLOAT_EXPONENT_MASK, FLOAT_EXPONENT_POS, FLOAT_MANTISSA_BITS, HALF_FLT_EXPONENT_BIAS,
    HALF_FLT_INF, HALF_FLT_INF_NEGATIVE, HALF_FLT_MANTISSA_BITS, HALF_FLT_NANQ, HALF_FLT_NANQ_NEGATIVE, HalfFloat,
    RoundingMode, from_float, from_float_fallback,
};

/// Hardware conversion (C++ `FromFloat` when `JPH_USE_F16C` is defined), `None` if the CPU has no F16C
#[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
fn from_float_f16c(v: f32, rounding_mode: RoundingMode) -> Option<HalfFloat> {
    #[cfg(target_arch = "x86")]
    use std::arch::x86::*;
    #[cfg(target_arch = "x86_64")]
    use std::arch::x86_64::*;

    #[target_feature(enable = "f16c")]
    fn convert(v: f32, rounding_mode: RoundingMode) -> HalfFloat {
        let val = _mm_set_ss(v);
        let hf = match rounding_mode {
            RoundingMode::RoundToNegInf => _mm_cvtps_ph::<_MM_FROUND_TO_NEG_INF>(val),
            RoundingMode::RoundToPosInf => _mm_cvtps_ph::<_MM_FROUND_TO_POS_INF>(val),
            RoundingMode::RoundToNearest => _mm_cvtps_ph::<_MM_FROUND_TO_NEAREST_INT>(val),
        };
        _mm_extract_epi16::<0>(hf) as HalfFloat
    }

    if is_x86_feature_detected!("f16c") {
        // SAFETY: the CPU supports F16C (checked above)
        Some(unsafe { convert(v, rounding_mode) })
    } else {
        None
    }
}

#[cfg(not(any(target_arch = "x86", target_arch = "x86_64")))]
fn from_float_f16c(v: f32, rounding_mode: RoundingMode) -> Option<HalfFloat> {
    // C++ FromFloat has no NEON path, so it is the fallback there as well
    Some(from_float(v, rounding_mode))
}

/// Helper function to compare the intrinsics version with the fallback version, returns false if there is no
/// intrinsics version
fn check_float_to_half_float(value: u32, sign: u32) -> bool {
    let fvalue = f32::from_bits(value.wrapping_add(sign.wrapping_mul(0x80000000)));

    for rounding_mode in [RoundingMode::RoundToNearest, RoundingMode::RoundToPosInf, RoundingMode::RoundToNegInf] {
        let Some(hf1) = from_float_f16c(fvalue, rounding_mode) else {
            return false;
        };
        let hf2 = from_float_fallback(fvalue, rounding_mode);
        // Not using assert_eq! with formatting per value to keep the loop fast
        if hf1 != hf2 {
            panic!("{value:#x} sign {sign} {rounding_mode:?}: intrinsic {hf1:#x} != fallback {hf2:#x}");
        }
        if from_float(fvalue, rounding_mode) != hf2 {
            panic!("{value:#x} sign {sign} {rounding_mode:?}: from_float differs from fallback");
        }
    }
    true
}

#[test]
fn test_float_to_half_float() {
    if !check_float_to_half_float(0, 0) {
        eprintln!("CPU has no F16C, skipping test_float_to_half_float");
        return;
    }

    for sign in 0..2 {
        // Zero and smallest possible float
        for value in 0..2 {
            check_float_to_half_float(value, sign);
        }

        // Floats that are large enough to become a denormalized half float, incrementing by smallest increment that can make a difference
        let start =
            ((FLOAT_EXPONENT_BIAS - HALF_FLT_EXPONENT_BIAS - HALF_FLT_MANTISSA_BITS) << FLOAT_EXPONENT_POS) as u32;
        let end = (FLOAT_EXPONENT_MASK << FLOAT_EXPONENT_POS) as u32;
        let step = 1usize << (FLOAT_MANTISSA_BITS - HALF_FLT_MANTISSA_BITS - 2);
        for value in (start..end).step_by(step) {
            check_float_to_half_float(value, sign);
        }

        // INF
        check_float_to_half_float(0x7f800000, sign);

        // Nan
        check_float_to_half_float(0x7fc00000, sign);
    }
}

#[test]
fn test_half_float_inf() {
    // Float -> half float
    assert_eq!(from_float_fallback(f32::from_bits(0x7f800000), RoundingMode::RoundToNearest), HALF_FLT_INF);
    assert_eq!(from_float_fallback(f32::from_bits(0xff800000), RoundingMode::RoundToNearest), HALF_FLT_INF_NEGATIVE);

    // Half float -> float: needs UVec4 / Vec4, follows in slice 4
}

#[test]
fn test_half_float_na_n() {
    // Float -> half float
    assert_eq!(from_float_fallback(f32::from_bits(0x7fc00000), RoundingMode::RoundToNearest), HALF_FLT_NANQ);
    assert_eq!(from_float_fallback(f32::from_bits(0xffc00000), RoundingMode::RoundToNearest), HALF_FLT_NANQ_NEGATIVE);

    // Half float -> float: needs UVec4 / Vec4, follows in slice 4
}
