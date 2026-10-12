//! Port of UnitTests/Math/HalfFloatTests.cpp
//!
//! `TestHalfFloatToFloat` and `TestFloatToHalfFloat` compare the fallbacks with the F16C instructions
//! (`_mm_cvtph_ps` / `_mm_cvtps_ph`), so they only check something on x86 CPUs that have F16C.

use jolt_core::math::half_float::{
    FLOAT_EXPONENT_BIAS, FLOAT_EXPONENT_MASK, FLOAT_EXPONENT_POS, FLOAT_MANTISSA_BITS, HALF_FLT_EXPONENT_BIAS,
    HALF_FLT_INF, HALF_FLT_INF_NEGATIVE, HALF_FLT_MANTISSA_BITS, HALF_FLT_NANQ, HALF_FLT_NANQ_NEGATIVE, HalfFloat,
    RoundingMode, from_float, from_float_fallback, to_float, to_float_fallback,
};
use jolt_core::math::{UVec4, Vec4};

/// Hardware conversion (C++ `ToFloat` when `JPH_USE_F16C` is defined), `None` if the CPU has no F16C
#[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
fn to_float_f16c(value: UVec4) -> Option<Vec4> {
    #[cfg(target_arch = "x86")]
    use std::arch::x86::*;
    #[cfg(target_arch = "x86_64")]
    use std::arch::x86_64::*;

    #[target_feature(enable = "f16c")]
    fn convert(value: UVec4) -> Vec4 {
        let v = value.to_array();
        let half = _mm_set_epi32(v[3] as i32, v[2] as i32, v[1] as i32, v[0] as i32);
        let mut out = [0.0f32; 4];
        // SAFETY: `out` has room for the 4 floats that _mm_storeu_ps writes (no alignment needed)
        unsafe { _mm_storeu_ps(out.as_mut_ptr(), _mm_cvtph_ps(half)) };
        Vec4::from_array(out)
    }

    if is_x86_feature_detected!("f16c") {
        // SAFETY: the CPU supports F16C (checked above)
        Some(unsafe { convert(value) })
    } else {
        None
    }
}

#[cfg(not(any(target_arch = "x86", target_arch = "x86_64")))]
fn to_float_f16c(value: UVec4) -> Option<Vec4> {
    // C++ ToFloat has no NEON path, so it is the fallback there as well
    Some(to_float(value))
}

#[test]
fn test_half_float_to_float() {
    if to_float_f16c(UVec4::zero()).is_none() {
        eprintln!("CPU has no F16C, skipping test_half_float_to_float");
        return;
    }

    // Check all half float values, 4 at a time, skip NaN's and INF
    for v in (0u32..0x7c00).step_by(2) {
        // Test value, next value and negative variants of both
        let half_float = UVec4::new(v | ((v + 1) << 16), (v | 0x8000) | (((v + 1) | 0x8000) << 16), 0, 0);

        // Compare hardware intrinsic version with fallback version
        let flt1 = to_float_f16c(half_float).unwrap();
        let flt2 = to_float_fallback(half_float);

        let flt1_as_int = flt1.reinterpret_as_int();
        let flt2_as_int = flt2.reinterpret_as_int();
        // Not using assert_eq! with formatting per value to keep the loop fast
        if flt1_as_int != flt2_as_int {
            panic!("{v:#x}: intrinsic {flt1_as_int} != fallback {flt2_as_int}");
        }
        if to_float(half_float).reinterpret_as_int() != flt2_as_int {
            panic!("{v:#x}: to_float differs from fallback");
        }
    }
}

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

    // Half float -> float
    let half_float = UVec4::new(HALF_FLT_INF as u32 | ((HALF_FLT_INF_NEGATIVE as u32) << 16), 0, 0, 0);
    let flt = to_float_fallback(half_float).reinterpret_as_int();
    assert_eq!(flt, UVec4::new(0x7f800000, 0xff800000, 0, 0));
}

#[test]
fn test_half_float_na_n() {
    // Float -> half float
    assert_eq!(from_float_fallback(f32::from_bits(0x7fc00000), RoundingMode::RoundToNearest), HALF_FLT_NANQ);
    assert_eq!(from_float_fallback(f32::from_bits(0xffc00000), RoundingMode::RoundToNearest), HALF_FLT_NANQ_NEGATIVE);

    // Half float -> float
    let half_float = UVec4::new(HALF_FLT_NANQ as u32 | ((HALF_FLT_NANQ_NEGATIVE as u32) << 16), 0, 0, 0);
    let flt = to_float_fallback(half_float).reinterpret_as_int();
    assert_eq!(flt, UVec4::new(0x7fc00000, 0xffc00000, 0, 0));
}
