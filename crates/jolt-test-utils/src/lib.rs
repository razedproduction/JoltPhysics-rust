//! Shared helpers for porting `UnitTests/` (port of `UnitTestFramework.h`, `PhysicsTestContext`,
//! `Layers.h` and the `Logging*Listener.h` helpers). Grows together with the crates it tests.

use std::fmt::{Debug, Display};

use jolt_core::math::vec3::RandomGenerator;
use jolt_core::math::{DVec3, Vec3, Vec4};

/// Types that `CHECK_APPROX_EQUAL` accepts
pub trait ApproxEqual: Copy + Debug {
    /// Type of the tolerance
    type Tolerance: Copy + Display;

    /// Default tolerance of `CHECK_APPROX_EQUAL` for this type
    const DEFAULT_TOLERANCE: Self::Tolerance;

    /// The check `CHECK_APPROX_EQUAL` does for this type
    fn approx_equal(self, rhs: Self, tolerance: Self::Tolerance) -> bool;
}

impl ApproxEqual for f32 {
    type Tolerance = f32;
    const DEFAULT_TOLERANCE: f32 = 1.0e-6;

    fn approx_equal(self, rhs: Self, tolerance: f32) -> bool {
        (rhs - self).abs() <= tolerance
    }
}

impl ApproxEqual for f64 {
    type Tolerance = f64;
    const DEFAULT_TOLERANCE: f64 = 1.0e-6;

    fn approx_equal(self, rhs: Self, tolerance: f64) -> bool {
        (rhs - self).abs() <= tolerance
    }
}

impl ApproxEqual for Vec3 {
    type Tolerance = f32;
    const DEFAULT_TOLERANCE: f32 = 1.0e-6;

    fn approx_equal(self, rhs: Self, tolerance: f32) -> bool {
        self.is_close(rhs, tolerance * tolerance)
    }
}

impl ApproxEqual for Vec4 {
    type Tolerance = f32;
    const DEFAULT_TOLERANCE: f32 = 1.0e-6;

    fn approx_equal(self, rhs: Self, tolerance: f32) -> bool {
        self.is_close(rhs, tolerance * tolerance)
    }
}

impl ApproxEqual for DVec3 {
    type Tolerance = f64;
    const DEFAULT_TOLERANCE: f64 = 1.0e-6;

    fn approx_equal(self, rhs: Self, tolerance: f64) -> bool {
        self.is_close(rhs, tolerance * tolerance)
    }
}

/// Port of `CHECK_APPROX_EQUAL(lhs, rhs, tolerance)`
#[track_caller]
pub fn check_approx_equal_tol<T: ApproxEqual>(lhs: T, rhs: T, tolerance: T::Tolerance) {
    assert!(lhs.approx_equal(rhs, tolerance), "{lhs:?} != {rhs:?} (tolerance {tolerance})");
}

/// Port of `CHECK_APPROX_EQUAL(lhs, rhs)` with the default tolerance of `1.0e-6`
#[track_caller]
pub fn check_approx_equal<T: ApproxEqual>(lhs: T, rhs: T) {
    check_approx_equal_tol(lhs, rhs, T::DEFAULT_TOLERANCE);
}

/// Port of `CHECK_APPROX_EQUAL(float, float, tolerance)`: `|rhs - lhs| <= tolerance`.
#[track_caller]
pub fn check_approx_equal_f32(lhs: f32, rhs: f32, tolerance: f32) {
    check_approx_equal_tol(lhs, rhs, tolerance);
}

/// Port of `CHECK_APPROX_EQUAL(double, double, tolerance)`: `|rhs - lhs| <= tolerance`.
#[track_caller]
pub fn check_approx_equal_f64(lhs: f64, rhs: f64, tolerance: f64) {
    check_approx_equal_tol(lhs, rhs, tolerance);
}

/// Port of `UnitTestRandom` (`std::mt19937`, the 32 bit Mersenne Twister). Produces the same sequence as the C++
/// standard library for the same seed.
#[derive(Clone, Debug)]
pub struct UnitTestRandom {
    state: [u32; 624],
    index: usize,
}

impl UnitTestRandom {
    /// Seed used by a default constructed `std::mt19937`
    pub const DEFAULT_SEED: u32 = 5489;

    /// Construct with a seed (`std::mt19937(seed)`)
    pub fn new(seed: u32) -> Self {
        let mut state = [0u32; 624];
        state[0] = seed;
        for i in 1..624 {
            state[i] = 1812433253u32.wrapping_mul(state[i - 1] ^ (state[i - 1] >> 30)).wrapping_add(i as u32);
        }
        Self { state, index: 624 }
    }

    /// Generate the next random number (`operator ()`)
    pub fn next_u32(&mut self) -> u32 {
        if self.index >= 624 {
            for i in 0..624 {
                let y = (self.state[i] & 0x8000_0000) | (self.state[(i + 1) % 624] & 0x7fff_ffff);
                let mut v = self.state[(i + 397) % 624] ^ (y >> 1);
                if y & 1 != 0 {
                    v ^= 0x9908_b0df;
                }
                self.state[i] = v;
            }
            self.index = 0;
        }

        let mut y = self.state[self.index];
        self.index += 1;
        y ^= y >> 11;
        y ^= (y << 7) & 0x9d2c_5680;
        y ^= (y << 15) & 0xefc6_0000;
        y ^= y >> 18;
        y
    }

    /// Shuffle `slice` (`std::shuffle(begin, end, random)`). The algorithm of `std::shuffle` is implementation defined,
    /// so this is a Fisher-Yates shuffle that does not reproduce any particular standard library.
    pub fn shuffle<T>(&mut self, slice: &mut [T]) {
        for i in (1..slice.len()).rev() {
            let j = (self.next_u32() as u64 * (i as u64 + 1)) >> 32;
            slice.swap(i, j as usize);
        }
    }
}

impl UnitTestRandom {
    /// Draw a value from `std::uniform_real_distribution<float>(a, b)`. The algorithm is implementation defined; this
    /// is the one of libstdc++ (`generate_canonical<float, 24>` takes one 32 bit sample).
    pub fn uniform_f32(&mut self, a: f32, b: f32) -> f32 {
        let mut p = self.next_u32() as f32 / 4294967296.0;
        if p >= 1.0 {
            p = 1.0 - f32::EPSILON / 2.0; // nextafter(1, 0)
        }
        (b - a) * p + a
    }
}

impl RandomGenerator for UnitTestRandom {
    fn min_value(&self) -> u64 {
        0
    }

    fn max_value(&self) -> u64 {
        u32::MAX as u64
    }

    fn generate(&mut self) -> u64 {
        self.next_u32() as u64
    }
}

impl Default for UnitTestRandom {
    fn default() -> Self {
        Self::new(Self::DEFAULT_SEED)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn mt19937_matches_std() {
        // The C++ standard requires the 10000th invocation of a default constructed mt19937 to produce 4123659995
        let mut random = UnitTestRandom::default();
        for _ in 0..9999 {
            random.next_u32();
        }
        assert_eq!(random.next_u32(), 4123659995);
    }

    #[test]
    fn approx_equal_accepts_within_tolerance() {
        check_approx_equal_f32(1.0, 1.0 + 0.5e-6, 1.0e-6);
        check_approx_equal_f64(1.0, 1.0 + 0.5e-6, 1.0e-6);
    }

    #[test]
    #[should_panic]
    fn approx_equal_rejects_outside_tolerance() {
        check_approx_equal_f32(1.0, 1.1, 1.0e-6);
    }
}
