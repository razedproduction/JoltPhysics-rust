//! Port of `Jolt/Math/FindRoot.h`.

use super::{Float, sign, sqrt};
use std::ops::{Add, Div, Mul, Neg};

/// Find the roots of \f$inA \: x^2 + inB \: x + inC = 0\f$.
/// @return The number of roots, actual roots in outX1 and outX2.
/// If number of roots returned is 1 then outX1 == outX2.
///
/// Returns `(number of roots, x1, x2)`; when there are no roots `x1` and `x2` are zero (C++ leaves them untouched).
#[inline]
pub fn find_root<T>(a: T, b: T, c: T) -> (i32, T, T)
where
    T: Float + Add<Output = T> + Mul<Output = T> + Div<Output = T> + Neg<Output = T> + From<f32>,
{
    let zero = T::ZERO;

    // Check if this is a linear equation
    if a == zero {
        // Check if this is a constant equation
        if b == zero {
            if c == zero {
                return (1, zero, zero); // Actually infinitely many solutions
            }

            return (0, zero, zero);
        }

        // Linear equation with 1 solution
        let x = -c / b;
        return (1, x, x);
    }

    // See Numerical Recipes in C, Chapter 5.6 Quadratic and Cubic Equations
    let det = T::difference_of_products(b, b, T::from(4.0) * a, c);
    if det < zero {
        return (0, zero, zero);
    }
    let q = (b + sign(b) * sqrt(det)) / T::from(-2.0);
    let x1 = q / a;
    if q == zero {
        return (1, x1, x1);
    }
    let x2 = c / q;
    (2, x1, x2)
}
