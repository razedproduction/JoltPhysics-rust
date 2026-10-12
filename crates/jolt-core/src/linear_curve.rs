//! Port of `Jolt/Core/LinearCurve.h` + `LinearCurve.cpp`.
//!
//! Not ported (see `port/DEVIATIONS.md`): the `ObjectStream` serialization (`JPH_DECLARE_SERIALIZABLE_NON_VIRTUAL`)
//! and `SaveBinaryState` / `RestoreBinaryState`, which follow with `StreamIn` / `StreamOut` in slice 11.

use crate::quick_sort::quick_sort_by;

/// A point on the curve
#[derive(Clone, Copy, Debug, Default, PartialEq)]
pub struct Point {
    pub x: f32,
    pub y: f32,
}

/// A set of points (x, y) that form a linear curve
#[derive(Clone, Debug, Default, PartialEq)]
pub struct LinearCurve {
    /// The points on the curve, should be sorted ascending by x
    pub points: Vec<Point>,
}

impl LinearCurve {
    /// Remove all points
    #[inline]
    pub fn clear(&mut self) {
        self.points.clear();
    }

    /// Reserve memory for `num_points` points
    #[inline]
    pub fn reserve(&mut self, num_points: u32) {
        self.points.reserve(num_points as usize);
    }

    /// Add a point to the curve. Points must be inserted in ascending X or Sort() needs to be called when all points have been added.
    /// @param x X value
    /// @param y Y value
    #[inline]
    pub fn add_point(&mut self, x: f32, y: f32) {
        self.points.push(Point { x, y });
    }

    /// Sort the points on X ascending
    #[inline]
    pub fn sort(&mut self) {
        quick_sort_by(&mut self.points, |lhs, rhs| lhs.x < rhs.x);
    }

    /// Get the lowest X value
    #[inline]
    pub fn min_x(&self) -> f32 {
        self.points.first().map_or(0.0, |p| p.x)
    }

    /// Get the highest X value
    #[inline]
    pub fn max_x(&self) -> f32 {
        self.points.last().map_or(0.0, |p| p.x)
    }

    /// Sample value on the curve
    /// @param x X value to sample at
    /// @return Interpolated Y value
    pub fn value(&self, x: f32) -> f32 {
        if self.points.is_empty() {
            return 0.0;
        }

        let i2 = lower_bound(&self.points, |point| point.x < x);

        if i2 == 0 {
            return self.points[0].y;
        } else if i2 == self.points.len() {
            return self.points[self.points.len() - 1].y;
        }

        let p1 = &self.points[i2 - 1];
        let p2 = &self.points[i2];
        p1.y + (x - p1.x) * (p2.y - p1.y) / (p2.x - p1.x)
    }
}

/// `std::lower_bound`: index of the first element for which `less` (element < value) returns false.
/// Uses the same probing sequence as the common STL implementations, so the result also matches C++ for input that is
/// not sorted (where `slice::partition_point` could differ).
fn lower_bound<T>(slice: &[T], mut less: impl FnMut(&T) -> bool) -> usize {
    let mut first = 0;
    let mut count = slice.len();
    while count > 0 {
        let step = count / 2;
        let it = first + step;
        if less(&slice[it]) {
            first = it + 1;
            count -= step + 1;
        } else {
            count = step;
        }
    }
    first
}
