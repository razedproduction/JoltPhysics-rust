//! Port of `Jolt/Core/InsertionSort.h`.
//!
//! The C++ version takes an iterator range and a "less than" predicate; here it takes a slice and a predicate
//! `compare(a, b) -> bool` that returns true if `a` goes before `b`. The predicate is called in exactly the same order
//! as in C++, so the resulting element order (the sort is not stable) is identical.

/// Implementation of the insertion sort algorithm.
pub fn insertion_sort_by<T, F>(slice: &mut [T], mut compare: F)
where
    F: FnMut(&T, &T) -> bool,
{
    insertion_sort_impl(slice, &mut compare);
}

/// Implementation of insertion sort algorithm without comparator (sorts ascending using `<`).
pub fn insertion_sort<T: PartialOrd>(slice: &mut [T]) {
    insertion_sort_impl(slice, &mut |a: &T, b: &T| a < b);
}

/// Shared implementation, takes the comparator by reference so that [`quick_sort`](crate::quick_sort) can reuse it.
pub(crate) fn insertion_sort_impl<T, F>(slice: &mut [T], compare: &mut F)
where
    F: FnMut(&T, &T) -> bool,
{
    // Empty arrays don't need to be sorted
    if slice.is_empty() {
        return;
    }

    // Start at the second element
    for i in 1..slice.len() {
        // C++ moves this element to a temporary value `x` and shifts elements to the right to make space for it. The
        // elements that are compared against are not touched by the shifting, so we compare against them in place and
        // perform the shift + insert as a single rotate.

        // Check if the element goes before the first element (we can't decrement the iterator before inBegin so this
        // needs to be a separate branch)
        let j = if compare(&slice[i], &slice[0]) {
            // Move all elements to the right to make space for x, x goes to the first place
            0
        } else {
            // Move elements to the right as long as they are bigger than x
            let mut j = i;
            while compare(&slice[i], &slice[j - 1]) {
                j -= 1;
            }
            j
        };

        // Move x into place
        slice[j..=i].rotate_right(1);
    }
}
