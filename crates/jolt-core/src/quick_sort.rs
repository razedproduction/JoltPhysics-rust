//! Port of `Jolt/Core/QuickSort.h`.
//!
//! The C++ version takes an iterator range and a "less than" predicate; here it takes a slice and a predicate
//! `compare(a, b) -> bool` that returns true if `a` goes before `b`. The algorithm, the predicate calls and therefore
//! the resulting element order (the sort is not stable) are identical to C++. Never replace this by
//! `sort_unstable_by`: the order of equal elements is observable and matters for determinism.

use crate::insertion_sort::insertion_sort_impl;

/// Helper function for QuickSort, will move the pivot element to `middle`.
fn quick_sort_median_of_three<T, F>(slice: &mut [T], first: usize, middle: usize, last: usize, compare: &mut F)
where
    F: FnMut(&T, &T) -> bool,
{
    // This should be guaranteed because we switch over to insertion sort when there's 32 or less elements
    debug_assert!(first != middle && middle != last);

    if compare(&slice[middle], &slice[first]) {
        slice.swap(first, middle);
    }

    if compare(&slice[last], &slice[first]) {
        slice.swap(first, last);
    }

    if compare(&slice[last], &slice[middle]) {
        slice.swap(middle, last);
    }
}

/// Helper function for QuickSort using the Ninther method, will move the pivot element to `middle`.
fn quick_sort_ninther<T, F>(slice: &mut [T], first: usize, middle: usize, last: usize, compare: &mut F)
where
    F: FnMut(&T, &T) -> bool,
{
    // Divide the range in 8 equal parts (this means there are 9 points)
    let diff = (last - first) >> 3;
    let two_diff = diff << 1;

    // Median of first 3 points
    let mid1 = first + diff;
    quick_sort_median_of_three(slice, first, mid1, first + two_diff, compare);

    // Median of second 3 points
    quick_sort_median_of_three(slice, middle - diff, middle, middle + diff, compare);

    // Median of third 3 points
    let mid3 = last - diff;
    quick_sort_median_of_three(slice, last - two_diff, mid3, last, compare);

    // Determine the median of the 3 medians
    quick_sort_median_of_three(slice, mid1, middle, mid3, compare);
}

/// Implementation of the quick sort algorithm. The STL version implementation is not consistent across platforms.
pub fn quick_sort_by<T, F>(slice: &mut [T], mut compare: F)
where
    F: FnMut(&T, &T) -> bool,
{
    quick_sort_impl(slice, &mut compare);
}

/// Implementation of quick sort algorithm without comparator (sorts ascending using `<`).
pub fn quick_sort<T: PartialOrd>(slice: &mut [T]) {
    quick_sort_impl(slice, &mut |a: &T, b: &T| a < b);
}

fn quick_sort_impl<T, F>(slice: &mut [T], compare: &mut F)
where
    F: FnMut(&T, &T) -> bool,
{
    // Implementation based on https://en.wikipedia.org/wiki/Quicksort using Hoare's partition scheme
    let mut slice = slice;

    // Loop so that we only need to do 1 recursive call instead of 2.
    loop {
        // If there's less than 2 elements we're done
        let num_elements = slice.len();
        if num_elements < 2 {
            return;
        }

        // Fall back to insertion sort if there are too few elements
        if num_elements <= 32 {
            insertion_sort_impl(slice, compare);
            return;
        }

        // Determine pivot
        let pivot_index = (num_elements - 1) >> 1;
        quick_sort_ninther(slice, 0, pivot_index, num_elements - 1, compare);

        // C++ takes a copy of the pivot here. The pivot element can be swapped around below, so we track its index
        // instead (the value, and therefore every comparison, is the same).
        let mut pivot = pivot_index;

        // Left and right iterators
        let mut i = 0;
        let mut j = num_elements;

        loop {
            // Find the first element that is bigger than the pivot
            while compare(&slice[i], &slice[pivot]) {
                i += 1;
            }

            // Find the last element that is smaller than the pivot
            loop {
                j -= 1;
                if !compare(&slice[pivot], &slice[j]) {
                    break;
                }
            }

            // If the two iterators crossed, we're done
            if i >= j {
                break;
            }

            // Swap the elements
            slice.swap(i, j);
            if pivot == i {
                pivot = j;
            } else if pivot == j {
                pivot = i;
            }

            // Note that the first while loop in this function should
            // have been do i++ while (...) but since we cannot decrement
            // the iterator from inBegin we left that out, so we need to do
            // it here.
            i += 1;
        }

        // Include the middle element on the left side
        j += 1;

        // Check which partition is smaller
        let (left, right) = std::mem::take(&mut slice).split_at_mut(j);
        if j < num_elements - j {
            // Left side is smaller, recurse to left first
            quick_sort_impl(left, compare);

            // Loop again with the right side to avoid a call
            slice = right;
        } else {
            // Right side is smaller, recurse to right first
            quick_sort_impl(right, compare);

            // Loop again with the left side to avoid a call
            slice = left;
        }
    }
}
