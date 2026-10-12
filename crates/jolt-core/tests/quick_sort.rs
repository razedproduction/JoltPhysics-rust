//! Port of UnitTests/Core/QuickSortTest.cpp

use jolt_core::quick_sort::{quick_sort, quick_sort_by};
use jolt_test_utils::UnitTestRandom;

#[test]
fn test_ordered_array() {
    let mut array: Vec<i32> = (0..100).collect();

    quick_sort(&mut array);

    for i in 0..100 {
        assert_eq!(array[i as usize], i);
    }
}

#[test]
fn test_ordered_array_comparator() {
    let mut array: Vec<i32> = (0..100).collect();

    quick_sort_by(&mut array, |a, b| a > b);

    for i in 0..100 {
        assert_eq!(array[i as usize], 99 - i);
    }
}

#[test]
fn test_reversed_array() {
    let mut array: Vec<i32> = (0..100).map(|i| 99 - i).collect();

    quick_sort(&mut array);

    for i in 0..100 {
        assert_eq!(array[i as usize], i);
    }
}

#[test]
fn test_random_array() {
    let mut random = UnitTestRandom::default();

    let mut array: Vec<u32> = Vec::new();
    for _ in 0..1000 {
        let value = random.next_u32();

        // Insert value at beginning
        array.insert(0, value);

        // Insert value at end
        array.push(value);
    }

    quick_sort(&mut array);

    for i in (0..array.len() - 2).step_by(2) {
        // We inserted the same value twice so these elements should be the same
        assert_eq!(array[i], array[i + 1]);

        // The next element should be bigger or equal
        assert!(array[i] <= array[i + 2]);
    }
}

#[test]
fn test_empty_array() {
    let mut array: Vec<i32> = Vec::new();
    quick_sort(&mut array);
    assert!(array.is_empty());
}

#[test]
fn test1_element_array() {
    let mut array = vec![1];
    quick_sort(&mut array);
    assert_eq!(array[0], 1);
}

#[test]
fn test2_element_array() {
    let mut array = vec![2, 1];
    quick_sort(&mut array);
    assert_eq!(array[0], 1);
    assert_eq!(array[1], 2);
}

/// Not in the C++ test: equal keys with distinct payloads must end up in the exact order the C++ algorithm produces.
/// Checked against a straightforward transcription of the C++ algorithm that copies the pivot value like C++ does
/// (the port tracks the pivot index instead).
#[test]
fn quick_sort_matches_pivot_copy_reference() {
    fn reference<T: Clone>(v: &mut [T], less: &impl Fn(&T, &T) -> bool) {
        let n = v.len();
        if n < 2 {
            return;
        }
        if n <= 32 {
            jolt_core::insertion_sort::insertion_sort_by(v, less);
            return;
        }
        let m3 = |v: &mut [T], a: usize, b: usize, c: usize| {
            if less(&v[b], &v[a]) {
                v.swap(a, b);
            }
            if less(&v[c], &v[a]) {
                v.swap(a, c);
            }
            if less(&v[c], &v[b]) {
                v.swap(b, c);
            }
        };
        let mid = (n - 1) >> 1;
        let last = n - 1;
        let diff = last >> 3;
        let two_diff = diff << 1;
        m3(v, 0, diff, two_diff);
        m3(v, mid - diff, mid, mid + diff);
        m3(v, last - two_diff, last - diff, last);
        m3(v, diff, mid, last - diff);
        let pivot = v[mid].clone();
        let (mut i, mut j) = (0, n);
        loop {
            while less(&v[i], &pivot) {
                i += 1;
            }
            loop {
                j -= 1;
                if !less(&pivot, &v[j]) {
                    break;
                }
            }
            if i >= j {
                break;
            }
            v.swap(i, j);
            i += 1;
        }
        j += 1;
        let (l, r) = v.split_at_mut(j);
        reference(l, less);
        reference(r, less);
    }

    let mut random = UnitTestRandom::new(42);
    for len in [0usize, 1, 2, 31, 32, 33, 100, 1000, 5000] {
        let data: Vec<(u32, usize)> = (0..len).map(|i| (random.next_u32() % 16, i)).collect();
        let less = |a: &(u32, usize), b: &(u32, usize)| a.0 < b.0;
        let mut expected = data.clone();
        reference(&mut expected, &less);
        let mut actual = data;
        quick_sort_by(&mut actual, less);
        assert_eq!(actual, expected, "len {len}");
    }
}
