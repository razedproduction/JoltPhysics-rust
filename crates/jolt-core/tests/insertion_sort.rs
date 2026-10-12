//! Port of UnitTests/Core/InsertionSortTest.cpp

use jolt_core::insertion_sort::{insertion_sort, insertion_sort_by};
use jolt_test_utils::UnitTestRandom;

#[test]
fn test_ordered_array() {
    let mut array: Vec<i32> = (0..10).collect();

    insertion_sort(&mut array);

    for i in 0..10 {
        assert_eq!(array[i as usize], i);
    }
}

#[test]
fn test_ordered_array_comparator() {
    let mut array: Vec<i32> = (0..100).collect();

    insertion_sort_by(&mut array, |a, b| a > b);

    for i in 0..100 {
        assert_eq!(array[i as usize], 99 - i);
    }
}

#[test]
fn test_reversed_array() {
    let mut array: Vec<i32> = (0..10).map(|i| 9 - i).collect();

    insertion_sort(&mut array);

    for i in 0..10 {
        assert_eq!(array[i as usize], i);
    }
}

#[test]
fn test_random_array() {
    let mut random = UnitTestRandom::default();

    let mut array: Vec<u32> = Vec::new();
    for _ in 0..100 {
        let value = random.next_u32();

        // Insert value at beginning
        array.insert(0, value);

        // Insert value at end
        array.push(value);
    }

    insertion_sort(&mut array);

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
    insertion_sort(&mut array);
    assert!(array.is_empty());
}

#[test]
fn test1_element_array() {
    let mut array = vec![1];
    insertion_sort(&mut array);
    assert_eq!(array[0], 1);
}

#[test]
fn test2_element_array() {
    let mut array = vec![2, 1];
    insertion_sort(&mut array);
    assert_eq!(array[0], 1);
    assert_eq!(array[1], 2);
}
