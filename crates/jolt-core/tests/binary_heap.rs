//! Port of UnitTests/Core/BinaryHeapTest.cpp

use jolt_core::binary_heap::{binary_heap_pop, binary_heap_push};
use jolt_test_utils::UnitTestRandom;

#[test]
fn test_binary_heap() {
    // Add some numbers
    let mut array: Vec<i32> = Vec::with_capacity(1100);
    for i in 0..1000 {
        array.push(i);
    }

    // Ensure we have some duplicates
    for i in (0..1000).step_by(10) {
        array.push(i);
    }

    // Shuffle the array
    let mut random = UnitTestRandom::new(123);
    random.shuffle(&mut array);

    // Add the numbers to the heap
    let mut heap: Vec<i32> = Vec::new();
    for &i in &array {
        heap.push(i);
        binary_heap_push(&mut heap, |a, b| a < b);
    }

    // Check that the heap is sorted
    let mut last = i32::MAX;
    let mut seen = [0; 1000];
    while !heap.is_empty() {
        binary_heap_pop(&mut heap, |a, b| a < b);
        let current = heap.pop().unwrap();
        seen[current as usize] += 1;
        assert!(seen[current as usize] <= if current % 10 == 0 { 2 } else { 1 });
        assert!(current <= last);
        last = current;
    }
}
