//! Port of `Jolt/Core/BinaryHeap.h`.
//!
//! The C++ functions take an iterator range; here they take a slice (`[inBegin, inEnd)`).

/// Push a new element into a binary max-heap.
/// `slice[..len - 1]` must be a a valid heap. Element `len - 1` will be inserted into the heap. The heap will be
/// `slice` after this call.
/// `pred` is a function that returns true if the first element is less or equal than the second element.
/// See: <https://en.wikipedia.org/wiki/Binary_heap>
pub fn binary_heap_push<T, F>(slice: &mut [T], mut pred: F)
where
    F: FnMut(&T, &T) -> bool,
{
    // New heap size
    let count = slice.len();

    // Start from the last element (an empty heap is left alone, like C++ where current = -1)
    let mut current = count.saturating_sub(1);
    while current > 0 {
        // Get parent element
        let parent = (current - 1) >> 1;

        // Sort them so that the parent is larger than the child
        if pred(&slice[parent], &slice[current]) {
            slice.swap(parent, current);
            current = parent;
        } else {
            // When there's no change, we're done
            break;
        }
    }
}

/// Pop an element from a binary max-heap.
/// `slice` must be a valid heap. The largest element will be removed from the heap. The heap will be
/// `slice[..len - 1]` after this call.
/// `pred` is a function that returns true if the first element is less or equal than the second element.
/// See: <https://en.wikipedia.org/wiki/Binary_heap>
pub fn binary_heap_pop<T, F>(slice: &mut [T], mut pred: F)
where
    F: FnMut(&T, &T) -> bool,
{
    // Begin by moving the highest element to the end, this is the popped element
    let last = slice.len() - 1;
    slice.swap(last, 0);

    // New heap size
    let count = last;

    // Start from the root
    let mut largest = 0;
    loop {
        // Get first child
        let mut child = (largest << 1) + 1;

        // Check if we're beyond the end of the heap, if so the 2nd child is also beyond the end
        if child >= count {
            break;
        }

        // Remember the largest element from the previous iteration
        let prev_largest = largest;

        // Check if first child is bigger, if so select it
        if pred(&slice[largest], &slice[child]) {
            largest = child;
        }

        // Switch to the second child
        child += 1;

        // Check if second child is bigger, if so select it
        if child < count && pred(&slice[largest], &slice[child]) {
            largest = child;
        }

        // If there was no change, we're done
        if prev_largest == largest {
            break;
        }

        // Swap element
        slice.swap(prev_largest, largest);
    }
}
