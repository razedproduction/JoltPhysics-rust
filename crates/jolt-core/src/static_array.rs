//! Port of `Jolt/Core/StaticArray.h`.
//!
//! Backed by [`arrayvec::ArrayVec`] instead of a raw union buffer (no `unsafe`). Derefs to a slice, which provides
//! indexing (`operator []`, `at`), `first`/`last` (`front`/`back`), `iter` (`begin`/`end`) and `as_ptr` (`data`).

use std::ops::{Deref, DerefMut, Range};

use arrayvec::ArrayVec;

use crate::hash_combine::{JoltHash, hash_combine};

/// Simple variable length array backed by a fixed size buffer
#[derive(Clone, PartialEq, Eq)]
#[must_use]
pub struct StaticArray<T, const N: usize> {
    elements: ArrayVec<T, N>,
}

impl<T, const N: usize> StaticArray<T, N> {
    /// Maximum amount of elements the array can hold
    pub const CAPACITY: usize = N;

    /// Default constructor
    #[inline]
    pub const fn new() -> Self {
        Self { elements: ArrayVec::new_const() }
    }

    /// Constructor from initializer list
    pub fn from_slice(list: &[T]) -> Self
    where
        T: Clone,
    {
        debug_assert!(list.len() <= N);
        let mut array = Self::new();
        for v in list {
            array.push(v.clone());
        }
        array
    }

    /// Destruct all elements and set length to zero
    #[inline]
    pub fn clear(&mut self) {
        self.elements.clear();
    }

    /// Add element to the back of the array (C++ `push_back` / `emplace_back`)
    #[inline]
    pub fn push(&mut self, element: T) {
        debug_assert!(self.elements.len() < N);
        self.elements.push(element);
    }

    /// Remove element from the back of the array (C++ `pop_back`), returns it
    #[inline]
    pub fn pop(&mut self) -> Option<T> {
        debug_assert!(!self.elements.is_empty());
        self.elements.pop()
    }

    /// Returns true if there are no elements in the array
    #[inline]
    pub fn is_empty(&self) -> bool {
        self.elements.is_empty()
    }

    /// Returns amount of elements in the array
    #[inline]
    pub fn len(&self) -> usize {
        self.elements.len()
    }

    /// Returns maximum amount of elements the array can hold
    #[inline]
    pub const fn capacity(&self) -> usize {
        N
    }

    /// Resize array to new length (new elements are default constructed)
    pub fn resize(&mut self, new_size: usize)
    where
        T: Default,
    {
        debug_assert!(new_size <= N);
        while self.elements.len() < new_size {
            self.elements.push(T::default());
        }
        self.elements.truncate(new_size);
    }

    /// Remove one element from the array
    #[inline]
    pub fn erase(&mut self, index: usize) {
        debug_assert!(index < self.elements.len());
        self.elements.remove(index);
    }

    /// Remove multiple element from the array
    #[inline]
    pub fn erase_range(&mut self, range: Range<usize>) {
        debug_assert!(range.end <= self.elements.len());
        self.elements.drain(range);
    }

    /// Assignment operator with static array of different max length
    pub fn assign<const M: usize>(&mut self, rhs: &StaticArray<T, M>)
    where
        T: Clone,
    {
        debug_assert!(rhs.len() <= N);
        self.clear();
        for v in rhs.iter() {
            self.push(v.clone());
        }
    }
}

impl<T, const N: usize> Default for StaticArray<T, N> {
    #[inline]
    fn default() -> Self {
        Self::new()
    }
}

impl<T, const N: usize> Deref for StaticArray<T, N> {
    type Target = [T];

    #[inline]
    fn deref(&self) -> &[T] {
        &self.elements
    }
}

impl<T, const N: usize> DerefMut for StaticArray<T, N> {
    #[inline]
    fn deref_mut(&mut self) -> &mut [T] {
        &mut self.elements
    }
}

impl<T: std::fmt::Debug, const N: usize> std::fmt::Debug for StaticArray<T, N> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_list().entries(self.iter()).finish()
    }
}

impl<'a, T, const N: usize> IntoIterator for &'a StaticArray<T, N> {
    type Item = &'a T;
    type IntoIter = std::slice::Iter<'a, T>;

    #[inline]
    fn into_iter(self) -> Self::IntoIter {
        self.iter()
    }
}

impl<'a, T, const N: usize> IntoIterator for &'a mut StaticArray<T, N> {
    type Item = &'a mut T;
    type IntoIter = std::slice::IterMut<'a, T>;

    #[inline]
    fn into_iter(self) -> Self::IntoIter {
        self.iter_mut()
    }
}

/// Get hash for this array
impl<T: JoltHash, const N: usize> JoltHash for StaticArray<T, N> {
    fn jolt_hash(&self) -> u64 {
        // Hash length first
        let mut ret = (self.len() as u32).jolt_hash();

        // Then hash elements
        for element in self.iter() {
            hash_combine(&mut ret, element);
        }

        ret
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn push_erase_resize() {
        let mut a = StaticArray::<u32, 8>::from_slice(&[1, 2, 3, 4, 5]);
        assert_eq!(a.capacity(), 8);
        a.erase(1);
        assert_eq!(&a[..], &[1, 3, 4, 5]);
        a.erase_range(1..3);
        assert_eq!(&a[..], &[1, 5]);
        a.resize(4);
        assert_eq!(&a[..], &[1, 5, 0, 0]);
        a.resize(1);
        assert_eq!(a.pop(), Some(1));
        assert!(a.is_empty());

        let mut b = StaticArray::<u32, 4>::new();
        b.assign(&StaticArray::<u32, 2>::from_slice(&[7, 8]));
        assert_eq!(&b[..], &[7, 8]);

        let mut expected = 2u32.jolt_hash();
        hash_combine(&mut expected, &7u32);
        hash_combine(&mut expected, &8u32);
        assert_eq!(b.jolt_hash(), expected);
    }
}
