//! Port of `Jolt/Core/HashTable.h`.
//!
//! Helper class for implementing an [`UnorderedSet`](crate::unordered_set::UnorderedSet) or
//! [`UnorderedMap`](crate::unordered_map::UnorderedMap).
//! Based on CppCon 2017: Matt Kulukundis "Designing a Fast, Efficient, Cache-friendly Hash Table, Step by Step"
//! See: <https://www.youtube.com/watch?v=ncHmEUmJZf4>
//!
//! The bucket layout, probing, growth, tombstone and rehash logic are identical to C++, so the iteration order (which
//! is observable and matters for determinism) is the same for the same hash function. Never replace this table by
//! `std::collections::HashMap`.
//!
//! Differences with C++ (see `port/DEVIATIONS.md`):
//! - The buckets are a `Vec<Option<KeyValue>>` instead of raw uninitialized memory (no `unsafe`).
//! - Iterators are Rust iterators ([`HashTable::iter`]); a C++ iterator obtained from `find` is the bucket index
//!   returned by [`HashTable::find_index`], `erase(iterator)` is [`HashTable::erase_at`].
//! - The `Hash` template parameter is a [`KeyHasher`], `KeyEqual` is always [`PartialEq`].

use std::fmt;
use std::marker::PhantomData;

use crate::hash_combine::JoltHash;
use crate::math::bvec16::BVec16;
use crate::math::{count_leading_zeros, count_trailing_zeros, get_next_power_of_2};

/// Hash function used by a [`HashTable`] (C++ `Hash` template parameter, a 64-bit hash functor).
pub trait KeyHasher<K: ?Sized> {
    /// Hash `key`.
    fn hash(key: &K) -> u64;
}

/// The default hash function: [`JoltHash`] (C++ `JPH::Hash<Key>`).
#[derive(Clone, Copy, Debug, Default)]
pub struct JoltHasher;

impl<K: JoltHash + ?Sized> KeyHasher<K> for JoltHasher {
    #[inline]
    fn hash(key: &K) -> u64 {
        key.jolt_hash()
    }
}

/// Provides the key of a stored value (C++ `HashTableDetail` template parameter).
pub trait HashTableDetail<K, KV> {
    /// Get key from key value pair
    fn key(key_value: &KV) -> &K;
}

/// Type of sizes and bucket indices (C++ `size_type`).
pub type SizeType = u32;

/// Max load factor is MAX_LOAD_FACTOR_NUMERATOR / MAX_LOAD_FACTOR_DENOMINATOR
const MAX_LOAD_FACTOR_NUMERATOR: u64 = 7;
const MAX_LOAD_FACTOR_DENOMINATOR: u64 = 8;

/// If we can recover this fraction of deleted elements, we'll reshuffle the buckets in place rather than growing the table
const MAX_DELETED_ELEMENTS_NUMERATOR: u64 = 1;
const MAX_DELETED_ELEMENTS_DENOMINATOR: u64 = 8;

/// Values that the control bytes can have
const BUCKET_EMPTY: u8 = 0;
const BUCKET_DELETED: u8 = 0x7f;
const BUCKET_USED: u8 = 0x80; // Lowest 7 bits are lowest 7 bits of the hash value

/// Marks the type parameters that are only used in the methods of [`HashTable`]
type TypeMarker<K, D, H> = PhantomData<fn() -> (K, D, H)>;

/// Hash table with open addressing, used to implement an unordered set / map.
///
/// - `K`: key type
/// - `KV`: stored value (the key itself for a set, a `(K, V)` pair for a map)
/// - `D`: extracts the key from a stored value
/// - `H`: hash function (note should be 64-bits)
pub struct HashTable<K, KV, D, H = JoltHasher> {
    /// The buckets, an array of size max_size (`Some` exactly where the control byte has [`BUCKET_USED`] set)
    data: Vec<Option<KV>>,

    /// Control bytes, an array of size max_size + 15
    control: Vec<u8>,

    /// Number of elements in the table
    size: SizeType,

    /// Max number of elements that can be stored in the table
    max_size: SizeType,

    /// Number of elements we can add to the table before we need to grow
    load_left: SizeType,

    _marker: TypeMarker<K, D, H>,
}

impl<K, KV, D, H> HashTable<K, KV, D, H> {
    /// Get the maximum number of elements that we can support given a number of buckets
    const fn max_load(bucket_count: SizeType) -> SizeType {
        ((MAX_LOAD_FACTOR_NUMERATOR * bucket_count as u64) / MAX_LOAD_FACTOR_DENOMINATOR) as SizeType
    }

    /// Default constructor
    pub const fn new() -> Self {
        Self { data: Vec::new(), control: Vec::new(), size: 0, max_size: 0, load_left: 0, _marker: PhantomData }
    }

    /// Update the control value for a bucket
    #[inline]
    fn set_control_value(&mut self, index: SizeType, value: u8) {
        debug_assert!(index < self.max_size);
        self.control[index as usize] = value;

        // Mirror the first 15 bytes to the 15 bytes beyond max_size
        // Note that this is equivalent to:
        // if (index < 15)
        //   control[index + max_size] = value
        // else
        //   control[index] = value
        // Which performs a needless write if index >= 15 but at least it is branch-less
        self.control[((index.wrapping_sub(15) & (self.max_size - 1)) + 15) as usize] = value;
    }

    /// Read 16 control values starting at `index`
    /// (note that we added 15 bytes at the end of the control values that mirror the first 15 bytes)
    #[inline]
    fn load_control_bytes(&self, index: SizeType) -> BVec16 {
        BVec16::load_byte16(&self.control[index as usize..])
    }

    /// Allocate space for the hash table
    fn allocate_table(&mut self, max_size: SizeType) {
        debug_assert!(self.data.is_empty());

        self.max_size = max_size;
        self.load_left = Self::max_load(max_size);
        self.data = std::iter::repeat_with(|| None).take(max_size as usize).collect();
        self.control = vec![BUCKET_EMPTY; max_size as usize + 15]; // Add 15 bytes to mirror the first 15 bytes of the control values
    }

    /// Get an element by index
    #[inline]
    pub(crate) fn element(&self, index: SizeType) -> &KV {
        self.data[index as usize].as_ref().expect("bucket is used")
    }

    /// Get an element by index
    #[inline]
    pub(crate) fn element_mut(&mut self, index: SizeType) -> &mut KV {
        self.data[index as usize].as_mut().expect("bucket is used")
    }

    /// Construct the element in bucket `index` (which must just have been allocated by `insert_key`)
    #[inline]
    pub(crate) fn set_at(&mut self, index: SizeType, value: KV) {
        debug_assert!(self.data[index as usize].is_none());
        self.data[index as usize] = Some(value);
    }

    /// Reset to the default constructed state, freeing the memory
    fn reset(&mut self) {
        self.data = Vec::new();
        self.control = Vec::new();
        self.size = 0;
        self.max_size = 0;
        self.load_left = 0;
    }

    /// Destroy the entire hash table
    pub fn clear(&mut self) {
        self.reset();
    }

    /// Destroy the entire hash table but keeps the memory allocated
    pub fn clear_and_keep_memory(&mut self) {
        // Destruct elements
        if !self.is_empty() {
            for element in &mut self.data {
                *element = None;
            }
        }
        self.size = 0;

        // If there are elements that are not marked BUCKET_EMPTY, we reset them
        let max_load = Self::max_load(self.max_size);
        if self.load_left != max_load {
            // Reset all control bytes
            self.control.fill(BUCKET_EMPTY);
            self.load_left = max_load;
        }
    }

    /// Iterate over all elements in bucket order
    pub fn iter(&self) -> Iter<'_, KV> {
        Iter { data: self.data.iter(), remaining: self.size }
    }

    /// Iterate over all elements in bucket order (non-const access to key value pair)
    pub fn iter_mut(&mut self) -> IterMut<'_, KV> {
        IterMut { data: self.data.iter_mut(), remaining: self.size }
    }

    /// Number of buckets in the table
    #[inline]
    pub fn bucket_count(&self) -> SizeType {
        self.max_size
    }

    /// Max number of buckets that the table can have
    #[inline]
    pub const fn max_bucket_count(&self) -> SizeType {
        1 << (SizeType::BITS - 1)
    }

    /// Check if there are no elements in the table
    #[inline]
    pub fn is_empty(&self) -> bool {
        self.size == 0
    }

    /// Number of elements in the table
    #[inline]
    pub fn len(&self) -> SizeType {
        self.size
    }

    /// Max number of elements that the table can hold
    #[inline]
    pub const fn max_size(&self) -> SizeType {
        ((self.max_bucket_count() as u64 * MAX_LOAD_FACTOR_NUMERATOR) / MAX_LOAD_FACTOR_DENOMINATOR) as SizeType
    }

    /// Get the max load factor for this table (max number of elements / number of buckets)
    #[inline]
    pub const fn max_load_factor(&self) -> f32 {
        MAX_LOAD_FACTOR_NUMERATOR as f32 / MAX_LOAD_FACTOR_DENOMINATOR as f32
    }

    /// Access the element in bucket `index` (e.g. as returned by [`HashTable::find_index`])
    #[inline]
    pub fn get_at(&self, index: SizeType) -> Option<&KV> {
        self.data.get(index as usize).and_then(Option::as_ref)
    }

    /// Access the element in bucket `index` (e.g. as returned by [`HashTable::find_index`])
    #[inline]
    pub fn get_at_mut(&mut self, index: SizeType) -> Option<&mut KV> {
        self.data.get_mut(index as usize).and_then(Option::as_mut)
    }

    /// Erase an element by bucket index (C++ `erase(const_iterator)`)
    pub fn erase_at(&mut self, index: SizeType) {
        debug_assert!(index < self.max_size && (self.control[index as usize] & BUCKET_USED) != 0);

        // Read 16 control values before and after the current index
        // (note that we added 15 bytes at the end of the control values that mirror the first 15 bytes)
        let control_bytes_before = self.load_control_bytes(index.wrapping_sub(16) & (self.max_size - 1));
        let control_bytes_after = self.load_control_bytes(index);
        let bucket_empty = BVec16::zero();
        let control_empty_before = BVec16::equals(control_bytes_before, bucket_empty).trues() as u32;
        let control_empty_after = BVec16::equals(control_bytes_after, bucket_empty).trues() as u32;

        // If (this index including) there exist 16 consecutive non-empty slots (represented by a bit being 0) then
        // a probe looking for some element needs to continue probing so we cannot mark the bucket as empty
        // but must mark it as deleted instead.
        // Note that we use: CountLeadingZeros(uint16) = CountLeadingZeros(uint32) - 16.
        let control_value =
            if count_leading_zeros(control_empty_before) - 16 + count_trailing_zeros(control_empty_after) < 16 {
                BUCKET_EMPTY
            } else {
                BUCKET_DELETED
            };

        // Mark the bucket as empty/deleted
        self.set_control_value(index, control_value);

        // Destruct the element
        self.data[index as usize] = None;

        // If we marked the bucket as empty we can increase the load left
        if control_value == BUCKET_EMPTY {
            self.load_left += 1;
        }

        // Decrease size
        self.size -= 1;
    }

    /// Swap the contents of two hash tables
    pub fn swap(&mut self, rhs: &mut Self) {
        std::mem::swap(self, rhs);
    }
}

impl<K, KV, D, H> HashTable<K, KV, D, H>
where
    K: PartialEq,
    D: HashTableDetail<K, KV>,
    H: KeyHasher<K>,
{
    /// Get the index and control value for a particular key
    #[inline]
    fn index_and_control_value(&self, key: &K) -> (SizeType, u8) {
        // Calculate hash
        let hash_value = H::hash(key);

        // Split hash into index and control value
        let index = ((hash_value >> 7) as SizeType) & (self.max_size - 1);
        let control = BUCKET_USED | hash_value as u8;
        (index, control)
    }

    /// Grow the table to a new size
    fn grow_table(&mut self, new_max_size: SizeType) {
        // Move the old table to a temporary structure
        let old_data = std::mem::take(&mut self.data);
        let old_control = std::mem::take(&mut self.control);
        self.size = 0;
        self.max_size = 0;
        self.load_left = 0;

        // Allocate new table (all control bytes are reset to empty)
        self.allocate_table(new_max_size);

        // Copy all elements from the old table
        for (element, control) in old_data.into_iter().zip(old_control) {
            if control & BUCKET_USED != 0 {
                let element = element.expect("bucket is used");
                let (inserted, index) = self.insert_key::<true>(D::key(&element));
                debug_assert!(inserted);
                self.data[index as usize] = Some(element);
            }
        }
    }

    /// Insert a key into the map, returns true if the element was inserted, false if it already existed.
    /// The returned index is the index at which the element should be constructed / where it is located.
    pub(crate) fn insert_key<const INSERT_AFTER_GROW: bool>(&mut self, key: &K) -> (bool, SizeType) {
        // Ensure we have enough space
        if self.load_left == 0 {
            // Should not be growing if we're already growing!
            debug_assert!(!INSERT_AFTER_GROW);

            // Decide if we need to clean up all tombstones or if we need to grow the map
            let num_deleted = Self::max_load(self.max_size) - self.size;
            if num_deleted as u64 * MAX_DELETED_ELEMENTS_DENOMINATOR
                > self.max_size as u64 * MAX_DELETED_ELEMENTS_NUMERATOR
            {
                self.rehash(0);
            } else {
                // Grow by a power of 2
                let new_max_size = (self.max_size << 1).max(16);
                assert!(new_max_size >= self.max_size, "Overflow in hash table size, can't grow!");
                self.grow_table(new_max_size);
            }
        }

        // Split hash into index and control value
        let (mut index, control) = self.index_and_control_value(key);

        // Keeps track of the index of the first deleted bucket we found
        const NO_DELETED: SizeType = !0;
        let mut first_deleted_index = NO_DELETED;

        // Linear probing
        let bucket_mask = self.max_size - 1;
        let control16 = BVec16::replicate(control);
        let bucket_empty = BVec16::zero();
        let bucket_deleted = BVec16::replicate(BUCKET_DELETED);
        loop {
            // Read 16 control values (note that we added 15 bytes at the end of the control values that mirror the first 15 bytes)
            let control_bytes = self.load_control_bytes(index);

            // Check if we must find the element before we can insert
            if !INSERT_AFTER_GROW {
                // Check for the control value we're looking for
                // Note that when deleting we can create empty buckets instead of deleted buckets.
                // This means we must unconditionally check all buckets in this batch for equality
                // (also beyond the first empty bucket).
                if let Some(found) = self.find_in_group(control_bytes, control16, index, key) {
                    // Element already exists
                    return (false, found);
                }

                // Check if we're still scanning for deleted buckets
                if first_deleted_index == NO_DELETED {
                    // Check if any buckets have been deleted, if so store the first one
                    let control_deleted = BVec16::equals(control_bytes, bucket_deleted).trues() as u32;
                    if control_deleted != 0 {
                        first_deleted_index = index + count_trailing_zeros(control_deleted);
                    }
                }
            }

            // Check for empty buckets
            let control_empty = BVec16::equals(control_bytes, bucket_empty).trues() as u32;
            if control_empty != 0 {
                // If we found a deleted bucket, use it.
                // It doesn't matter if it is before or after the first empty bucket we found
                // since we will always be scanning in batches of 16 buckets.
                if first_deleted_index == NO_DELETED || INSERT_AFTER_GROW {
                    index += count_trailing_zeros(control_empty);
                    self.load_left -= 1; // Using an empty bucket decreases the load left
                } else {
                    index = first_deleted_index;
                }

                // Make sure that our index is not beyond the end of the table
                index &= bucket_mask;

                // Update control byte
                self.set_control_value(index, control);
                self.size += 1;

                // Return index to newly allocated bucket
                return (true, index);
            }

            // Move to next batch of 16 buckets
            index = (index + 16) & bucket_mask;
        }
    }

    /// Check the 16 buckets starting at `index` with control value `control16` for `key`, returns the bucket index if found
    #[inline]
    fn find_in_group(&self, control_bytes: BVec16, control16: BVec16, index: SizeType, key: &K) -> Option<SizeType> {
        let bucket_mask = self.max_size - 1;
        let mut control_equal = BVec16::equals(control_bytes, control16).trues() as u32;

        // Index within the 16 buckets
        let mut local_index = index;

        // Loop while there's still buckets to process
        while control_equal != 0 {
            // Get the first equal bucket
            let first_equal = count_trailing_zeros(control_equal);

            // Skip to the bucket
            local_index += first_equal;

            // Make sure that our index is not beyond the end of the table
            local_index &= bucket_mask;

            // We found a bucket with same control value
            if D::key(self.element(local_index)) == key {
                return Some(local_index);
            }

            // Skip past this bucket
            control_equal >>= first_equal + 1;
            local_index += 1;
        }

        None
    }

    /// Reserve memory for a certain number of elements
    pub fn reserve(&mut self, max_size: SizeType) {
        // Calculate max size based on load factor
        let max_size = get_next_power_of_2(
            (((MAX_LOAD_FACTOR_DENOMINATOR * max_size as u64) / MAX_LOAD_FACTOR_NUMERATOR) as SizeType).max(16),
        );
        if max_size <= self.max_size {
            return;
        }

        self.grow_table(max_size);
    }

    /// Insert a new element, returns the element and if the element was inserted
    pub fn insert(&mut self, value: KV) -> (&mut KV, bool) {
        let (inserted, index) = self.insert_key::<false>(D::key(&value));
        if inserted {
            self.data[index as usize] = Some(value);
        }
        (self.element_mut(index), inserted)
    }

    /// Find an element, returns the bucket index of the element or `None` if not found (C++ `find` returning an
    /// iterator)
    pub fn find_index(&self, key: &K) -> Option<SizeType> {
        // Check if we have any data
        if self.is_empty() {
            return None;
        }

        // Split hash into index and control value
        let (mut index, control) = self.index_and_control_value(key);

        // Linear probing
        let bucket_mask = self.max_size - 1;
        let control16 = BVec16::replicate(control);
        let bucket_empty = BVec16::zero();
        loop {
            // Read 16 control values
            // (note that we added 15 bytes at the end of the control values that mirror the first 15 bytes)
            let control_bytes = self.load_control_bytes(index);

            // Check for the control value we're looking for
            // Note that when deleting we can create empty buckets instead of deleted buckets.
            // This means we must unconditionally check all buckets in this batch for equality
            // (also beyond the first empty bucket).
            if let Some(found) = self.find_in_group(control_bytes, control16, index, key) {
                // Element found
                return Some(found);
            }

            // Check for empty buckets
            let control_empty = BVec16::equals(control_bytes, bucket_empty).trues() as u32;
            if control_empty != 0 {
                // An empty bucket was found, we didn't find the element
                return None;
            }

            // Move to next batch of 16 buckets
            index = (index + 16) & bucket_mask;
        }
    }

    /// Find an element, returns `None` if not found
    #[inline]
    pub fn find(&self, key: &K) -> Option<&KV> {
        self.find_index(key).map(|index| self.element(index))
    }

    /// Find an element, returns `None` if not found
    #[inline]
    pub fn find_mut(&mut self, key: &K) -> Option<&mut KV> {
        self.find_index(key).map(|index| self.element_mut(index))
    }

    /// Erase an element by key, returns the number of elements erased (0 or 1)
    pub fn erase(&mut self, key: &K) -> SizeType {
        match self.find_index(key) {
            None => 0,
            Some(index) => {
                self.erase_at(index);
                1
            }
        }
    }

    /// In place re-hashing of all elements in the table. Removes all BUCKET_DELETED elements
    /// The std version takes a bucket count, but we just re-hash to the same size.
    pub fn rehash(&mut self, _bucket_count: SizeType) {
        // Update the control value for all buckets
        for control in &mut self.control[..self.max_size as usize] {
            match *control {
                // Deleted buckets become empty
                BUCKET_DELETED => *control = BUCKET_EMPTY,
                // Remains empty
                BUCKET_EMPTY => {}
                // Mark all occupied as deleted, to indicate it needs to move to the correct place
                _ => *control = BUCKET_DELETED,
            }
        }

        // Replicate control values to the last 15 entries
        let max_size = self.max_size as usize;
        for i in 0..15.min(max_size) {
            self.control[max_size + i] = self.control[i];
        }

        // Loop over all elements that have been 'deleted' and move them to their new spot
        let bucket_used = BVec16::replicate(BUCKET_USED);
        let bucket_mask = self.max_size.wrapping_sub(1);
        let probe_mask = bucket_mask & !0b1111u32; // Mask out lower 4 bits because we test 16 buckets at a time
        for src in 0..self.max_size {
            if self.control[src as usize] == BUCKET_DELETED {
                loop {
                    // Split hash into index and control value
                    let (src_index, src_control) = self.index_and_control_value(D::key(self.element(src)));

                    // Linear probing
                    let mut dst = src_index;
                    loop {
                        // Check if any buckets are free
                        let control_bytes = self.load_control_bytes(dst);
                        let control_free = (BVec16::and(control_bytes, bucket_used).trues() as u32) ^ 0xffff;
                        if control_free != 0 {
                            // Select this bucket as destination
                            dst += count_trailing_zeros(control_free);
                            dst &= bucket_mask;
                            break;
                        }

                        // Move to next batch of 16 buckets
                        dst = (dst + 16) & bucket_mask;
                    }

                    // Check if we stay in the same probe group
                    if (dst.wrapping_sub(src_index) & probe_mask) == (src.wrapping_sub(src_index) & probe_mask) {
                        // We stay in the same group, we can stay where we are
                        self.set_control_value(src, src_control);
                        break;
                    } else if self.control[dst as usize] == BUCKET_EMPTY {
                        // There's an empty bucket, move us there
                        self.set_control_value(dst, src_control);
                        self.set_control_value(src, BUCKET_EMPTY);
                        self.data[dst as usize] = self.data[src as usize].take();
                        break;
                    } else {
                        // There's an element in the bucket we want to move to, swap them
                        debug_assert!(self.control[dst as usize] == BUCKET_DELETED);
                        self.set_control_value(dst, src_control);
                        self.data.swap(src as usize, dst as usize);
                        // Iterate again with the same source bucket
                    }
                }
            }
        }

        // Reinitialize load left
        self.load_left = Self::max_load(self.max_size) - self.size;
    }
}

impl<K, KV, D, H> Default for HashTable<K, KV, D, H> {
    fn default() -> Self {
        Self::new()
    }
}

/// Copy constructor (an empty table copies to an unallocated table, like C++)
impl<K, KV: Clone, D, H> Clone for HashTable<K, KV, D, H> {
    fn clone(&self) -> Self {
        if self.is_empty() {
            return Self::new();
        }

        Self {
            data: self.data.clone(),
            control: self.control.clone(),
            size: self.size,
            max_size: self.max_size,
            load_left: self.load_left,
            _marker: PhantomData,
        }
    }
}

impl<K, KV: fmt::Debug, D, H> fmt::Debug for HashTable<K, KV, D, H> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_set().entries(self.iter()).finish()
    }
}

impl<'a, K, KV, D, H> IntoIterator for &'a HashTable<K, KV, D, H> {
    type Item = &'a KV;
    type IntoIter = Iter<'a, KV>;

    fn into_iter(self) -> Self::IntoIter {
        self.iter()
    }
}

impl<'a, K, KV, D, H> IntoIterator for &'a mut HashTable<K, KV, D, H> {
    type Item = &'a mut KV;
    type IntoIter = IterMut<'a, KV>;

    fn into_iter(self) -> Self::IntoIter {
        self.iter_mut()
    }
}

/// Iterator over the elements of a [`HashTable`] in bucket order (C++ `const_iterator`)
#[derive(Debug)]
pub struct Iter<'a, KV> {
    data: std::slice::Iter<'a, Option<KV>>,
    remaining: SizeType,
}

impl<'a, KV> Iterator for Iter<'a, KV> {
    type Item = &'a KV;

    #[inline]
    fn next(&mut self) -> Option<&'a KV> {
        // Skip buckets that are not in use (exactly the buckets whose control byte does not have BUCKET_USED set)
        let element = self.data.by_ref().flatten().next()?;
        self.remaining -= 1;
        Some(element)
    }

    #[inline]
    fn size_hint(&self) -> (usize, Option<usize>) {
        (self.remaining as usize, Some(self.remaining as usize))
    }
}

impl<KV> ExactSizeIterator for Iter<'_, KV> {}

/// Iterator over the elements of a [`HashTable`] in bucket order (C++ `iterator`)
#[derive(Debug)]
pub struct IterMut<'a, KV> {
    data: std::slice::IterMut<'a, Option<KV>>,
    remaining: SizeType,
}

impl<'a, KV> Iterator for IterMut<'a, KV> {
    type Item = &'a mut KV;

    #[inline]
    fn next(&mut self) -> Option<&'a mut KV> {
        let element = self.data.by_ref().flatten().next()?;
        self.remaining -= 1;
        Some(element)
    }

    #[inline]
    fn size_hint(&self) -> (usize, Option<usize>) {
        (self.remaining as usize, Some(self.remaining as usize))
    }
}

impl<KV> ExactSizeIterator for IterMut<'_, KV> {}
